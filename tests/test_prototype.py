"""Deterministic component and encoded-video regression tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from fishtrack.cli import main
from fishtrack.detection import DetectionConfig, MotionDetector, extract_detections
from fishtrack.rendering import annotate
from fishtrack.tracking import Tracker
from fishtrack.video import ProcessingConfig, process_video, read_metadata


def candidate(x: int = 30):
    """Build a known rectangular candidate."""
    mask = np.zeros((120, 160), np.uint8)
    cv2.rectangle(mask, (x, 40), (x + 20, 55), 255, -1)
    return extract_detections(mask, DetectionConfig())[0]


def test_filtering_and_centroid():
    """Reject tiny, oversized and elongated regions while retaining valid shapes."""
    detection = candidate()
    assert detection.centroid == (40, 47.5)
    assert detection.box == (30, 40, 21, 16)
    mask = np.zeros((120, 160), np.uint8)
    mask[5:7, 5:7] = 255
    mask[20:23, 10:110] = 255
    assert extract_detections(mask, DetectionConfig()) == []
    mask[:] = 255
    assert extract_detections(mask, DetectionConfig(max_area=500)) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_area=0),
        dict(max_area=2),
        dict(motion_threshold=256),
        dict(motion_threshold=float("nan")),
        dict(max_aspect_ratio=0),
    ],
)
def test_invalid_detection_config(kwargs):
    """Reject unsafe numerical thresholds."""
    with pytest.raises(ValueError):
        DetectionConfig(**kwargs)


def test_tracking_lifecycle_and_one_to_one():
    """Retain IDs across short gaps and expire them after the configured tolerance."""
    tracker = Tracker(max_distance=30, max_missing=1)
    identity = np.eye(2, 3)
    assert tracker.update([candidate()], identity)[0].identifier == 1
    tracker.update([], identity)
    track = tracker.update([candidate(35)], identity)[0]
    assert track.identifier == 1 and track.hits == 2 and track.missing == 0
    tracks = tracker.update([candidate(36), candidate(38)], identity)
    assert len(tracks) == 2 and len({t.identifier for t in tracks}) == 2
    tracker.update([], identity)
    assert tracker.update([], identity) == []
    assert tracker.update([candidate()], identity)[0].identifier == 3


def test_camera_transform_tracking_and_rendering():
    """Associate through camera translation and leave input frames untouched."""
    tracker = Tracker(max_distance=5)
    tracker.update([candidate()], np.eye(2, 3))
    tracks = tracker.update([candidate(50)], np.array([[1.0, 0, 20], [0, 1, 0]]))
    assert len(tracks) == 1 and tracks[0].identifier == 1
    frame = np.zeros((120, 160, 3), np.uint8)
    assert annotate(frame, tracks).any()
    assert not frame.any()
    tracker.update([], np.eye(2, 3))
    assert not annotate(frame, tracks).any()


def test_compensation_reduces_background_motion():
    """A known camera pan should produce far less mask area after alignment."""
    rng = np.random.default_rng(13)
    first = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    second = cv2.warpAffine(first, np.array([[1.0, 0, 5], [0, 1, 3]]), (320, 240))
    compensated = MotionDetector(DetectionConfig(motion_threshold=12, local_motion=False))
    raw = MotionDetector(DetectionConfig(motion_threshold=12, compensate=False, local_motion=False))
    compensated.update(first)
    raw.update(first)
    _, aligned_mask, _, matrix, reliable = compensated.update(second)
    _, raw_mask, _, _, _ = raw.update(second)
    assert reliable
    assert np.allclose(matrix[:, 2], [5, 3], atol=0.5)
    assert cv2.countNonZero(aligned_mask) < cv2.countNonZero(raw_mask) * 0.1


def test_motion_threshold_and_first_frame():
    """First/static frames are empty; a moving shape generates candidates."""
    detector = MotionDetector(DetectionConfig(compensate=False, local_motion=False))
    frame = np.zeros((120, 160, 3), np.uint8)
    assert not detector.update(frame)[0]
    assert not detector.update(frame)[0]
    cv2.rectangle(frame, (40, 40), (65, 55), (255, 255, 255), -1)
    assert detector.update(frame)[0]


def make_video(path: Path, frames: int = 12):
    """Encode a small deterministic moving-shape MP4."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((120, 160, 3), np.uint8)
        cv2.ellipse(frame, (30 + index * 4, 60), (12, 7), 0, 0, 360, (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def test_video_roundtrip_and_source_safety(tmp_path):
    """Export a limited clip with diagnostics and preserve the exact source bytes."""
    source, output = tmp_path / "input.mp4", tmp_path / "output.mp4"
    make_video(source)
    original = source.read_bytes()
    count = process_video(
        source,
        output,
        DetectionConfig(),
        ProcessingConfig(start=0.25, duration=0.5),
        tmp_path / "debug",
    )
    assert count == 6
    capture = cv2.VideoCapture(str(output))
    metadata = read_metadata(capture)
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert (metadata.width, metadata.height, metadata.fps, decoded) == (160, 120, 12, 6)
    assert len(list((tmp_path / "debug").glob("*-mask.png"))) == 6
    assert source.read_bytes() == original
    with pytest.raises(ValueError):
        process_video(source, source, DetectionConfig(), ProcessingConfig())
    with pytest.raises(ValueError):
        process_video(source, output, DetectionConfig(), ProcessingConfig())
    with pytest.raises(ValueError):
        process_video(source, tmp_path / "empty.mp4", DetectionConfig(), ProcessingConfig(start=5))
    assert not (tmp_path / "empty.mp4").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--min-area", "-1"],
        ["--duration", "0"],
        ["--start", "nan"],
        ["--max-frames", "0"],
        ["--max-missing", "-1"],
    ],
)
def test_cli_validation(arguments):
    """Invalid CLI settings produce argparse errors before any processing."""
    with pytest.raises(SystemExit) as error:
        main(["--input", "input.mp4", "--output", "output.mp4", *arguments])
    assert error.value.code == 2


def test_cli_missing_input(tmp_path):
    """Missing files report a controlled failure."""
    assert (
        main(["--input", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "out.mp4")]) == 1
    )


def test_featureless_camera_fallback():
    """Low-texture footage falls back to identity with an explicit failed-fit flag."""
    detector = MotionDetector(DetectionConfig())
    frame = np.zeros((120, 160, 3), np.uint8)
    detector.update(frame)
    detections, mask, _, matrix, reliable = detector.update(frame)
    assert not reliable and not detections and not mask.any()
    assert np.array_equal(matrix, np.eye(2, 3))


def test_corrupt_video_does_not_publish(tmp_path):
    """Unusable input fails without publishing a misleading output."""
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not a video")
    output = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="Cannot open"):
        process_video(source, output, DetectionConfig(), ProcessingConfig())
    assert not output.exists()


def test_metadata_validation():
    """An unopened video has invalid metadata and is rejected explicitly."""
    capture = cv2.VideoCapture()
    try:
        with pytest.raises(ValueError, match="invalid dimensions"):
            read_metadata(capture)
    finally:
        capture.release()


def test_consecutive_confirmation_restarts_after_gap():
    """Total sightings cannot confirm an intermittently detected track."""
    tracker = Tracker()
    identity = np.eye(2, 3)
    tracker.update([candidate()], identity)
    tracker.update([candidate(31)], identity)
    tracker.update([], identity)
    track = tracker.update([candidate(32)], identity)[0]
    assert track.hits == 3 and track.consecutive_hits == 1
    frame = np.zeros((120, 160, 3), np.uint8)
    assert not annotate(frame, [track], min_hits=3).any()
    tracker.update([candidate(33)], identity)
    tracker.update([candidate(34)], identity)
    assert track.consecutive_hits == 3
    assert annotate(frame, [track], min_hits=3).any()


def test_scene_change_suppresses_and_recovers():
    """Abrupt transitions produce no candidates and reinitialize the reference."""
    detector = MotionDetector(DetectionConfig(compensate=False, local_motion=False))
    detector.update(np.zeros((120, 160, 3), np.uint8))
    bright = np.full((120, 160, 3), 255, np.uint8)
    detections, mask, _, _, _ = detector.update(bright)
    assert detector.suppression_reason == "scene_change"
    assert not detections and not mask.any()
    assert not detector.update(bright)[0]
    assert detector.suppression_reason is None


def test_failed_alignment_suppresses_candidates():
    """A featureless camera estimate must not publish frame-difference candidates."""
    detector = MotionDetector(DetectionConfig(local_motion=False))
    frame = np.zeros((120, 160, 3), np.uint8)
    detector.update(frame)
    cv2.rectangle(frame, (40, 40), (65, 55), (255, 255, 255), -1)
    detections, mask, _, _, _ = detector.update(frame)
    assert not detections and not mask.any()
    assert detector.suppression_reason == "unreliable_alignment"


def test_local_flow_retains_independent_motion_and_rejects_camera_pan():
    """Dense local flow must reject camera-driven objects yet retain a moving patch."""
    from fishtrack.detection import filter_local_motion

    rng = np.random.default_rng(24)
    previous = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (3, 3), 0)
    camera = cv2.warpAffine(previous, np.array([[1.0, 0, 3], [0, 1, 0]]), (240, 180))
    mask = np.zeros_like(previous)
    mask[65:100, 95:135] = 255
    detections = extract_detections(mask, DetectionConfig())
    assert not filter_local_motion(previous, camera, detections, 1.5)
    # Move a textured object four pixels against an otherwise stationary background.
    stationary = previous.copy()
    moving = previous.copy()
    stationary[65:100, 91:131] = previous[110:145, 40:80]
    moving[65:100, 95:135] = previous[110:145, 40:80]
    assert filter_local_motion(stationary, moving, detections, 1.5)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_residual_threshold(value):
    """Residual speed thresholds must be finite and positive."""
    with pytest.raises(ValueError):
        DetectionConfig(min_residual_motion=value)


def test_local_flow_rejects_two_background_depths():
    """Different background planes can move differently without being independent objects."""
    from fishtrack.detection import filter_local_motion

    rng = np.random.default_rng(42)
    previous = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (3, 3), 0)
    current = previous.copy()
    for top, bottom, shift in ((0, 120, 2), (120, 240, 5)):
        current[top:bottom] = cv2.warpAffine(
            previous[top:bottom], np.array([[1.0, 0, shift], [0, 1, 0]]), (320, bottom - top)
        )
    mask = np.zeros_like(previous)
    mask[40:70, 110:150] = 255
    mask[160:190, 110:150] = 255
    detections = extract_detections(mask, DetectionConfig())
    assert len(detections) == 2
    assert not filter_local_motion(previous, current, detections, 1.5)


def test_unreliable_motion_opt_out():
    """The explicit comparison option restores unfiltered failed-fit behaviour."""
    detector = MotionDetector(DetectionConfig(local_motion=False, suppress_unreliable=False))
    frame = np.zeros((120, 160, 3), np.uint8)
    detector.update(frame)
    cv2.rectangle(frame, (40, 40), (65, 55), (255, 255, 255), -1)
    assert detector.update(frame)[0]
    assert detector.suppression_reason is None


@pytest.mark.parametrize("extra_frames, succeeds", [(0, True), (3, True), (4, False), (30, False)])
def test_inaccurate_tail_metadata(tmp_path, monkeypatch, caplog, extra_frames, succeeds):
    """Tolerate small metadata overcounts but reject substantial premature decoding stops."""
    from fishtrack import video

    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    make_video(source)
    original_read_metadata = video.read_metadata

    def inaccurate_metadata(capture):
        """Simulate a backend overstating the source length."""
        metadata = original_read_metadata(capture)
        return video.VideoMetadata(
            metadata.width, metadata.height, metadata.fps, metadata.frame_count + extra_frames
        )

    monkeypatch.setattr(video, "read_metadata", inaccurate_metadata)
    if succeeds:
        assert process_video(source, output, DetectionConfig(), ProcessingConfig()) == 12
        capture = cv2.VideoCapture(str(output))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        assert decoded == 12
        assert ("tail discrepancy" in caplog.text) == (extra_frames > 0)
    else:
        with pytest.raises(OSError, match="stopped early at frame 12"):
            process_video(source, output, DetectionConfig(), ProcessingConfig())
        assert not output.exists()
        assert set(tmp_path.glob("*.mp4")) == {source}
