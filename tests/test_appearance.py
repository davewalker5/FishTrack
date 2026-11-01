"""Regression tests for following independently of motion-based acquisition."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from fishtrack.appearance import optical_prediction
from fishtrack.cli import main
from fishtrack.detection import DetectionConfig
from fishtrack.rendering import annotate
from fishtrack.tracking import Tracker, box_detection
from fishtrack.video import ProcessingConfig, process_video


def object_frame(index: int = 0, visible: bool = True) -> np.ndarray:
    """Render the same distinctive textured object translated against a flat background."""
    frame = np.full((180, 240, 3), (60, 90, 60), dtype=np.uint8)
    if visible:
        patch = np.random.default_rng(6).integers(40, 240, (40, 36, 3), dtype=np.uint8)
        patch[:, :, 0] //= 3
        x, y = 50 + index * 3, 60 + index
        frame[y : y + 40, x : x + 36] = patch
    return frame


def test_manual_track_follows_without_any_detections():
    """A selected object keeps its ID, position and visibility through detector gaps."""
    tracker = Tracker()
    first = object_frame()
    tracker.seed(first, ((50, 60, 36, 40),))
    assert tracker.tracks[0].confirmed and tracker.tracks[0].manual
    for index in range(1, 9):
        frame = object_frame(index)
        tracks = tracker.update([], np.eye(2, 3), frame)
        assert len(tracks) == 1
        track = tracks[0]
        assert track.identifier == 1 and track.missing == 0 and track.propagated
        assert abs(track.position[0] - (67.5 + index * 3)) < 8
        assert abs(track.position[1] - (79.5 + index)) < 8
        assert np.any(annotate(frame, tracks) != frame)
    assert track.hits == 1 and track.followed_frames == 8


def test_automatic_confirmation_starts_visual_following():
    """Three motion detections can acquire an object which then needs no detections."""
    tracker = Tracker()
    for index in range(3):
        detection = box_detection((50 + index * 3, 60 + index, 36, 40))
        tracks = tracker.update([detection], np.eye(2, 3), object_frame(index))
    assert tracks[0].confirmed and tracks[0].follower is not None
    assert not tracks[0].manual
    for index in range(3, 7):
        tracks = tracker.update([], np.eye(2, 3), object_frame(index))
        assert tracks[0].identifier == 1 and tracks[0].missing == 0
        assert tracks[0].propagated


def test_visual_track_does_not_duplicate_current_candidates():
    """Candidates inside a followed object do not acquire a second ID."""
    tracker = Tracker()
    tracker.seed(object_frame(), ((50, 60, 36, 40),))
    tracks = tracker.update([box_detection((53, 61, 36, 40))], np.eye(2, 3), object_frame(1))
    assert len(tracks) == 1 and tracks[0].identifier == 1


def test_lost_object_stops_drawing_and_expires():
    """Disappearance must never leave an unobserved, camera-predicted box on the video."""
    tracker = Tracker(max_missing=2)
    tracker.seed(object_frame(), ((50, 60, 36, 40),))
    blank = object_frame(visible=False)
    for _ in range(2):
        tracks = tracker.update([], np.eye(2, 3), blank)
        assert tracks[0].missing > 0
        assert np.array_equal(annotate(blank, tracks), blank)
    assert tracker.update([], np.eye(2, 3), blank) == []


def test_failed_follower_cannot_jump_to_motion_contour():
    """A confirmed identity cannot be reassigned to a nearby unrelated moving region."""
    tracker = Tracker()
    tracker.seed(object_frame(), ((50, 60, 36, 40),))
    tracks = tracker.update(
        [box_detection((55, 65, 30, 30))], np.eye(2, 3), object_frame(visible=False)
    )
    original = next(track for track in tracks if track.identifier == 1)
    assert original.missing == 1
    assert original.detection.box == (50, 60, 36, 40)
    assert len(tracks) == 2 and tracks[1].identifier == 2


def test_optical_prediction_tracks_local_translation():
    """Forward/backward checked feature flow predicts the moving object position."""
    previous = cv2.cvtColor(object_frame(), cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(object_frame(1), cv2.COLOR_BGR2GRAY)
    detection = box_detection((50, 60, 36, 40))
    prediction = optical_prediction(previous, current, detection)
    assert prediction is not None
    assert np.allclose(prediction, (70.5, 80.5), atol=1)


def test_scene_reset_preserves_identifier_uniqueness():
    """Discarding old visual models never recycles an ID in the same run."""
    tracker = Tracker()
    tracker.seed(object_frame(), ((50, 60, 36, 40),))
    tracker.reset()
    assert not tracker.tracks and tracker.previous_gray is None
    tracker.seed(object_frame(), ((50, 60, 36, 40),))
    assert tracker.tracks[0].identifier == 2


@pytest.mark.parametrize(
    "box", [(-1, 0, 20, 20), (0, 0, 7, 20), (230, 170, 20, 20), (0, 0, 20, 20)]
)
def test_invalid_or_untextured_manual_regions(box):
    """Reject invalid, off-image or flat selections without adding partial seeds."""
    tracker = Tracker()
    with pytest.raises(ValueError):
        tracker.seed(object_frame(), (box,))
    assert not tracker.tracks


@pytest.mark.parametrize("value", [0, -1, 1.1, float("nan")])
def test_invalid_tracking_threshold(value):
    """Invalid response thresholds fail before video processing."""
    with pytest.raises(ValueError):
        ProcessingConfig(tracking_threshold=value)


def test_manual_roi_needs_visual_tracking():
    """Explicitly contradictory manual-selection settings are rejected."""
    with pytest.raises(ValueError, match="require appearance"):
        ProcessingConfig(rois=((50, 60, 36, 40),), appearance=False)


@pytest.mark.parametrize("automatic", [False, True])
def test_reference_and_manual_tracking_video(tmp_path: Path, monkeypatch, automatic):
    """Manual tracking survives poor global alignment and exports usable diagnostics."""
    from fishtrack import detection

    def failed_alignment(previous, current):
        """Simulate a globally unreliable camera estimate throughout the clip."""
        return np.eye(2, 3), False

    monkeypatch.setattr(detection, "estimate_motion", failed_alignment)
    source = tmp_path / "source.mp4"
    output = tmp_path / "tracked.mp4"
    reference = tmp_path / "reference.png"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (240, 180))
    assert writer.isOpened()
    for index in range(8):
        writer.write(object_frame(index))
    writer.release()
    if automatic:
        from fishtrack import grounding, video
        from fishtrack.grounding import RegionProposal

        monkeypatch.setattr(
            grounding,
            "propose_regions",
            lambda frame, config: [
                RegionProposal((50, 60, 36, 40), 0.9, "fish"),
                RegionProposal((150, 100, 20, 20), 0.7, "fish"),
            ],
        )
        monkeypatch.setattr(
            video, "select_regions", lambda frame, initial_regions: initial_regions[:1]
        )
        options = ProcessingConfig(auto_roi=True, save_start_frame=reference)
    else:
        options = ProcessingConfig(rois=((50, 60, 36, 40),), save_start_frame=reference)
    assert process_video(source, output, DetectionConfig(), options, tmp_path / "debug") == 8
    assert cv2.imread(str(reference)).shape == (180, 240, 3)
    with (tmp_path / "debug" / "statistics.csv").open() as handle:
        records = list(csv.DictReader(handle))
    assert all(record["detections"] == "0" for record in records)
    assert all(record["followed_tracks"] == "1" for record in records[1:])
    assert all(record["suppression_reason"] == "unreliable_alignment" for record in records[1:])
    # Refuse to overwrite a user-owned reference image and do not publish the new video.
    original = reference.read_bytes()
    with pytest.raises(FileExistsError):
        process_video(source, tmp_path / "other.mp4", DetectionConfig(), options)
    assert reference.read_bytes() == original and not (tmp_path / "other.mp4").exists()


def test_new_cli_validation():
    """CLI rejects invalid selections and tracking thresholds with an argparse error."""
    for extra in (["--roi", "-1", "0", "20", "20"], ["--tracking-threshold", "nan"]):
        with pytest.raises(SystemExit) as error:
            main(["--input", "input.mp4", "--output", "output.mp4", *extra])
        assert error.value.code == 2


def test_selection_coordinates_and_drag_direction():
    """Selections use original pixels even when previewed smaller or dragged backwards."""
    from fishtrack.selection import selection_box

    assert selection_box((100, 80), (20, 30), 0.5, 640, 480) == (40, 60, 160, 100)
    assert selection_box((-10, -10), (400, 300), 0.5, 640, 480) == (0, 0, 640, 480)
    with pytest.raises(ValueError):
        selection_box((0, 0), (2, 2), 1, 640, 480)


def test_missing_tk_has_actionable_error(monkeypatch):
    """Headless installations can still use explicit rectangles without Tk."""
    import sys

    from fishtrack.selection import select_regions

    monkeypatch.setitem(sys.modules, "tkinter", None)
    with pytest.raises(OSError, match="use --roi"):
        select_regions(object_frame())


def test_cancelled_picker_does_not_publish_video(tmp_path, monkeypatch):
    """Cancelling a selection cleans up the temporary video and leaves the input alone."""
    from fishtrack import video

    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (240, 180))
    assert writer.isOpened()
    writer.write(object_frame())
    writer.release()
    original = source.read_bytes()

    def cancelled(frame):
        """Simulate the user closing the first-frame selection window."""
        raise ValueError("Fish selection cancelled")

    monkeypatch.setattr(video, "select_regions", cancelled)
    with pytest.raises(ValueError, match="cancelled"):
        process_video(source, output, DetectionConfig(), ProcessingConfig(select_roi=True))
    assert source.read_bytes() == original
    assert set(tmp_path.glob("*.mp4")) == {source}


def test_initial_size_drift_is_rejected():
    """A native tracker which collapses onto a small patch must not retain the ID."""
    from fishtrack.appearance import AppearanceFollower

    follower = AppearanceFollower(object_frame(), box_detection((50, 60, 36, 40)), 0.035)

    class CollapsedModel:
        """Return a spurious but internally accepted native tracking result."""

        def update(self, frame):
            """Simulate a tracker shrinking onto background texture."""
            return True, (55, 65, 12, 14)

    follower.tracker = CollapsedModel()
    gray = cv2.cvtColor(object_frame(), cv2.COLOR_BGR2GRAY)
    assert (
        follower.update(
            object_frame(), gray, gray, box_detection((50, 60, 36, 40)), (67.5, 79.5), 60
        )
        is None
    )
    assert follower.failure_reason == "size_drift"


def test_picker_uses_first_processed_frame(tmp_path, monkeypatch):
    """Interactive selections refer to the requested start frame, not source frame zero."""
    from fishtrack import video

    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (240, 180))
    assert writer.isOpened()
    for index in range(6):
        writer.write(object_frame(index))
    writer.release()
    seen = []

    def selected(frame):
        """Return a rectangle on the third decoded source frame."""
        seen.append(frame.copy())
        return ((56, 62, 36, 40),)

    monkeypatch.setattr(video, "select_regions", selected)
    options = ProcessingConfig(start=2 / 12, max_frames=2, select_roi=True)
    assert process_video(source, output, DetectionConfig(), options, tmp_path / "debug") == 2
    assert len(seen) == 1
    capture = cv2.VideoCapture(str(source))
    for _ in range(3):
        _, expected = capture.read()
    capture.release()
    assert np.array_equal(seen[0], expected)
    with (tmp_path / "debug" / "statistics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["confirmed_tracks"] == "1" and rows[0]["source_frame"] == "2"
    assert rows[1]["followed_tracks"] == "1"


def test_small_automatic_region_does_not_abort_high_resolution_video():
    """Candidates too small for the visual model remain detector-only tracks."""
    tracker = Tracker(min_hits=1)
    frame = np.zeros((120, 3840, 3), np.uint8)
    frame[50:70, 50:56] = np.random.default_rng(1).integers(0, 256, (20, 6, 3), dtype=np.uint8)
    tracks = tracker.update([box_detection((50, 50, 8, 20))], np.eye(2, 3), frame)
    assert len(tracks) == 1 and tracks[0].follower is None
