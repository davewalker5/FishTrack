"""Synthetic source-coordinate results and backend capture regression checks."""

import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fishtrack.analysis import (
    Observation,
    TrackingRun,
    capture_cv,
    capture_sam2,
    export_csv,
    load_run,
    save_run,
    source_fingerprint,
    track_events,
    track_statistics,
    validate_comparison,
    validate_run,
)


def example_run() -> TrackingRun:
    """Build a known 3-4-5 path with one missing source frame after a nonzero start."""
    run = TrackingRun(
        "absent.mp4",
        "fingerprint",
        640,
        480,
        10,
        100,
        20,
        "conventional",
        {},
        processed_frames=5,
        completion="complete",
    )
    run.observations = [
        Observation(20, 2, 1, "observed", 10, 10, (5, 5, 10, 10)),
        Observation(21, 2.1, 1, "observed", 13, 14, (8, 9, 10, 10)),
        Observation(22, 2.2, 1, "lost"),
        Observation(23, 2.3, 1, "observed", 100, 100, (95, 95, 10, 10)),
        Observation(24, 2.4, 1, "observed", 103, 104, (98, 99, 10, 10)),
    ]
    return run


def test_statistics_do_not_bridge_gaps() -> None:
    """Known distances exclude the unobserved interval; coverage uses inclusive source frames."""
    run = example_run()
    stats = track_statistics(run, 1)
    assert stats["path_length_pixels"] == 10
    assert stats["coverage"] == 0.8
    assert stats["interruptions"] == 1
    assert stats["span_seconds"] == pytest.approx(0.4)
    assert stats["mean_box_area"] == 100
    assert stats["displacement_pixels"] == pytest.approx((93**2 + 94**2) ** 0.5)
    kinds = [event.kind for event in track_events(run, 1)]
    assert kinds.count("reported_loss") == 1
    assert kinds.count("reacquired") == 1
    assert "possible_position_jump" not in kinds


def test_empty_singleton_and_irregular_tracks() -> None:
    """Empty metrics distinguish unavailable from zero; a singleton has full coverage."""
    run = example_run()
    run.observations = []
    assert track_statistics(run, 1)["coverage"] is None
    assert track_events(run, 1) == []
    run.observations = [Observation(20, 2, 1, "visible", 0, 0)]
    stats = track_statistics(run, 1)
    assert stats["coverage"] == 1
    assert stats["path_length_pixels"] == 0
    assert stats["span_seconds"] == 0
    run.observations.append(Observation(24, 2.4, 1, "visible", 100, 100))
    assert track_statistics(run, 1)["coverage"] == 0.4
    assert track_statistics(run, 1)["path_length_pixels"] == 0


def test_persistence_and_exports(tmp_path) -> None:
    """Reopen without the video; exports retain nulls, units, provenance and derived metrics."""
    run = example_run()
    path = tmp_path / "run.json"
    save_run(run, path)
    loaded = load_run(path)
    assert loaded == run
    assert track_statistics(loaded, 1) == track_statistics(run, 1)
    assert json.loads(path.read_text())["statistics"]["1"]["coverage"] == 0.8
    with pytest.raises(FileExistsError):
        save_run(run, path)
    export_csv(loaded, tmp_path / "run.csv")
    with (tmp_path / "run.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[2]["x"] == ""
    assert rows[0]["frame"] == "20"
    assert rows[0]["backend"] == "conventional"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 99},
        {"fps": 0},
        {"fps": float("nan")},
        {"width": -1},
        {"coordinates": "resized"},
        {"completion": "unknown"},
        {"frame_step": True},
    ],
)
def test_invalid_metadata(change) -> None:
    """Invalid timelines and unsupported schemas cannot enter analysis."""
    with pytest.raises(ValueError):
        validate_run(replace(example_run(), **change))


@pytest.mark.parametrize(
    "change",
    [
        {"frame": 19},
        {"timestamp": 0},
        {"x": float("nan")},
        {"y": None},
        {"box": (1, 1, 0, 3)},
        {"box": (639, 1, 4, 3)},
        {"mask_area": -1},
    ],
)
def test_invalid_observations(change) -> None:
    """Reject incorrect timeline, partial coordinates and invalid measurements."""
    run = example_run()
    run.observations[0] = replace(run.observations[0], **change)
    with pytest.raises(ValueError):
        validate_run(run)


def test_duplicate_and_malformed_data(tmp_path) -> None:
    """Reject duplicate reports, broken JSON and incorrectly typed JSON structures."""
    run = example_run()
    run.observations.append(run.observations[0])
    with pytest.raises(ValueError):
        validate_run(run)
    path = tmp_path / "invalid.json"
    for value in ("{", "null", "[]", '{"observations": null}'):
        path.write_text(value)
        with pytest.raises(ValueError, match="Invalid tracking results"):
            load_run(path)


@pytest.mark.parametrize(
    "change",
    [
        {"fingerprint": "other"},
        {"width": 800},
        {"fps": 20},
        {"start_frame": 19},
        {"processed_frames": 6},
        {"source_frames": 200},
    ],
)
def test_incompatible_runs(change) -> None:
    """Matching filenames and local IDs never establish timeline compatibility."""
    with pytest.raises(ValueError):
        validate_comparison(example_run(), replace(example_run(), **change))


def test_compatible_backends_and_explicit_pairing() -> None:
    """Source identity survives renaming and backends need not share configuration or IDs."""
    run = example_run()
    second = replace(run, source="renamed.mp4", backend="sam2", configuration={"model": "tiny"})
    validate_comparison(run, second)


def test_backend_adapters_preserve_missing_measurements() -> None:
    """CV stale positions are null, while SAM mask area and box centre retain their meaning."""
    run = example_run()
    run.observations = []
    track = SimpleNamespace(
        identifier=3, position=(10, 20), missing=0, detection=SimpleNamespace(box=(5, 10, 10, 20))
    )
    capture_cv(run, 20, [track])
    track.missing = 1
    capture_cv(run, 21, [track])
    assert run.observations[0].x == 10
    assert run.observations[1].x is None
    assert run.observations[1].box is None
    report = SimpleNamespace(identifier=4, pixels=40, status="visible", box=(10, 20, 8, 10))
    capture_sam2(run, 22, [report])
    assert run.observations[-1].x == 14
    assert run.observations[-1].y == 25
    assert run.observations[-1].mask_area == 40


def test_discontinuity_thresholds() -> None:
    """Large adjacent position/area changes are possible events, never accuracy judgements."""
    run = example_run()
    run.observations[1] = replace(run.observations[1], x=300, box=(0, 0, 100, 100))
    possible = [event for event in track_events(run, 1) if event.inferred]
    assert {event.kind for event in possible} >= {"possible_position_jump", "possible_area_jump"}
    with pytest.raises(ValueError):
        track_events(run, 1, area_ratio_threshold=1)


def test_fingerprint_uses_content_not_name(tmp_path) -> None:
    """Renamed source files match, changed content does not."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"source")
    second.write_bytes(b"source")
    assert source_fingerprint(first) == source_fingerprint(second)
    second.write_bytes(b"edited")
    assert source_fingerprint(first) != source_fingerprint(second)
