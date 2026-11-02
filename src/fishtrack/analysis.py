"""Backend-independent image-space tracking results, statistics and persistence.

Coordinates are source pixels, origin top left; boxes are x, y, width, height.
Frames are zero-based source indices and timestamps are frame / source fps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from fishtrack.sam2_backend import MaskObservation
    from fishtrack.tracking import Track

SCHEMA_VERSION = 1
COORDINATES = "source_pixels_top_left"


@dataclass(frozen=True)
class Observation:
    """One backend report; missing positions are null, never predicted measurements."""

    frame: int
    timestamp: float
    track_id: int
    status: str
    x: float | None = None
    y: float | None = None
    box: tuple[float, float, float, float] | None = None
    mask_area: int | None = None


@dataclass(frozen=True)
class TrackEvent:
    """Reported state change or explicitly inferred possible discontinuity."""

    frame: int
    kind: str
    inferred: bool = False


@dataclass
class TrackingRun:
    """One processing execution and lightweight observations, without images or masks."""

    source: str
    fingerprint: str
    width: int
    height: int
    fps: float
    source_frames: int
    start_frame: int
    backend: str
    configuration: dict[str, object]
    run_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = SCHEMA_VERSION
    application_version: str = "1.1.0"
    coordinates: str = COORDINATES
    frame_step: int = 1
    processed_frames: int = 0
    completion: str = "incomplete"
    observations: list[Observation] = field(default_factory=list)

    @property
    def track_ids(self) -> list[int]:
        """Return sorted local identities, with no cross-run association implied."""
        return sorted({item.track_id for item in self.observations})

    def track(self, track_id: int) -> list[Observation]:
        """Return the ordered reports for a local identity."""
        return [item for item in self.observations if item.track_id == track_id]


def source_fingerprint(path: Path) -> str:
    """Hash file size and up to three 1 MiB samples, avoiding a full large-video read.

    :param path: Source file to identify independently of its name or location.
    :return: Versioned sampled SHA-256 fingerprint (not a full-file integrity hash).
    """

    # Sampling keeps source identification inexpensive even for multi-gigabyte videos.
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    block = 1024 * 1024
    with path.open("rb") as handle:
        # Deduplicate offsets so short files are not hashed repeatedly at the same position.
        for offset in sorted({0, max(0, size // 2 - block // 2), max(0, size - block)}):
            handle.seek(offset)
            digest.update(handle.read(block))

    return "sampled-sha256-v1:" + digest.hexdigest()


def capture_cv(run: TrackingRun, frame: int, tracks: list[Track]) -> None:
    """Adapt CV reports, excluding stale predictions from position measurements.

    :param run: Destination run.
    :param frame: Canonical source frame.
    :param tracks: Native CV identities; positions are their reported centroids.
    """

    # Retained lost identities may contain predictions, which are not observed positions.
    for track in tracks:
        visible = track.missing == 0
        run.observations.append(
            Observation(
                frame,
                frame / run.fps,
                track.identifier,
                "observed" if visible else "lost",
                float(track.position[0]) if visible else None,
                float(track.position[1]) if visible else None,
                tuple(float(value) for value in track.detection.box) if visible else None,
            )
        )

    # Count processed source frames even when this frame contains no object reports.
    run.processed_frames = frame - run.start_frame + 1


def capture_sam2(run: TrackingRun, frame: int, reports: list[MaskObservation]) -> None:
    """Adapt mask reports using the mask bounding-box centre, not a fabricated centroid."""

    # A bounding-box centre is available without retaining the full segmentation mask.
    for report in reports:
        box = report.box
        run.observations.append(
            Observation(
                frame,
                frame / run.fps,
                report.identifier,
                report.status,
                box[0] + box[2] / 2 if box else None,
                box[1] + box[3] / 2 if box else None,
                tuple(float(value) for value in box) if box else None,
                report.pixels,
            )
        )

    # Count processed source frames even when this frame contains no object reports.
    run.processed_frames = frame - run.start_frame + 1


def track_events(
    run: TrackingRun,
    track_id: int,
    position_threshold: float = 100,
    area_ratio_threshold: float = 2,
) -> list[TrackEvent]:
    """Report loss, gaps and possible jumps; thresholds are pixels and area ratios.

    :return: Events describing reports, never a claim of tracking error.
    :raises ValueError: For invalid discontinuity thresholds.
    """
    if (
        not math.isfinite(position_threshold)
        or position_threshold <= 0
        or not math.isfinite(area_ratio_threshold)
        or area_ratio_threshold <= 1
    ):
        raise ValueError("Discontinuity thresholds must be positive; area ratio must exceed one")

    events = []
    previous = None
    lost = False
    for item in run.track(track_id):
        # Collapse consecutive absent reports into one loss event until a position returns.
        if item.x is None:
            if not lost:
                events.append(TrackEvent(item.frame, "reported_loss"))
            lost = True
            continue
        if previous is None:
            events.append(TrackEvent(item.frame, "first_observation"))
        else:
            # Compare source indices so absent reports and omitted frames both reveal gaps.
            gap = item.frame - previous.frame > run.frame_step
            if gap:
                events.append(TrackEvent(previous.frame + run.frame_step, "missing_observations"))
            if lost or gap:
                events.append(TrackEvent(item.frame, "reacquired"))

            # Movement through an unobserved interval is unknown, not a measured jump.
            if not gap:
                if math.hypot(item.x - previous.x, item.y - previous.y) > position_threshold:
                    events.append(TrackEvent(item.frame, "possible_position_jump", True))
                if item.box and previous.box:
                    ratio = item.box[2] * item.box[3] / (previous.box[2] * previous.box[3])

                    # Treat proportional growth and shrinkage symmetrically.
                    if max(ratio, 1 / ratio) > area_ratio_threshold:
                        events.append(TrackEvent(item.frame, "possible_area_jump", True))
    
        previous, lost = item, False

    if previous is not None:
        events.append(TrackEvent(previous.frame, "last_observation"))

        # An early endpoint is inferred from missing reports, not proof of tracker failure.
        if previous.frame < run.start_frame + run.processed_frames - 1:
            events.append(TrackEvent(previous.frame + run.frame_step, "observations_ended", True))

    return events


def track_statistics(run: TrackingRun, track_id: int) -> dict[str, float | int | None]:
    """Calculate descriptive output metrics; path length never bridges missing frames.

    :return: Frames, seconds, coverage fraction, pixel distances and square-pixel areas.
    """

    # Missing positions must not inflate coverage or contribute stale coordinates to distances.
    items = [item for item in run.track(track_id) if item.x is not None]
    if not items:
        return {
            "observed_frames": 0,
            "first_frame": None,
            "last_frame": None,
            "first_timestamp": None,
            "last_timestamp": None,
            "span_seconds": 0,
            "coverage": None,
            "interruptions": 0,
            "mean_box_area": None,
            "displacement_pixels": None,
            "path_length_pixels": 0,
        }

    # Coverage includes both endpoint frames; elapsed time remains their timestamp difference.
    first, last = items[0], items[-1]
    pairs = list(zip(items, items[1:]))
    areas = [item.box[2] * item.box[3] for item in items if item.box]
    return {
        "observed_frames": len(items),
        "first_frame": first.frame,
        "last_frame": last.frame,
        "first_timestamp": first.timestamp,
        "last_timestamp": last.timestamp,
        "span_seconds": last.timestamp - first.timestamp,
        "coverage": len(items) / ((last.frame - first.frame) // run.frame_step + 1),
        "interruptions": sum(b.frame - a.frame > run.frame_step for a, b in pairs),
        "mean_box_area": sum(areas) / len(areas) if areas else None,
        "displacement_pixels": math.hypot(last.x - first.x, last.y - first.y),
        "path_length_pixels": sum(
            math.hypot(b.x - a.x, b.y - a.y)
            for a, b in pairs
            if b.frame - a.frame == run.frame_step
        ),
    }


def validate_run(run: TrackingRun) -> None:
    """Reject malformed metadata, out-of-range observations and ambiguous timelines.

    :raises ValueError: If the schema or source-coordinate observations are invalid.
    """
    if type(run.schema_version) is not int or (
        run.schema_version != SCHEMA_VERSION or run.coordinates != COORDINATES
    ):
        raise ValueError("Unsupported tracking schema or coordinate system")

    integers = (
        run.width,
        run.height,
        run.source_frames,
        run.start_frame,
        run.processed_frames,
        run.frame_step,
    )

    # Explicit type checks reject booleans, which Python otherwise accepts as integers.
    if any(type(value) is not int for value in integers):
        raise ValueError("Frame dimensions and timeline values must be integers")

    if (
        min(run.width, run.height, run.frame_step) <= 0
        or min(run.start_frame, run.source_frames, run.processed_frames) < 0
        or not math.isfinite(run.fps)
        or run.fps <= 0
    ):
        raise ValueError("Invalid source dimensions or timeline")

    if (
        run.backend not in {"conventional", "sam2"}
        or run.completion not in {"complete", "incomplete", "cancelled", "failed"}
        or not isinstance(run.configuration, dict)
        or not all(
            isinstance(value, str) and value
            for value in (run.source, run.fingerprint, run.run_id, run.application_version)
        )
    ):
        raise ValueError("Invalid run metadata")

    # Track ordering independently: reports from different identities may be interleaved.
    previous = {}
    for item in run.observations:
        if (
            type(item.frame) is not int
            or type(item.track_id) is not int
            or item.track_id < 1
            or not run.start_frame <= item.frame < run.start_frame + run.processed_frames
            or (item.frame - run.start_frame) % run.frame_step
            or item.frame <= previous.get(item.track_id, -1)
            or not isinstance(item.status, str)
            or not item.status
            or not math.isfinite(item.timestamp)
            or not math.isclose(item.timestamp, item.frame / run.fps, abs_tol=1e-8)
        ):
            raise ValueError("Invalid, duplicate or unordered observation timeline")

        previous[item.track_id] = item.frame

        # A position is either fully available or fully absent; partial points are ambiguous.
        if (item.x is None) != (item.y is None):
            raise ValueError("Observation must contain both coordinates or neither")

        if item.x is not None and not (
            math.isfinite(item.x)
            and math.isfinite(item.y)
            and 0 <= item.x <= run.width
            and 0 <= item.y <= run.height
        ):
            raise ValueError("Observation position is outside the source frame")

        if item.box is not None:
            if len(item.box) != 4 or not all(math.isfinite(value) for value in item.box):
                raise ValueError("Invalid bounding box")

            x, y, width, height = item.box
            if min(x, y) < 0 or min(width, height) <= 0:
                raise ValueError("Invalid bounding-box dimensions")

            if x + width > run.width or y + height > run.height:
                raise ValueError("Bounding box is outside the source frame")
    
        if item.mask_area is not None and (
            type(item.mask_area) is not int or not 0 <= item.mask_area <= run.width * run.height
        ):
            raise ValueError("Invalid mask area")


def validate_comparison(first: TrackingRun, second: TrackingRun) -> None:
    """Require matching sampled source identity and exact source-coordinate timelines."""
    validate_run(first)
    validate_run(second)

    # Backend settings may differ, but source identity, geometry and timeline must agree.
    for name in (
        "fingerprint",
        "width",
        "height",
        "source_frames",
        "start_frame",
        "processed_frames",
        "frame_step",
        "coordinates",
    ):
        if getattr(first, name) != getattr(second, name):
            raise ValueError(f"Cannot compare runs: {name.replace('_', ' ')} differs")

    if not math.isclose(first.fps, second.fps, rel_tol=1e-9):
        raise ValueError("Cannot compare runs: source frame rate differs")


def save_run(run: TrackingRun, path: Path) -> None:
    """Atomically publish new JSON, including reproducible derived metrics; refuse overwrite."""
    validate_run(run)

    # Keep raw reports alongside derived metrics so later sessions can reproduce the analysis.
    payload = asdict(run)
    payload["statistics"] = {str(key): track_statistics(run, key) for key in run.track_ids}
    payload["events"] = {
        str(key): [asdict(event) for event in track_events(run, key)] for key in run.track_ids
    }

    # Stage beside the destination so publication stays on the same filesystem.
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")

        # Exclusive hard-link creation publishes complete JSON without replacing an existing run.
        os.link(temporary, path)

    finally:
        temporary.unlink(missing_ok=True)


def load_run(path: Path) -> TrackingRun:
    """Load and validate JSON without requiring access to the original video.

    :raises ValueError: For malformed or unsupported saved tracking data.
    :raises OSError: If the file cannot be read.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "source",
            "fingerprint",
            "width",
            "height",
            "fps",
            "source_frames",
            "start_frame",
            "backend",
            "configuration",
            "run_id",
            "schema_version",
            "application_version",
            "coordinates",
            "frame_step",
            "processed_frames",
            "completion",
            "observations",
        }
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("Missing required tracking metadata")

        # Derived values are recomputed from validated observations rather than trusted on load.
        data.pop("statistics", None)
        data.pop("events", None)
        reports = data.pop("observations")
        observations = []
        for item in reports:
            if item.get("box") is not None:
                # Restore the internal immutable box representation from a JSON array.
                item["box"] = tuple(item["box"])

            observations.append(Observation(**item))

        run = TrackingRun(**data, observations=observations)
        validate_run(run)
        return run

    # Present malformed structures through one public error type while preserving I/O errors.
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError(f"Invalid tracking results: {error}") from error


def export_csv(run: TrackingRun, path: Path) -> None:
    """Export source frames, seconds, pixels and optional square-pixel mask areas.

    Null measurements become empty fields. Existing files are never overwritten.
    """
    validate_run(run)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "backend",
                "frame",
                "timestamp",
                "track_id",
                "x",
                "y",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "status",
                "mask_area",
            ]
        )

        # CSV writes None as empty fields, preserving the distinction between missing and zero.
        for item in run.observations:
            writer.writerow(
                [
                    run.run_id,
                    run.backend,
                    item.frame,
                    item.timestamp,
                    item.track_id,
                    item.x,
                    item.y,
                    *(item.box or (None,) * 4),
                    item.status,
                    item.mask_area,
                ]
            )


def result_path(video: Path) -> Path:
    """Return the JSON sidecar path following the existing output filename stem."""
    return video.with_name(video.stem + "-tracking.json")
