"""Validated MP4 input and atomic annotated-video export."""

import csv
import json
import logging
import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from fishtrack.analysis import TrackingRun, capture_cv, result_path, save_run, source_fingerprint
from fishtrack.control import ProcessingCancelled, ProcessingHooks
from fishtrack.detection import DetectionConfig, Image, MotionDetector
from fishtrack.rendering import annotate
from fishtrack.selection import select_regions
from fishtrack.tracking import Tracker, box_detection

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Source dimensions, frames per second and advertised frame count."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        """Return advertised duration in seconds."""
        return self.frame_count / self.fps


def read_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read metadata from an open capture, rejecting unusable values.

    :param capture: Open OpenCV video capture.
    :return: Valid source metadata.
    :raises ValueError: If dimensions or frame rate are unavailable.
    """
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not all(math.isfinite(value) and value > 0 for value in (width, height, fps)):
        raise ValueError("Video has invalid dimensions or frame rate")
    return VideoMetadata(
        int(width), int(height), fps, max(0, int(count)) if math.isfinite(count) else 0
    )


@dataclass(frozen=True)
class ProcessingConfig:
    """Clip, tracking and annotation options."""

    start: float = 0
    duration: float | None = None
    max_frames: int | None = None
    max_distance: float = 60
    max_missing: int = 5
    min_hits: int = 3
    outline: bool = True
    track_ids: bool = False
    bounding_boxes: bool = False
    rois: tuple[tuple[int, int, int, int], ...] = ()
    detect_additional: bool = False
    appearance: bool = True
    tracking_threshold: float = 0.035
    save_start_frame: Path | None = None
    select_roi: bool = False
    backend: str = "conventional"
    sam2_checkpoint: Path | None = None
    sam2_model: str = "tiny"
    sam2_device: str = "auto"
    sam2_min_area: int = 40
    auto_roi: bool = False
    accept_auto_roi: bool = False
    grounding_model: Path = Path("models/grounding-dino-tiny")
    grounding_prompt: str = "fish."
    grounding_threshold: float = 0.3
    grounding_text_threshold: float = 0.25
    grounding_device: str = "cpu"
    max_auto_rois: int = 10

    def __post_init__(self) -> None:
        """Validate clip limits and tracking parameters before opening files."""
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("start must be finite and nonnegative")
        if self.duration is not None and (not math.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("duration must be finite and positive")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.min_hits < 1:
            raise ValueError("min_hits must be positive")
        Tracker(
            self.max_distance,
            self.max_missing,
            self.min_hits,
            self.appearance,
            self.tracking_threshold,
        )
        for box in self.rois:
            box_detection(box)
        if self.backend not in {"conventional", "sam2"}:
            raise ValueError("backend must be conventional or sam2")
        if self.sam2_model not in {"tiny", "small", "base-plus", "large"}:
            raise ValueError("Unknown SAM 2 model size")
        if self.sam2_device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("sam2_device must be auto, cpu, mps or cuda")
        if self.sam2_min_area < 1:
            raise ValueError("sam2_min_area must be positive")
        if self.accept_auto_roi and not self.auto_roi:
            raise ValueError("--accept-auto-roi requires --auto-roi")
        if self.accept_auto_roi and self.select_roi:
            raise ValueError("Use either review with --select-roi or --accept-auto-roi")
        if self.grounding_device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unknown Grounding DINO device")
        if not self.grounding_prompt.strip().strip("."):
            raise ValueError("Grounding prompt must contain text")
        if self.max_auto_rois < 1:
            raise ValueError("max_auto_rois must be positive")
        for value in (self.grounding_threshold, self.grounding_text_threshold):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Grounding thresholds must be finite and in (0, 1]")
        if self.backend == "sam2":
            if not (self.rois or self.select_roi or self.auto_roi):
                raise ValueError("SAM 2 requires --roi, --select-roi or --auto-roi to select fish")
            if self.detect_additional:
                raise ValueError("SAM 2 does not support --detect-additional")
            if self.duration is None and self.max_frames is None:
                raise ValueError("SAM 2 experiments require --duration or --max-frames")
        if (self.rois or self.select_roi or self.auto_roi) and not self.appearance:
            raise ValueError("Manual ROIs require appearance tracking")
        if self.save_start_frame is not None and self.save_start_frame.suffix.lower() != ".png":
            raise ValueError("save_start_frame must use the .png extension")


def process_video(
    input_path: Path,
    output_path: Path,
    detection: DetectionConfig,
    options: ProcessingConfig,
    debug_dir: Path | None = None,
    hooks: ProcessingHooks | None = None,
) -> int:
    """Process a clip and atomically publish an MP4 after successful completion.

    :param input_path: Existing MP4, never modified.
    :param output_path: New MP4 path; existing files are refused.
    :param detection: Motion and contour thresholds.
    :param options: Clip and display configuration.
    :param debug_dir: Optional new directory for masks, aligned frames and CSV statistics.
    :param hooks: Optional cancellation, progress, preview and region-review callbacks.
    :return: Number of exported frames.
    :raises ValueError: For unsafe paths, unusable video or empty clip.
    :raises OSError: If video encoding or diagnostic output fails.
    """
    control = hooks or ProcessingHooks()
    control.report(0, 0, "Opening video")

    # Resolve aliases before checking that export cannot overwrite the source or an existing file.
    input_path, output_path = input_path.resolve(), output_path.resolve()
    if input_path.suffix.lower() != ".mp4" or not input_path.is_file():
        raise ValueError("Input must be an existing MP4 file")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    if input_path == output_path or output_path.exists():
        raise ValueError("Output must be a new file, distinct from the input")
    if not output_path.parent.is_dir():
        raise ValueError("Output parent directory must exist")

    if result_path(output_path).exists():
        raise ValueError("Tracking results already exist for this output")

    # Create a new diagnostics directory rather than mixing results with an earlier run.
    if debug_dir is not None:
        debug_dir = debug_dir.resolve()
        if debug_dir == output_path or debug_dir in output_path.parents:
            raise ValueError("Diagnostic directory must be separate from the output path")
        debug_dir.mkdir(parents=True, exist_ok=False)

    # Register cleanup as resources open so errors and cancellation release them reliably.
    with ExitStack() as stack:
        capture = cv2.VideoCapture(str(input_path))
        stack.callback(capture.release)
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        metadata = read_metadata(capture)
        LOGGER.info(
            "Source: %dx%d, %.3f fps, %d frames, %.2f seconds",
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.frame_count,
            metadata.duration,
        )
        start_frame = int(options.start * metadata.fps)
        if metadata.frame_count and start_frame >= metadata.frame_count:
            raise ValueError("Start time is beyond the video")

        # Decode sequentially for predictable clip boundaries across codec backends.
        for _ in range(start_frame):
            control.check()
            if not capture.read()[0]:
                raise ValueError("Cannot decode video up to the requested start time")

        # Encode beside the destination and publish only after processing and verification succeed.
        descriptor, temporary_name = tempfile.mkstemp(suffix=".mp4", dir=output_path.parent)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        stack.callback(temporary_path.unlink, missing_ok=True)
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        stack.callback(writer.release)
        if not writer.isOpened():
            raise OSError("Cannot initialise MP4 encoder (mp4v)")

        configuration = {"processing": asdict(options), "detection": asdict(detection)}
        # Paths are provenance strings rather than platform-specific JSON objects.
        configuration = json.loads(json.dumps(configuration, default=str))
        run = TrackingRun(
            str(input_path),
            source_fingerprint(input_path),
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.frame_count,
            start_frame,
            options.backend,
            configuration,
        )

        # Persist incomplete data even if validation/encoding/cancellation fails after tracking.
        stack.callback(_save_partial_run, run, result_path(output_path))

        # Share clip boundaries and cancellation handling between both tracking backends.
        frames = iter_clip(capture, metadata, start_frame, options, control)
        try:
            if options.backend == "sam2":
                from fishtrack.sam2_backend import process_sam2_frames

                count = process_sam2_frames(
                    frames, writer, metadata, start_frame, options, debug_dir, control, run
                )
            else:
                count = _process_conventional(
                    frames,
                    writer,
                    start_frame,
                    metadata,
                    detection,
                    options,
                    debug_dir,
                    stack,
                    control,
                    run,
                )
        except (Exception, KeyboardInterrupt) as error:
            run.completion = "cancelled" if isinstance(error, ProcessingCancelled) else "failed"
            raise

        # Finalize the MP4 container before attempting to decode the temporary export.
        writer.release()
        if count == 0:
            raise ValueError("Requested clip contains no decodable frames")

        # Check the encoded file is readable before exposing it as the completed result.
        verification = cv2.VideoCapture(str(temporary_path))
        try:
            if not verification.isOpened() or not verification.read()[0]:
                raise OSError("Encoded output could not be decoded")
        finally:
            verification.release()

        # Hard-link creation refuses concurrent replacement of an existing output.
        try:
            control.check()
        except ProcessingCancelled:
            run.completion = "cancelled"
            raise
        # Publish both outputs together; roll back our video if sidecar publication fails.
        os.link(temporary_path, output_path)
        try:
            run.completion = "complete"
            save_run(run, result_path(output_path))
        except (OSError, ValueError):
            run.completion = "failed"
            output_path.unlink()
            raise
        LOGGER.info("Exported %d frames to %s (video only; audio is omitted)", count, output_path)
        return count


def _save_partial_run(run: TrackingRun, path: Path) -> None:
    """Preserve available reports on unsuccessful exits without masking the original error."""
    if run.completion == "complete":
        return
    if run.completion == "incomplete":
        run.completion = "failed"
    try:
        save_run(run, path)
    except (OSError, ValueError):
        LOGGER.exception("Could not save partial tracking results to %s", path)


def _process_conventional(
    frames: Iterator[tuple[int, Image]],
    writer: cv2.VideoWriter,
    start_frame: int,
    metadata: VideoMetadata,
    detection: DetectionConfig,
    options: ProcessingConfig,
    debug_dir: Path | None,
    stack: ExitStack,
    control: ProcessingHooks,
    run: TrackingRun,
) -> int:
    """Run the conventional detector and visual tracker over a validated frame iterator."""
    statistics = None
    if debug_dir is not None:
        handle = stack.enter_context((debug_dir / "statistics.csv").open("w", newline=""))
        statistics = csv.writer(handle)
        statistics.writerow(
            [
                "source_frame",
                "seconds",
                "detections",
                "visible_tracks",
                "motion_pixels",
                "camera_fit",
                "suppression_reason",
                "confirmed_tracks",
                "followed_tracks",
                "unobserved_confirmed_tracks",
                "tracking_failures",
            ]
        )
    detector = MotionDetector(detection)
    tracker = Tracker(
        options.max_distance,
        options.max_missing,
        options.min_hits,
        options.appearance,
        options.tracking_threshold,
    )
    count = 0
    selected_rois = options.rois
    manual_selection = bool(selected_rois) or options.select_roi or options.auto_roi
    for count, frame in frames:
        if count == 0:
            selected_rois = prepare_initial_frame(frame, options, debug_dir, control)
            run.configuration["selected_rois"] = selected_rois
        detections, mask, aligned, matrix, reliable = detector.update(
            frame, detect=not manual_selection or options.detect_additional
        )
        if detector.suppression_reason == "scene_change":
            tracker.reset()
        if detector.suppression_reason is not None:
            LOGGER.debug(
                "Frame %d suppressed: %s", start_frame + count, detector.suppression_reason
            )
        if manual_selection and not options.detect_additional:
            detections = []
        tracks = tracker.update(detections, matrix, frame)
        if count == 0:
            if selected_rois:
                tracker.seed(frame, selected_rois)
                tracks = tracker.tracks
        capture_cv(run, start_frame + count, tracks)
        rendered = annotate(
            frame,
            tracks,
            options.outline,
            options.track_ids,
            options.bounding_boxes,
            options.min_hits,
        )
        writer.write(rendered)
        control.show(rendered)
        control.report(count + 1, clip_frame_limit(metadata, start_frame, options), "Tracking fish")
        if statistics is not None:
            statistics.writerow(
                [
                    start_frame + count,
                    (start_frame + count) / metadata.fps,
                    len(detections),
                    sum(t.missing == 0 for t in tracks),
                    cv2.countNonZero(mask),
                    reliable,
                    detector.suppression_reason or "",
                    sum(t.is_visible(options.min_hits) for t in tracks),
                    sum(t.propagated and t.missing == 0 for t in tracks),
                    sum(t.confirmed and t.missing > 0 for t in tracks),
                    ";".join(
                        f"{t.identifier}:{t.follower.failure_reason}"
                        for t in tracks
                        if t.missing
                        and t.follower is not None
                        and t.follower.failure_reason is not None
                    ),
                ]
            )
            for label, image in (("mask", mask), ("aligned", aligned)):
                path = debug_dir / f"{count:06d}-{label}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Cannot write diagnostic image: {path}")
        count += 1
        if count % 100 == 0:
            LOGGER.info("Processed %d frames", count)
    return count


def iter_clip(
    capture: cv2.VideoCapture,
    metadata: VideoMetadata,
    start_frame: int,
    options: ProcessingConfig,
    hooks: ProcessingHooks | None = None,
) -> Iterator[tuple[int, Image]]:
    """Yield clip-relative frame indices and original BGR images within requested limits.

    :param capture: Capture already positioned at the requested start frame.
    :param metadata: Advertised source metadata.
    :param start_frame: Zero-based source start index.
    :param options: Requested duration and frame limit.
    :return: Ordered clip-relative indices and frames.
    :raises OSError: For decoding failures substantially before the advertised end.
    """
    limits = [
        value
        for value in (
            options.max_frames,
            math.ceil(options.duration * metadata.fps) if options.duration else None,
        )
        if value is not None
    ]
    limit = min(limits) if limits else None
    count = 0
    while limit is None or count < limit:
        if hooks is not None:
            hooks.check()
        success, frame = capture.read()
        if not success:
            decoded_position = start_frame + count
            remaining = metadata.frame_count - decoded_position
            tail_tolerance = max(3, math.ceil(metadata.fps * 0.1))
            if remaining > tail_tolerance:
                raise OSError(f"Video decoding stopped early at frame {decoded_position}")
            if remaining > 0:
                LOGGER.warning(
                    "Video ended at frame %d; metadata advertises %d frames "
                    "(%d-frame tail discrepancy). Exporting decoded frames.",
                    decoded_position,
                    metadata.frame_count,
                    remaining,
                )
            break
        yield count, frame
        count += 1


def prepare_initial_frame(
    frame: Image,
    options: ProcessingConfig,
    debug_dir: Path | None = None,
    hooks: ProcessingHooks | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """Resolve selected regions and optionally save an unannotated reference PNG.

    :param frame: First processed source frame.
    :param options: Manual/interactive region and reference-image options.
    :param debug_dir: Optional directory for raw proposals and final accepted selections.
    :return: Selected full-resolution rectangles.
    :raises OSError: If reference output cannot be written without overwriting a file.
    """
    control = hooks or ProcessingHooks()
    control.check()
    regions = options.rois
    if options.auto_roi:
        control.report(0, 0, "Finding fish in the starting frame")
        from fishtrack.grounding import propose_regions, save_proposals

        proposals = propose_regions(frame, options)
        if debug_dir is not None:
            save_proposals(frame, proposals, debug_dir)
        regions += tuple(proposal.box for proposal in proposals)
    control.check()
    review = options.select_roi or (options.auto_roi and not options.accept_auto_roi)
    if review and control.select is not None:
        control.report(0, 0, "Review fish selections")
        regions = control.select(frame, regions)
    elif review:
        regions = (
            select_regions(frame, initial_regions=regions) if regions else select_regions(frame)
        )
    if options.auto_roi and not regions:
        raise ValueError("No fish regions were accepted; review proposals and add a fish manually")
    if options.auto_roi:
        LOGGER.info("Tracking %d accepted region(s)", len(regions))
    if debug_dir is not None and options.auto_roi:
        import json

        with (debug_dir / "selected-rois.json").open("x") as handle:
            json.dump(regions, handle, indent=2)
            handle.write("\n")
    if options.save_start_frame is not None:
        encoded, image_bytes = cv2.imencode(".png", frame)
        if not encoded:
            raise OSError("Cannot encode the reference frame")
        with options.save_start_frame.open("xb") as reference_file:
            reference_file.write(image_bytes.tobytes())
    return regions


def clip_frame_limit(metadata: VideoMetadata, start_frame: int, options: ProcessingConfig) -> int:
    """Estimate clip frames for progress reporting from advertised metadata and explicit limits."""
    limits = [
        value
        for value in (
            max(0, metadata.frame_count - start_frame) or None,
            options.max_frames,
            math.ceil(options.duration * metadata.fps) if options.duration else None,
        )
        if value is not None
    ]
    return min(limits) if limits else 0
