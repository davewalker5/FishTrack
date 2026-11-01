"""Optional, offline SAM 2.1 selected-object video segmentation experiment."""

from __future__ import annotations

import csv
import logging
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from fishtrack.control import ProcessingHooks
from fishtrack.detection import Image

if TYPE_CHECKING:
    from fishtrack.video import ProcessingConfig, VideoMetadata

LOGGER = logging.getLogger(__name__)
MAX_EXPERIMENT_FRAMES = 240
MODEL_NAMES = {"tiny": "t", "small": "s", "base-plus": "b+", "large": "l"}
CHECKPOINT_NAMES = {"tiny": "tiny", "small": "small", "base-plus": "base_plus", "large": "large"}
CHECKPOINT_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


@dataclass(frozen=True)
class ModelSpec:
    """Official SAM 2.1 configuration and checkpoint location."""

    config: str
    filename: str
    url: str


def model_spec(model: str) -> ModelSpec:
    """Return the matching official configuration/checkpoint for a supported size.

    :param model: tiny, small, base-plus or large.
    :return: Configuration path packaged with SAM 2 and an official download URL.
    :raises ValueError: If the size is unsupported.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown SAM 2 model: {model}")

    # Pair each supported model size with its matching upstream config and checkpoint.
    filename = f"sam2.1_hiera_{CHECKPOINT_NAMES[model]}.pt"
    return ModelSpec(
        f"configs/sam2.1/sam2.1_hiera_{MODEL_NAMES[model]}.yaml",
        filename,
        f"{CHECKPOINT_BASE_URL}/{filename}",
    )


def choose_device(torch: Any, requested: str) -> str:
    """Resolve an available inference device without silent device fallback.

    :param torch: Imported PyTorch module, injected to permit hardware-independent tests.
    :param requested: auto, cpu, mps or cuda.
    :return: An available concrete device name.
    :raises OSError: If an explicitly requested accelerator is unavailable.
    """
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }

    # Automatic selection prefers available accelerators; explicit choices must be available.
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available or not available[requested]:
        raise OSError(f"SAM 2 device {requested!r} is unavailable; try --sam2-device cpu")
    return requested


def load_predictor(options: ProcessingConfig) -> tuple[Any, Any, str]:
    """Lazily load optional dependencies and local weights, without network requests.

    :param options: SAM 2 model size, checkpoint and device configuration.
    :return: PyTorch module, initialized video predictor and selected device.
    :raises OSError: For missing dependencies, weights or model initialization errors.
    """

    # Check local weights before constructing the predictor or loading heavy dependencies.
    specification = model_spec(options.sam2_model)
    checkpoint = options.sam2_checkpoint or Path("models") / specification.filename
    if not checkpoint.is_file():
        raise OSError(
            f"Missing SAM 2 checkpoint: {checkpoint}. Run scripts/setup-sam2.sh "
            f"{options.sam2_model}, or supply --sam2-checkpoint"
        )
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise OSError("SAM 2 dependencies are missing. Run scripts/setup-sam2.sh") from error
    device = choose_device(torch, options.sam2_device)
    LOGGER.info("Loading SAM 2.1 %s on %s from %s", options.sam2_model, device, checkpoint)
    try:

        # CUDA-only hole filling is unnecessary for this portable CPU/MPS experiment.
        predictor = build_sam2_video_predictor(
            specification.config, str(checkpoint), device=device, apply_postprocessing=False
        )
    except (RuntimeError, ValueError, OSError) as error:
        raise OSError(
            f"Cannot load SAM 2: {error}. Check the checkpoint/model pairing "
            "and try --sam2-device cpu"
        ) from error
    return torch, predictor, device


@dataclass(frozen=True)
class MaskObservation:
    """Per-frame mask area, disposition and optional bounding box for a selected identity."""

    identifier: int
    pixels: int
    status: str
    box: tuple[int, int, int, int] | None


def annotate_masks(
    frame: Image, object_ids: list[int], masks: np.ndarray, options: ProcessingConfig
) -> tuple[Image, list[MaskObservation]]:
    """Render current model masks on an original frame copy, omitting empty predictions.

    :param frame: Original BGR source frame.
    :param object_ids: Stable selected-object IDs corresponding to mask order.
    :param masks: Boolean masks shaped objects x height x width.
    :param options: Outline, ID, box and pixel-area settings.
    :return: Annotated image and one diagnostic observation per selected identity.
    :raises ValueError: If model outputs do not match the source frame or object count.
    """
    if masks.shape != (len(object_ids), *frame.shape[:2]):
        raise ValueError("SAM 2 masks do not match the frame dimensions or selected IDs")

    # Keep source pixels untouched while adding annotations to a separate output image.
    result = frame.copy()
    observations = []
    for identifier, mask in zip(object_ids, masks, strict=True):
        binary = (mask != 0).astype(np.uint8) * 255
        pixels = cv2.countNonZero(binary)
        status = "visible"

        # Record absent or tiny masks without drawing a misleading visible track.
        if pixels == 0:
            status = "no_mask"
        elif pixels < options.sam2_min_area:
            status = "below_min_area"
        if status != "visible":
            observations.append(MaskObservation(int(identifier), pixels, status, None))
            continue

        # Use the current mask silhouette rather than transporting a previous contour.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box = cv2.boundingRect(cv2.findNonZero(binary))
        observations.append(MaskObservation(int(identifier), pixels, status, box))
        color = tuple(64 + (int(identifier) * factor) % 192 for factor in (47, 89, 137))
        if options.outline:
            cv2.drawContours(result, contours, -1, color, 2)
        x, y, width, height = box
        if options.bounding_boxes:
            cv2.rectangle(result, (x, y), (x + width, y + height), color, 1)
        if options.track_ids:
            cv2.putText(
                result,
                str(identifier),
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return result, observations


def process_sam2_frames(
    frames: Iterator[tuple[int, Image]],
    writer: cv2.VideoWriter,
    metadata: VideoMetadata,
    start_frame: int,
    options: ProcessingConfig,
    debug_dir: Path | None,
    hooks: ProcessingHooks | None = None,
) -> int:
    """Segment a bounded selected clip, exporting original frames with fresh model outlines.

    :param frames: Validated clip-relative indices and source BGR images.
    :param writer: Open MP4 writer managed by the common video-processing session.
    :param metadata: Source dimensions and frame rate.
    :param start_frame: Source start index, used for diagnostics.
    :param options: Selected regions, model options and annotation settings.
    :param debug_dir: Optional new diagnostic directory already created by the caller.
    :return: Number of frames written; incomplete propagation raises instead of publishing.
    :raises OSError: For model execution, staging or incomplete propagation failures.
    :raises ValueError: For invalid regions, empty clips or experiments above the frame bound.
    """
    from fishtrack.video import prepare_initial_frame


    # Use the same cancellation and progress callbacks for desktop and command-line runs.
    control = hooks or ProcessingHooks()
    control.report(0, 0, "Loading SAM 2")
    torch, predictor, device = load_predictor(options)
    control.check()
    started = time.perf_counter()

    # Ensure staged frames are removed even if inference fails or the user cancels.
    with ExitStack() as stack:
        temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="fishtrack-sam2-"))
        staging = Path(temporary)
        frame_count = 0
        regions = ()
        LOGGER.info(
            "Preparing selected frames for SAM 2 (maximum %d frames)", MAX_EXPERIMENT_FRAMES
        )
        for index, frame in frames:
            control.report(index, 0, "Preparing video frames")
            if index >= MAX_EXPERIMENT_FRAMES:
                raise ValueError(
                    f"SAM 2 experiment is limited to {MAX_EXPERIMENT_FRAMES} frames; "
                    "shorten --duration or --max-frames"
                )

            # Choose identities once on the first processed frame and validate their source bounds.
            if index == 0:
                regions = prepare_initial_frame(frame, options, debug_dir, control)
                for x, y, width, height in regions:
                    if x + width > metadata.width or y + height > metadata.height:
                        raise ValueError("Selected SAM 2 ROI is outside the source frame")

            # JPEGs follow the upstream loader contract; lossless PNGs preserve the export image.
            for extension, parameters in (("jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]), ("png", [])):
                if not cv2.imwrite(str(staging / f"{index:06d}.{extension}"), frame, parameters):
                    raise OSError("Cannot stage frames for SAM 2")
            frame_count += 1
        if not frame_count:
            raise ValueError("Requested clip contains no decodable frames")
        if not regions:
            raise ValueError("Select at least one fish for SAM 2")
        statistics = None
        if debug_dir is not None:
            handle = stack.enter_context((debug_dir / "sam2-statistics.csv").open("w", newline=""))
            statistics = csv.writer(handle)
            statistics.writerow(
                [
                    "source_frame",
                    "seconds",
                    "object_id",
                    "mask_pixels",
                    "status",
                    "x",
                    "y",
                    "width",
                    "height",
                ]
            )
        count = 0
        try:
            with torch.inference_mode():
                control.report(0, frame_count, "Loading video into SAM 2")

                # Keep video frames and tracking state in CPU memory to save accelerator memory.
                state = predictor.init_state(
                    str(staging),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=False,
                )

                # Assign stable IDs to accepted boxes and prompt SAM 2 on frame zero.
                for identifier, (x, y, width, height) in enumerate(regions, start=1):
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=identifier,
                        box=np.array([x, y, x + width, y + height], dtype=np.float32),
                    )
                for index, object_ids, logits in predictor.propagate_in_video(state):
                    control.check()

                    # Reject reordered frames or unexpected IDs before publishing.
                    if index != count or index >= frame_count:
                        raise OSError("SAM 2 returned unexpected frame ordering")
                    if len(object_ids) != len(regions) or set(object_ids) != set(
                        range(1, len(regions) + 1)
                    ):
                        raise OSError("SAM 2 returned unexpected object IDs")
                    original = cv2.imread(str(staging / f"{index:06d}.png"))
                    if original is None:
                        raise OSError("Cannot read staged original frame")

                    # Threshold logits and move masks to CPU for rendering and diagnostics.
                    masks = (logits > 0).detach().cpu().numpy()[:, 0]
                    rendered, observations = annotate_masks(
                        original, list(object_ids), masks, options
                    )
                    writer.write(rendered)
                    control.show(rendered)
                    control.report(index + 1, frame_count, "Tracking fish with SAM 2")
                    if statistics is not None:
                        for object_index, observation in enumerate(observations):
                            statistics.writerow(
                                [
                                    start_frame + index,
                                    (start_frame + index) / metadata.fps,
                                    observation.identifier,
                                    observation.pixels,
                                    observation.status,
                                    *(observation.box or ("", "", "", "")),
                                ]
                            )
                            path = debug_dir / f"{index:06d}-object-{observation.identifier}.png"
                            if not cv2.imwrite(
                                str(path), masks[object_index].astype(np.uint8) * 255
                            ):
                                raise OSError("Cannot write SAM 2 diagnostic mask")
                    count += 1
                    if count == 1 or count % 10 == 0:
                        LOGGER.info("SAM 2: processed %d/%d frames", count, frame_count)
        except RuntimeError as error:
            raise OSError(
                f"SAM 2 execution failed on {device}: {error}. "
                "For accelerator errors, retry with --sam2-device cpu"
            ) from error

        # Require a result for every staged frame before the caller can publish the video.
        if count != frame_count:
            raise OSError(
                f"SAM 2 produced {count} of {frame_count} frames; output was not published"
            )
        LOGGER.info(
            "SAM 2 completed %d frames in %.1f seconds", count, time.perf_counter() - started
        )
        return count
