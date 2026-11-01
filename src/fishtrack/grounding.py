"""Optional, local text-prompted starting-frame ROI proposals using Grounding DINO."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from fishtrack.detection import Image
from fishtrack.sam2_backend import choose_device

if TYPE_CHECKING:
    from fishtrack.video import ProcessingConfig

LOGGER = logging.getLogger(__name__)
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


@dataclass(frozen=True)
class RegionProposal:
    """A proposed source-pixel rectangle with detector score and text label."""

    box: tuple[int, int, int, int]
    score: float
    label: str


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    """Return rectangular overlap divided by union for positive-width/height boxes."""
    x, y, width, height = first
    other_x, other_y, other_width, other_height = second

    # Measure shared area without counting separated rectangles as overlapping.
    intersection = max(0, min(x + width, other_x + other_width) - max(x, other_x)) * max(
        0, min(y + height, other_y + other_height) - max(y, other_y)
    )
    return intersection / (width * height + other_width * other_height - intersection)


def filter_proposals(
    boxes: np.ndarray, scores: np.ndarray, labels: list[str], width: int, height: int, maximum: int
) -> list[RegionProposal]:
    """Clamp XYXY predictions to the image and suppress duplicate rectangles.

    :param boxes: Source-pixel detector boxes shaped N x 4.
    :param scores: Detector scores shaped N.
    :param labels: Text labels corresponding to predictions.
    :param width: Source width in pixels.
    :param height: Source height in pixels.
    :param maximum: Maximum proposals retained after score-ordered IoU suppression.
    :return: Valid regions sorted by score, with IoU above 0.5 suppressed.
    :raises ValueError: For inconsistent model outputs or invalid image/limit parameters.
    """
    if scores.ndim != 1 or boxes.shape != (len(scores), 4) or len(labels) != len(scores):
        raise ValueError("Grounding DINO outputs have inconsistent box/score/label counts")
    if width < 1 or height < 1 or maximum < 1:
        raise ValueError("Image dimensions and proposal limit must be positive")

    # Convert finite detector coordinates into usable, clipped source-pixel rectangles.
    candidates = []
    for coordinates, score, label in zip(boxes, scores, labels, strict=True):
        if not np.isfinite(coordinates).all() or not np.isfinite(score):
            continue
        left = max(0, min(width, math.floor(coordinates[0])))
        top = max(0, min(height, math.floor(coordinates[1])))
        right = max(0, min(width, math.ceil(coordinates[2])))
        bottom = max(0, min(height, math.ceil(coordinates[3])))
        if right - left < 8 or bottom - top < 8:
            continue
        candidates.append(
            RegionProposal((left, top, right - left, bottom - top), float(score), label)
        )

    # Keep the strongest box first when overlapping proposals compete.
    candidates.sort(key=lambda proposal: (-proposal.score, proposal.box))
    accepted: list[RegionProposal] = []
    for candidate in candidates:
        if all(
            intersection_over_union(candidate.box, previous.box) <= 0.5 for previous in accepted
        ):
            accepted.append(candidate)
            if len(accepted) == maximum:
                break
    return accepted


def propose_regions(frame: Image, options: ProcessingConfig) -> list[RegionProposal]:
    """Generate local fish-related proposals on one starting frame, without network access.

    :param frame: First processed BGR source frame.
    :param options: Local model directory, text prompt, thresholds and device.
    :return: Scored, clamped and deduplicated candidate regions; not confirmed fish.
    :raises OSError: If optional dependencies, local weights or inference are unavailable.
    """
    if not options.grounding_model.is_dir():
        raise OSError("Grounding DINO weights are missing. Run scripts/setup-grounding.sh")
    try:
        import torch
        from PIL import Image as PillowImage
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as error:
        raise OSError(
            "Grounding DINO dependencies are missing. Run scripts/setup-grounding.sh"
        ) from error

    # Normalize the text prompt to the detector's period-terminated label format.
    device = choose_device(torch, options.grounding_device)
    prompt = options.grounding_prompt.strip().lower().rstrip(".") + "."
    started = time.perf_counter()
    LOGGER.info("Proposing starting-frame regions for %r on %s", prompt, device)
    try:

        # Use only local model assets and disallow downloaded executable model code.
        processor = AutoProcessor.from_pretrained(
            str(options.grounding_model), local_files_only=True, trust_remote_code=False
        )
        model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                str(options.grounding_model),
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )

        # Convert OpenCV's BGR pixels to the RGB format expected by the processor.
        image = PillowImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        # Inference needs no gradients; disabling them avoids unnecessary memory use.
        with torch.inference_mode():
            outputs = model(**inputs)

        # Map predictions back to source dimensions and apply the configured score gates.
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=options.grounding_threshold,
            text_threshold=options.grounding_text_threshold,
            target_sizes=[(frame.shape[0], frame.shape[1])],
        )[0]
        proposals = filter_proposals(
            result["boxes"].detach().cpu().numpy(),
            result["scores"].detach().cpu().numpy(),
            list(result["text_labels"]),
            frame.shape[1],
            frame.shape[0],
            options.max_auto_rois,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise OSError(
            f"Grounding DINO failed: {error}. Check local setup and try --grounding-device cpu"
        ) from error
    LOGGER.info(
        "Proposed %d region(s) in %.1f seconds", len(proposals), time.perf_counter() - started
    )
    return proposals


def save_proposals(frame: Image, proposals: list[RegionProposal], directory: Path) -> None:
    """Write scored proposal JSON and a labelled preview into a new diagnostics directory.

    :param frame: Original starting frame.
    :param proposals: Unreviewed model proposals in source coordinates.
    :param directory: Existing diagnostic directory managed by video processing.
    :raises OSError: If diagnostic output cannot be written.
    """
    with (directory / "roi-proposals.json").open("x") as handle:
        json.dump([asdict(proposal) for proposal in proposals], handle, indent=2)
        handle.write("\n")

    # Draw on a copy so diagnostics cannot alter the frame used for tracking.
    preview = frame.copy()
    for index, proposal in enumerate(proposals, start=1):
        x, y, width, height = proposal.box
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.putText(
            preview,
            f"{index}: {proposal.label} {proposal.score:.2f}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    if not cv2.imwrite(str(directory / "roi-proposals.jpg"), preview):
        raise OSError("Cannot write ROI proposal preview")
