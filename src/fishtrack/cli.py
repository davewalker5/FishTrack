"""Command-line interface for the FishTrack prototype."""

import argparse
import logging
from pathlib import Path

import cv2

from fishtrack.detection import DetectionConfig
from fishtrack.video import ProcessingConfig, process_video


def build_parser() -> argparse.ArgumentParser:
    """Return the documented command-line parser."""
    parser = argparse.ArgumentParser(
        description="Detect and track moving candidates in MP4 footage"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--outline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track-ids", action="store_true")
    parser.add_argument("--bounding-boxes", action="store_true")
    parser.add_argument("--min-area", type=float, default=80)
    parser.add_argument("--max-area", type=float, default=50000)
    parser.add_argument("--max-aspect-ratio", type=float, default=8)
    parser.add_argument("--motion-threshold", type=float, default=25)
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument(
        "--no-local-motion",
        action="store_true",
        help="Disable local optical-flow filtering for comparisons",
    )
    parser.add_argument(
        "--min-residual-motion",
        type=float,
        default=2.0,
        help="Minimum local residual speed in source pixels per frame",
    )
    parser.add_argument(
        "--allow-unreliable-motion",
        action="store_true",
        help="Allow detections during poor alignment or scene changes",
    )
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, help="Clip duration in seconds")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-distance", type=float, default=60, help="Tracking gate in pixels")
    parser.add_argument("--max-missing", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        action="append",
        metavar=("X", "Y", "W", "H"),
        help="Select a fish in the first processed frame; repeat for multiple fish",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Open the first processed frame and drag boxes around fish",
    )
    parser.add_argument(
        "--detect-additional",
        action="store_true",
        help="Also acquire automatic tracks when manual ROIs are supplied",
    )
    parser.add_argument(
        "--no-appearance",
        action="store_true",
        help="Disable independent visual following for comparison",
    )
    parser.add_argument(
        "--tracking-threshold",
        type=float,
        default=0.035,
        help="Visual matching threshold in (0, 1]; higher values are stricter",
    )
    parser.add_argument(
        "--save-start-frame",
        type=Path,
        help="Save the first processed frame as a new PNG for ROI selection",
    )
    parser.add_argument(
        "--backend",
        choices=("conventional", "sam2"),
        default="conventional",
        help="Tracking method; SAM 2 is an optional selected-fish experiment",
    )
    parser.add_argument(
        "--sam2-model",
        choices=("tiny", "small", "base-plus", "large"),
        default="tiny",
        help="SAM 2.1 checkpoint size",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, help="Local SAM 2.1 checkpoint path")
    parser.add_argument(
        "--sam2-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="SAM 2 inference device",
    )
    parser.add_argument(
        "--sam2-min-area",
        type=int,
        default=40,
        help="Minimum visible SAM 2 mask area in source pixels",
    )
    parser.add_argument(
        "--auto-roi",
        action="store_true",
        help="Propose fish boxes on the starting frame, then review them",
    )
    parser.add_argument(
        "--accept-auto-roi",
        action="store_true",
        help="Accept proposals without opening the review window",
    )
    parser.add_argument(
        "--grounding-model",
        type=Path,
        default=Path("models/grounding-dino-tiny"),
        help="Local Grounding DINO model directory",
    )
    parser.add_argument("--grounding-prompt", default="fish.", help="Text describing ROI targets")
    parser.add_argument("--grounding-threshold", type=float, default=0.3)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-auto-rois", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, help="New directory for PNG diagnostics and CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Process CLI arguments and return a shell status; validation errors exit with status 2.

    :param argv: Optional explicit argument list, otherwise use process arguments.
    :return: Zero on success, one on processing failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        detection = DetectionConfig(
            args.motion_threshold,
            args.min_area,
            args.max_area,
            args.max_aspect_ratio,
            not args.no_compensation,
            local_motion=not args.no_local_motion,
            min_residual_motion=args.min_residual_motion,
            suppress_unreliable=not args.allow_unreliable_motion,
        )
        options = ProcessingConfig(
            args.start,
            args.duration,
            args.max_frames,
            args.max_distance,
            args.max_missing,
            args.min_hits,
            args.outline,
            args.track_ids,
            args.bounding_boxes,
            rois=tuple(tuple(box) for box in args.roi or []),
            detect_additional=args.detect_additional,
            appearance=not args.no_appearance,
            tracking_threshold=args.tracking_threshold,
            save_start_frame=args.save_start_frame,
            select_roi=args.select_roi,
            backend=args.backend,
            sam2_model=args.sam2_model,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_device=args.sam2_device,
            sam2_min_area=args.sam2_min_area,
            auto_roi=args.auto_roi,
            accept_auto_roi=args.accept_auto_roi,
            grounding_model=args.grounding_model,
            grounding_prompt=args.grounding_prompt,
            grounding_threshold=args.grounding_threshold,
            grounding_text_threshold=args.grounding_text_threshold,
            grounding_device=args.grounding_device,
            max_auto_rois=args.max_auto_rois,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        process_video(args.input, args.output, detection, options, args.debug_dir)
    except (ValueError, OSError, cv2.error) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1
    return 0
