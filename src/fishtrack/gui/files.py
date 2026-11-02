"""Immediate-folder input discovery and non-destructive output naming."""

from dataclasses import dataclass
from pathlib import Path

import cv2

from fishtrack.analysis import result_path
from fishtrack.detection import Image
from fishtrack.gui.settings import UiOptions
from fishtrack.video import VideoMetadata, read_metadata


@dataclass(frozen=True)
class InputFile:
    """Eligible MP4 file and its measured on-disk size."""

    path: Path
    size_bytes: int


def scan_folder(folder: Path) -> list[InputFile]:
    """Return immediate MP4 files, including uppercase extensions, sorted by filename.

    :param folder: Existing footage directory; subfolders are not traversed.
    :return: Eligible inputs and sizes.
    :raises OSError: For unavailable or unreadable directories/files.
    """
    return sorted(
        (
            InputFile(path, path.stat().st_size)
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda item: item.path.name.casefold(),
    )


def output_path(source: Path, options: UiOptions) -> Path:
    """Resolve an explicit new output or choose the first unused automatic filename.

    :param source: Selected input MP4.
    :param options: Output directory and optional filename preferences.
    :return: New path distinct from source, refusing explicit overwrites.
    :raises ValueError: If output naming or directory choices are invalid.
    """
    folder = source.parent if options.output_beside_input else Path(options.output_folder)
    if not options.output_beside_input and not options.output_folder:
        raise ValueError("Choose an output folder")
    if not folder.is_dir():
        raise ValueError("Output folder does not exist")
    if options.output_name:
        name = options.output_name
        if Path(name).name != name or "\\" in name or Path(name).suffix.lower() != ".mp4":
            raise ValueError("Output name must be an MP4 filename without a folder")
        candidate = folder / name
        if (
            candidate.resolve() == source.resolve()
            or candidate.exists()
            or result_path(candidate).exists()
        ):
            raise ValueError("Choose a new output filename; existing files are never overwritten")
        return candidate
    find_method = "Auto" if options.selection == "automatic" else "Manual"
    method = "CV" if options.backend == "conventional" else "SAM2"
    stem = f"{source.stem}-{find_method}-{method}-tracked"
    candidate = folder / f"{stem}.mp4"
    number = 1
    while (
        candidate.exists()
        or result_path(candidate).exists()
        or candidate.resolve() == source.resolve()
    ):
        candidate = folder / f"{stem}-{number:03d}.mp4"
        number += 1
    return candidate


def preview_frame(path: Path, start: float) -> tuple[Image, VideoMetadata]:
    """Read the requested starting frame and metadata, always releasing the capture.

    :param path: Selected input video.
    :param start: Preview time in seconds, rounded down to a frame.
    :return: BGR image and validated video metadata.
    :raises ValueError: If the video or requested starting frame cannot be decoded.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open the selected video")
        metadata = read_metadata(capture)
        frame_number = int(start * metadata.fps)
        if metadata.frame_count and frame_number >= metadata.frame_count:
            raise ValueError("Start time is beyond the video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            raise ValueError("Cannot decode the selected starting frame")
        return frame, metadata
    finally:
        capture.release()


def diagnostic_path(output: Path) -> Path:
    """Choose a new diagnostics folder without replacing results from an earlier failed job."""
    candidate = output.parent / f"{output.stem}-diagnostics"
    number = 1
    while candidate.exists():
        candidate = output.parent / f"{output.stem}-diagnostics-{number:03d}"
        number += 1
    return candidate
