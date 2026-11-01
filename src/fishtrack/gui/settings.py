"""Immutable shipped UI defaults and atomic preferences in Qt standard locations."""

import json
import logging
import math
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import QIODevice, QSaveFile, QStandardPaths

from fishtrack.detection import DetectionConfig
from fishtrack.video import ProcessingConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UiOptions:
    """Typed values shared by the settings widgets, persisted preferences and job builder."""

    backend: str
    selection: str
    review: bool
    start: float
    duration: float
    max_frames: int
    sam2_model: str
    sam2_device: str
    sam2_checkpoint: str
    sam2_min_area: int
    grounding_prompt: str
    grounding_threshold: float
    grounding_text_threshold: float
    grounding_device: str
    grounding_model: str
    max_auto_rois: int
    outline: bool
    track_ids: bool
    bounding_boxes: bool
    min_area: float
    max_area: float
    motion_threshold: float
    compensate: bool
    local_motion: bool
    min_residual_motion: float
    max_distance: float
    max_missing: int
    min_hits: int
    tracking_threshold: float
    output_beside_input: bool
    output_folder: str
    output_name: str
    diagnostics: bool


def default_options() -> UiOptions:
    """Load a fresh set of shipped values from src/config/defaults/ui.json."""
    document = json.loads(files("config").joinpath("defaults/ui.json").read_text())
    return UiOptions(**document["settings"])


def validated_options(values: dict) -> UiOptions:
    """Validate preference types and numeric ranges before creating typed options.

    :param values: Complete settings mapping, normally read from widgets or preferences.
    :return: Validated options.
    :raises ValueError: For malformed types, unsupported choices or invalid values.
    """
    defaults = asdict(default_options())
    if not isinstance(values, dict) or values.keys() != defaults.keys():
        raise ValueError("Settings must contain exactly the supported option names")
    for key, default in defaults.items():
        value = values[key]
        if isinstance(default, float):
            valid = type(value) in (float, int) and math.isfinite(value)
        else:
            valid = type(value) is type(default)
        if not valid:
            raise ValueError(f"Invalid setting: {key}")
    options = UiOptions(**values)
    if options.selection not in ("automatic", "manual", "motion"):
        raise ValueError("Choose automatic proposals, manual selection or motion acquisition")
    if options.start < 0 or options.duration < 0 or options.max_frames < 0:
        raise ValueError("Start time, duration and frame limit cannot be negative")
    if options.output_name and (
        Path(options.output_name).name != options.output_name
        or "\\" in options.output_name
        or Path(options.output_name).suffix.lower() != ".mp4"
    ):
        raise ValueError("Output name must be an MP4 filename without a folder")
    detection_config(options)
    processing_config(options, Path.cwd())
    return options


def processing_config(options: UiOptions, project_root: Path) -> ProcessingConfig:
    """Translate presentation settings into the existing engine's public configuration.

    :param options: Validated desktop settings.
    :param project_root: Base for bundled experiment model paths, independent of working directory.
    :return: Existing processing configuration consumed by both CLI and UI engines.
    """
    sam2 = options.backend == "sam2"
    automatic = options.selection == "automatic"
    manual = options.selection == "manual" or (sam2 and not automatic)
    return ProcessingConfig(
        start=options.start,
        duration=options.duration or None,
        max_frames=options.max_frames or None,
        backend=options.backend,
        select_roi=manual,
        auto_roi=automatic,
        accept_auto_roi=automatic and not options.review,
        sam2_model=options.sam2_model,
        sam2_device=options.sam2_device,
        sam2_checkpoint=Path(options.sam2_checkpoint)
        if options.sam2_checkpoint
        else project_root / "models" / f"sam2.1_hiera_{options.sam2_model.replace('-', '_')}.pt",
        sam2_min_area=options.sam2_min_area,
        grounding_prompt=options.grounding_prompt,
        grounding_threshold=options.grounding_threshold,
        grounding_text_threshold=options.grounding_text_threshold,
        grounding_device=options.grounding_device,
        grounding_model=Path(options.grounding_model)
        if options.grounding_model
        else project_root / "models/grounding-dino-tiny",
        max_auto_rois=options.max_auto_rois,
        outline=options.outline,
        track_ids=options.track_ids,
        bounding_boxes=options.bounding_boxes,
        max_distance=options.max_distance,
        max_missing=options.max_missing,
        min_hits=options.min_hits,
        tracking_threshold=options.tracking_threshold,
    )


def detection_config(options: UiOptions) -> DetectionConfig:
    """Build conventional detection options without importing any learned model."""
    return DetectionConfig(
        motion_threshold=options.motion_threshold,
        min_area=options.min_area,
        max_area=options.max_area,
        compensate=options.compensate,
        local_motion=options.local_motion,
        min_residual_motion=options.min_residual_motion,
    )


def standard_config_directory() -> Path:
    """Return the per-user application configuration location selected by Qt."""
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation))


def standard_log_directory() -> Path:
    """Return the per-user application-local data location for rotating logs."""
    return (
        Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        / "logs"
    )


@dataclass
class Preferences:
    """User choices separate from immutable packaged defaults."""

    folder: str
    geometry: str
    options: UiOptions


class SettingsStore:
    """Versioned JSON preferences with recovery from invalid or interrupted writes."""

    def __init__(self, directory: Path | None = None) -> None:
        """Select the platform-standard location or an explicit test directory."""
        self.directory = directory if directory is not None else standard_config_directory()
        self.path = self.directory / "preferences.json"

    def load(self) -> Preferences:
        """Restore validated user values, logging and falling back on malformed preferences."""
        fallback = Preferences(str(Path.home()), "", default_options())
        if not self.path.exists():
            return fallback
        try:
            document = json.loads(self.path.read_text())
            if not isinstance(document, dict) or (
                type(document.get("version")) is not int or document["version"] != 1
            ):
                raise ValueError("Unsupported preferences version")
            if not isinstance(document.get("folder"), str) or not isinstance(
                document.get("geometry"), str
            ):
                raise ValueError("Invalid folder or window geometry")
            return Preferences(
                document["folder"], document["geometry"], validated_options(document["options"])
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            LOGGER.warning("Could not restore preferences; using defaults: %s", error)
            return fallback

    def save(self, preferences: Preferences) -> None:
        """Atomically persist the user's folder, geometry and validated settings.

        :param preferences: Current application choices.
        :raises OSError: If Qt cannot commit the preference file.
        """
        options = validated_options(asdict(preferences.options))
        document = {
            "version": 1,
            "folder": preferences.folder,
            "geometry": preferences.geometry,
            "options": asdict(options),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        output = QSaveFile(str(self.path))
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(output.errorString())
        payload = (json.dumps(document, indent=2) + "\n").encode()
        if output.write(payload) != len(payload) or not output.commit():
            raise OSError(output.errorString())
