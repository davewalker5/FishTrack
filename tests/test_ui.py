"""Headless Qt interaction, preference, output-safety and background-job regressions."""

import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from fishtrack.control import ProcessingCancelled, ProcessingHooks
from fishtrack.detection import DetectionConfig
from fishtrack.gui.canvas import FrameView, RegionCanvas, RegionDialog
from fishtrack.gui.files import output_path, scan_folder
from fishtrack.gui.main_window import MainWindow
from fishtrack.gui.options import OptionsPanel
from fishtrack.gui.settings import (
    Preferences,
    SettingsStore,
    default_options,
    processing_config,
    validated_options,
)
from fishtrack.gui.worker import ProcessingWorker
from fishtrack.video import ProcessingConfig, process_video


@pytest.fixture(scope="module")
def application():
    """Keep one QApplication alive for offscreen widget and queued-signal tests."""
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("FishTrack")
    app.setOrganizationName("FishTrack")
    app.setStyle("Fusion")
    yield app


def source_video(path: Path, frames: int = 8) -> None:
    """Encode a reproducible moving textured patch for real GUI processing jobs."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (240, 180))
    assert writer.isOpened()
    patch = np.random.default_rng(6).integers(0, 256, (35, 30, 3), dtype=np.uint8)
    for index in range(frames):
        frame = np.full((180, 240, 3), 60, np.uint8)
        frame[50:85, 40 + index : 70 + index] = patch
        writer.write(frame)
    writer.release()


def wait_until(app, condition, timeout=10):
    """Pump Qt events with a deadline while waiting for a background result."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        # Sleep releases the GIL so Python-heavy model startup is not starved by the test loop.
        time.sleep(0.002)
    assert condition(), "Qt operation did not finish within the test deadline"


def test_defaults_preferences_roundtrip_and_corruption(application, tmp_path, caplog):
    """Persist folder/settings atomically and recover from malformed preference data."""
    store = SettingsStore(tmp_path)
    defaults = default_options()
    options = replace(defaults, start=6.0, review=False)
    preferences = Preferences("/footage", "", options)
    store.save(preferences)
    assert store.load() == preferences
    assert default_options() == defaults
    store.path.write_text("{broken")
    assert store.load().options == defaults
    assert "using defaults" in caplog.text
    assert set(tmp_path.iterdir()) == {store.path}


@pytest.mark.parametrize(
    "change",
    [
        {"start": float("nan")},
        {"review": "false"},
        {"min_area": 0},
        {"output_name": "../source.mp4"},
    ],
)
def test_preferences_reject_invalid_types_and_paths(change):
    """Malformed settings cannot bypass core configuration and output validation."""
    values = asdict(default_options())
    values.update(change)
    with pytest.raises(ValueError):
        validated_options(values)


def test_file_filtering_and_safe_automatic_output(tmp_path):
    """Include uppercase MP4s and refuse explicit collisions or source aliases."""
    for name in ("reef.mp4", "UPPER.MP4", "notes.txt", "reef.mov"):
        (tmp_path / name).touch()
    (tmp_path / "nested.mp4").mkdir()
    assert [item.path.name for item in scan_folder(tmp_path)] == ["reef.mp4", "UPPER.MP4"]
    options = default_options()
    source = tmp_path / "reef.mp4"
    assert output_path(source, options).name == "reef-Auto-SAM2-tracked.mp4"
    (tmp_path / "reef-Auto-SAM2-tracked.mp4").touch()
    assert output_path(source, options).name == "reef-Auto-SAM2-tracked-001.mp4"
    with pytest.raises(ValueError):
        output_path(source, replace(options, output_name="reef.mp4"))


def test_option_groups_restore_unlimited_conventional_and_switch_to_sam2(application):
    """Changing methods must not turn an unlimited conventional limit into one frame."""
    panel = OptionsPanel(default_options())
    conventional = replace(
        default_options(), backend="conventional", selection="motion", duration=0.0, max_frames=0
    )
    panel.set_values(conventional)
    assert panel.values().max_frames == 0
    assert panel.isTabEnabled(2) and not panel.isTabEnabled(1)
    panel.controls["backend"].setCurrentIndex(panel.controls["backend"].findData("sam2"))
    assert panel.values().max_frames == 240
    assert panel.values().selection == "automatic"
    assert panel.isTabEnabled(1) and not panel.isTabEnabled(2)
    panel.deleteLater()


def test_model_paths_are_independent_of_current_working_directory(tmp_path):
    """GUI model defaults resolve against the project, while manual mode avoids DINO."""
    configuration = processing_config(default_options(), tmp_path)
    assert configuration.sam2_checkpoint == tmp_path / "models/sam2.1_hiera_tiny.pt"
    assert configuration.grounding_model == tmp_path / "models/grounding-dino-tiny"
    assert configuration.auto_roi and not configuration.accept_auto_roi
    manual = processing_config(replace(default_options(), selection="manual"), tmp_path)
    assert manual.select_roi and not manual.auto_roi


def test_preview_owns_memory_and_preserves_aspect(application):
    """Qt previews do not retain borrowed frame storage or stretch the image."""
    frame = np.full((100, 200, 3), (20, 40, 60), np.uint8)
    view = FrameView()
    view.resize(800, 600)
    view.set_frame(frame)
    frame[:] = 0
    assert view.image.pixelColor(0, 0).red() == 60
    assert view.image_rect().height() == 400 and view.image_rect().top() == 100
    assert view.source_point(QPointF(400, 300)) == (100.0, 50.0)
    assert view.source_point(QPointF(400, 20)) is None
    view.deleteLater()


def test_region_canvas_real_mouse_edits_and_scaling(application):
    """Actual Qt mouse events add/remove source-coordinate boxes despite letterboxing."""
    canvas = RegionCanvas(np.zeros((100, 200, 3), np.uint8), ((20, 20, 30, 30),))
    canvas.resize(800, 600)
    canvas.show()
    application.processEvents()
    QTest.mouseClick(canvas, Qt.MouseButton.RightButton, pos=QPoint(120, 220))
    assert not canvas.regions
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(160, 180))
    QTest.mouseMove(canvas, QPoint(400, 340))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(400, 340))
    assert canvas.regions == [(40, 20, 60, 40)]
    QTest.keyClick(canvas, Qt.Key.Key_Backspace)
    assert not canvas.regions
    canvas.close()
    canvas.deleteLater()


def test_region_dialog_cannot_accept_empty_selection(application):
    """Empty selections remain editable and cannot start a model on zero fish."""
    dialog = RegionDialog(np.zeros((100, 200, 3), np.uint8))
    assert not dialog.accept_button.isEnabled()
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.canvas.regions.append((20, 20, 30, 30))
    dialog.canvas.changed.emit()
    assert dialog.accept_button.isEnabled()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_engine_cancellation_removes_partial_video(tmp_path):
    """Cancellation before publishing leaves source bytes and existing files untouched."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source_video(source)
    original = source.read_bytes()
    cancelled = False

    def progress(completed, total, stage):
        """Request cancellation after a real annotated frame has been written."""
        nonlocal cancelled
        if completed >= 1:
            cancelled = True

    hooks = ProcessingHooks(cancelled=lambda: cancelled, progress=progress)
    with pytest.raises(ProcessingCancelled):
        process_video(source, output, DetectionConfig(), ProcessingConfig(), hooks=hooks)
    assert source.read_bytes() == original
    assert set(tmp_path.glob("*.mp4")) == {source}


def test_worker_review_request_and_cancellation_are_nonblocking(application, tmp_path):
    """A worker waiting for the main thread can be cancelled without an unsafe termination."""
    source = tmp_path / "source.mp4"
    source_video(source)
    worker = ProcessingWorker(
        source, tmp_path / "output.mp4", DetectionConfig(), ProcessingConfig(select_roi=True)
    )
    requests = []
    worker.review_requested.connect(requests.append)
    worker.start()
    wait_until(application, lambda: bool(requests))
    worker.cancel()
    wait_until(application, lambda: worker.isFinished())
    worker.wait()
    assert worker.result.status == "cancelled"
    assert not worker.output.exists()
    worker.deleteLater()


def test_main_window_processing_success_refresh_and_restore(application, tmp_path):
    """A real background job produces MP4 output and keeps controls/preferences usable."""
    source = tmp_path / "source.mp4"
    source_video(source)
    store = SettingsStore(tmp_path / "preferences")
    conventional = replace(
        default_options(), backend="conventional", selection="motion", duration=0.0, max_frames=4
    )
    store.save(Preferences(str(tmp_path), "", conventional))
    window = MainWindow(store)
    window.show()
    application.processEvents()
    assert window.table.rowCount() == 1 and window.selected_source() == source
    assert not window.windowIcon().isNull()
    window.start_processing()
    assert not window.folder_button.isEnabled() and window.cancel_button.isEnabled()
    wait_until(application, lambda: window.worker is None)
    assert window.last_output.is_file()
    assert window.progress.value() == 100
    assert window.folder_button.isEnabled() and window.table.rowCount() == 2
    assert window.selected_source() == source
    window.close()
    application.processEvents()
    restored = MainWindow(store)
    assert restored.folder == tmp_path and restored.options.values().max_frames == 4
    restored.close()
    application.processEvents()


def test_folder_refresh_handles_deletion_and_keeps_path_identity(application, tmp_path):
    """Removed inputs disappear and selection follows paths rather than stale row numbers."""
    for name in ("one.mp4", "two.MP4"):
        source_video(tmp_path / name, 1)
    window = MainWindow(SettingsStore(tmp_path / "preferences"), tmp_path)
    window.table.selectRow(1)
    selected = window.selected_source()
    (tmp_path / "aaa.txt").touch()
    window.refresh()
    assert window.selected_source() == selected
    selected.unlink()
    window.refresh()
    assert window.table.rowCount() == 1 and window.selected_source() != selected
    window.close()


def test_main_window_reports_background_failure(application, tmp_path):
    """A worker failure leaves the application alive and re-enables its inputs."""
    source = tmp_path / "source.mp4"
    source_video(source, 2)
    store = SettingsStore(tmp_path / "preferences")
    store.save(
        Preferences(
            str(tmp_path),
            "",
            replace(
                default_options(),
                backend="conventional",
                selection="motion",
                duration=0.0,
                max_frames=2,
            ),
        )
    )
    window = MainWindow(store)
    source.unlink()
    window.start_processing()
    wait_until(application, lambda: window.worker is None)
    assert "failed" in window.status.text().lower()
    assert window.folder_button.isEnabled()
    window.close()


def test_qt_review_is_used_during_real_processing(application, tmp_path):
    """Manual GUI processing opens Qt review on the main thread and resumes its waiting worker."""
    from PySide6.QtCore import QTimer

    source = tmp_path / "source.mp4"
    source_video(source, 4)
    store = SettingsStore(tmp_path / "preferences")
    store.save(
        Preferences(
            str(tmp_path),
            "",
            replace(
                default_options(),
                backend="conventional",
                selection="manual",
                duration=0.0,
                max_frames=4,
            ),
        )
    )
    window = MainWindow(store)
    window.show()
    reviews = []

    def accept_when_ready():
        """Interact with the actual nested Qt dialog event loop when it is shown."""
        if window.region_dialog is None:
            QTimer.singleShot(5, accept_when_ready)
            return
        dialog = window.region_dialog
        reviews.append(dialog)
        dialog.canvas.regions = [(40, 50, 30, 35)]
        dialog.canvas.changed.emit()
        dialog.accept()

    QTimer.singleShot(5, accept_when_ready)
    window.start_processing()
    wait_until(application, lambda: window.worker is None)
    assert len(reviews) == 1 and window.last_output.is_file()
    window.close()
    application.processEvents()


def test_closing_cancels_before_destroying_active_thread(application, tmp_path, monkeypatch):
    """Closing waits for cooperative cancellation before destroying a worker thread."""
    from fishtrack.gui import worker as worker_module

    source = tmp_path / "source.mp4"
    source_video(source, 1)
    store = SettingsStore(tmp_path / "preferences")
    store.save(
        Preferences(
            str(tmp_path),
            "",
            replace(default_options(), backend="conventional", selection="motion", max_frames=1),
        )
    )

    def waiting_job(source, output, detection, options, debug_dir, hooks):
        """Stay active until the GUI close request is observed by the engine."""
        while True:
            hooks.check()
            time.sleep(0.005)

    monkeypatch.setattr(worker_module, "process_video", waiting_job)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window = MainWindow(store)
    window.show()
    window.start_processing()
    window.close()
    wait_until(application, lambda: window.worker is None and not window.isVisible())
    assert window.close_requested and not (tmp_path / "source-Manual-CV-tracked.mp4").exists()


def test_diagnostics_choose_unused_folder(tmp_path):
    """A diagnostic directory left by cancellation never blocks a subsequent job."""
    from fishtrack.gui.files import diagnostic_path

    output = tmp_path / "reef-tracked.mp4"
    first = diagnostic_path(output)
    first.mkdir()
    second = diagnostic_path(output)
    assert second != first and not second.exists()


def test_rotating_log_setup_uses_selected_standard_directory(application, tmp_path, monkeypatch):
    """Application logs are bounded, persistent and separate from preferences/defaults."""
    from fishtrack.gui import app as app_module

    monkeypatch.setattr(app_module, "standard_log_directory", lambda: tmp_path / "logs")
    root = logging.getLogger()
    previous_level = root.level
    handler = app_module.configure_logging()
    try:
        logging.getLogger("fishtrack.ui-test").info("Persistent log verification")
        handler.flush()
        assert "Persistent log verification" in (tmp_path / "logs/fishtrack.log").read_text()
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.close()


def test_conventional_proposals_enable_grounding_controls(application, tmp_path):
    """Conventional tracking can review DINO proposals without SAM 2 weights."""
    options = replace(
        default_options(), backend="conventional", selection="automatic", max_frames=0
    )
    panel = OptionsPanel(options)
    assert panel.values().selection == "automatic"
    assert panel.isTabEnabled(1) and panel.isTabEnabled(2)
    assert not panel.sam_group.isEnabled()
    configuration = processing_config(panel.values(), tmp_path)
    assert configuration.auto_roi and not configuration.select_roi
    assert not configuration.accept_auto_roi
    assert configuration.max_frames is None
    with pytest.raises(ValueError, match="Grounding DINO"):
        MainWindow._check_dependencies(configuration)
    panel.deleteLater()


@pytest.mark.parametrize(
    ("backend", "selection", "expected"),
    [
        ("sam2", "automatic", "reef-Auto-SAM2-tracked.mp4"),
        ("sam2", "manual", "reef-Manual-SAM2-tracked.mp4"),
        ("conventional", "automatic", "reef-Auto-CV-tracked.mp4"),
        ("conventional", "manual", "reef-Manual-CV-tracked.mp4"),
        ("conventional", "motion", "reef-Manual-CV-tracked.mp4"),
    ],
)
def test_output_name_identifies_acquisition_and_tracker(tmp_path, backend, selection, expected):
    """Only DINO acquisition uses Auto, and explicit filenames remain user controlled."""
    source = tmp_path / "reef.mp4"
    source.touch()
    options = replace(default_options(), backend=backend, selection=selection)
    assert output_path(source, options).name == expected
    assert output_path(source, replace(options, output_name="custom.mp4")).name == "custom.mp4"
