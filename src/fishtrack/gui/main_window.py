"""Wireframe-based folder browser, video preview, grouped settings and job controller."""

import importlib.util
import logging
from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fishtrack.gui.canvas import FrameView, RegionDialog
from fishtrack.gui.files import diagnostic_path, output_path, preview_frame, scan_folder
from fishtrack.gui.options import OptionsPanel
from fishtrack.gui.settings import (
    Preferences,
    SettingsStore,
    default_options,
    detection_config,
    processing_config,
    standard_log_directory,
)
from fishtrack.gui.worker import ProcessingWorker, RegionRequest
from fishtrack.video import ProcessingConfig

LOGGER = logging.getLogger(__name__)


class LogBridge(QObject):
    """Queue log messages from any processing thread to the main window."""

    line = Signal(str)


class WindowLogHandler(logging.Handler):
    """Deliver formatted records to a bounded Qt log view through signals."""

    def __init__(self, bridge: LogBridge) -> None:
        """Hold a GUI-owned signal bridge rather than directly accessing its widgets."""
        super().__init__(logging.INFO)
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Send one formatted log line to the receiving GUI thread."""
        self.bridge.line.emit(self.format(record))


class MainWindow(QMainWindow):
    """Present one selected-video processing job with responsive, persistent controls."""

    def __init__(
        self,
        store: SettingsStore | None = None,
        folder: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Restore user choices and construct the two-column reference layout."""
        super().__init__()
        self.store = store or SettingsStore()
        self.preferences = self.store.load()
        self.folder = folder if folder is not None else Path(self.preferences.folder).expanduser()
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.worker: ProcessingWorker | None = None
        self.region_dialog: RegionDialog | None = None
        self.close_requested = False
        self.last_output: Path | None = None
        self.preview_path: Path | None = None
        self.preview_start: float | None = None
        self.setWindowTitle("FishTrack • Underwater fish tracking")
        self.resize(1240, 860)
        self.setMinimumSize(980, 680)
        self.icon = QIcon(str(files("fishtrack").joinpath("resources/fishtrack.svg")))
        self.setWindowIcon(self.icon)
        self._build_window()
        self.bridge = LogBridge(self)
        self.bridge.line.connect(self.log.appendPlainText)
        self.log_handler = WindowLogHandler(self.bridge)
        logging.getLogger().addHandler(self.log_handler)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._load_preview)
        self.options.changed.connect(self._settings_changed)
        if self.preferences.geometry:
            try:
                self.restoreGeometry(
                    QByteArray.fromBase64(self.preferences.geometry.encode("ascii"))
                )
            except (ValueError, TypeError) as error:
                LOGGER.warning("Could not restore window geometry: %s", error)
        self.refresh()

    def _build_window(self) -> None:
        """Arrange preview/logs on the left and folder/files/settings/actions on the right."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        left, right = QWidget(), QWidget()
        left_layout, right_layout = QVBoxLayout(left), QVBoxLayout(right)
        left_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setContentsMargins(8, 0, 0, 0)
        header = QHBoxLayout()
        artwork = QLabel()
        artwork.setPixmap(self.icon.pixmap(64, 64))
        artwork.setFixedSize(64, 64)
        artwork.setAccessibleName("FishTrack fish icon")
        header.addWidget(artwork)
        heading = QVBoxLayout()
        title = QLabel("FishTrack")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #25c6bb")
        heading.addWidget(title)
        subtitle = QLabel("Detect, outline and track fish in underwater video")
        subtitle.setWordWrap(True)
        heading.addWidget(subtitle)
        header.addLayout(heading, 1)
        analysis_button = QPushButton("Analysis")
        analysis_button.clicked.connect(self.open_analysis)
        header.addWidget(analysis_button)
        left_layout.addLayout(header)
        self.folder_label = QLabel()
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet(
            "background-color:#123f49; color:#e5faf7; "
            "border:1px solid #277079; border-radius:6px; padding:8px 12px;"
        )
        left_layout.addWidget(self.folder_label)
        self.preview = FrameView()
        left_layout.addWidget(self.preview, 1)
        self.metadata_label = QLabel("Select an input video")
        self.metadata_label.setWordWrap(True)
        left_layout.addWidget(self.metadata_label)
        self.status = QLabel("Ready")
        self.status.setStyleSheet(
            "background-color:#183348; color:#e9f4ff; "
            "border:1px solid #42637e; border-radius:6px; padding:8px 12px;"
        )
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(QLabel("Processing log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setMinimumHeight(110)
        self.log.setMaximumHeight(210)
        left_layout.addWidget(self.log)

        folder_actions = QHBoxLayout()
        self.folder_button = QPushButton("Choose folder…")
        self.folder_button.clicked.connect(self.choose_folder)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        folder_actions.addWidget(self.folder_button, 1)
        folder_actions.addWidget(self.refresh_button)
        right_layout.addLayout(folder_actions)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Filename", "Size (MiB)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget::item:selected {background-color:#126d78;color:white;}"
        )
        self.table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(self._open_source)
        self.table.setMinimumHeight(155)
        self.table.setMaximumHeight(245)
        right_layout.addWidget(self.table)
        self.file_count = QLabel("0 MP4 files")
        right_layout.addWidget(self.file_count)
        self.options = OptionsPanel(self.preferences.options)
        right_layout.addWidget(self.options, 1)
        self.output_label = QLabel("Output: select an input video")
        self.output_label.setWordWrap(True)
        self.output_label.setTextFormat(Qt.TextFormat.PlainText)
        self.output_label.setMinimumHeight(64)
        self.output_label.setStyleSheet(
            "background-color:#183348; color:#e9f4ff; "
            "border:1px solid #42637e; border-radius:6px; "
            "padding:10px 12px; font-weight:600;"
        )
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self.output_label)
        actions = QHBoxLayout()
        self.process_button = QPushButton("Process video")
        self.process_button.setStyleSheet(
            "background-color:#126d78; color:white; "
            "padding:10px; border-radius:5px; font-weight:600;"
        )
        self.process_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        actions.addWidget(self.process_button, 1)
        actions.addWidget(self.cancel_button)
        right_layout.addLayout(actions)
        utilities = QHBoxLayout()
        self.output_button = QPushButton("Open result")
        self.output_button.setEnabled(False)
        self.output_button.clicked.connect(self._open_result)
        self.defaults_button = QPushButton("Restore defaults")
        self.defaults_button.clicked.connect(self._restore_defaults)
        log_button = QPushButton("Logs…")
        log_button.clicked.connect(self._open_logs)
        for button in (self.output_button, self.defaults_button, log_button):
            utilities.addWidget(button)
        right_layout.addLayout(utilities)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([730, 450])

    def open_analysis(self) -> None:
        """Open saved-run analysis, initially showing the most recent processing result."""
        from fishtrack.analysis import load_run, result_path
        from fishtrack.gui.analysis import AnalysisDialog

        dialog = AnalysisDialog(self)
        if self.last_output is not None and result_path(self.last_output).is_file():
            try:
                dialog.set_run(0, load_run(result_path(self.last_output)))
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Cannot open results", str(error))
        dialog.exec()

    def selected_source(self) -> Path | None:
        """Resolve table selection by stored path identity, even after sorting or refresh."""
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        if item is None or not item.isSelected():
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def choose_folder(self) -> None:
        """Choose and immediately remember the current footage folder."""
        name = QFileDialog.getExistingDirectory(self, "Choose footage folder", str(self.folder))
        if name:
            self.set_folder(Path(name))

    def set_folder(self, folder: Path) -> None:
        """Load a chosen folder while keeping invalid settings from losing the folder preference."""
        self.folder = folder
        self.preview_path = None
        self.refresh()
        self.save_preferences()

    def refresh(self) -> None:
        """Scan only immediate eligible files and retain a selection by path when possible."""
        selected = self.selected_source()
        self.folder_label.setText(str(self.folder))
        try:
            inputs = scan_folder(self.folder)
        except OSError as error:
            LOGGER.warning("Cannot read folder: %s", error)
            self.status.setText(f"Cannot read folder: {error}")
            inputs = []
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(inputs))
        for row, item in enumerate(inputs):
            filename = QTableWidgetItem(item.path.name)
            filename.setData(Qt.ItemDataRole.UserRole, str(item.path))
            filename.setToolTip(str(item.path))
            size = QTableWidgetItem()
            size.setData(Qt.ItemDataRole.DisplayRole, round(item.size_bytes / 1024**2, 2))
            self.table.setItem(row, 0, filename)
            self.table.setItem(row, 1, size)
        self.table.setSortingEnabled(True)
        chosen_row = 0
        for row in range(self.table.rowCount()):
            if Path(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)) == selected:
                chosen_row = row
                break
        self.table.clearSelection()
        if inputs:
            self.table.setCurrentCell(chosen_row, 0)
            self.table.selectRow(chosen_row)
        self.table.blockSignals(False)
        self.file_count.setText(f"{len(inputs)} MP4 file(s)")
        self._selection_changed()

    def _selection_changed(self) -> None:
        """Show the selected starting frame without resetting a completed preview on refresh."""
        source = self.selected_source()
        if source is None:
            self.preview.clear("No eligible MP4 selected")
            self.preview_path = None
            self.metadata_label.setText("Choose a folder containing MP4 files")
        elif source != self.preview_path:
            self._load_preview()
        self._update_output()

    def _load_preview(self) -> None:
        """Decode the selected starting frame, reporting invalid inputs in the main window."""
        if self.worker is not None:
            return
        source = self.selected_source()
        if source is None:
            return
        start = self.options.controls["start"].value()
        try:
            frame, metadata = preview_frame(source, start)
        except (ValueError, OSError) as error:
            self.preview.clear(str(error))
            self.status.setText(str(error))
            LOGGER.warning("Preview unavailable: %s", error)
            return
        self.preview.set_frame(frame)
        self.preview_path, self.preview_start = source, start
        self.metadata_label.setText(
            f"{metadata.width} × {metadata.height} • {metadata.fps:.3f} fps • "
            f"{metadata.duration:.2f} s • starting at {start:.2f} s"
        )

    def _settings_changed(self) -> None:
        """Update naming immediately and debounce start-time preview changes."""
        self._update_output()
        if self.worker is None and self.options.controls["start"].value() != self.preview_start:
            self.preview_timer.start()

    def _update_output(self) -> None:
        """Show the resolved output and disable processing when settings cannot be used."""
        source = self.selected_source()
        valid = False
        if source is None:
            self.output_label.setText("Output: select an input video")
        else:
            try:
                destination = output_path(source, self.options.values())
            except ValueError as error:
                self.output_label.setText(str(error))
            else:
                self.output_label.setText(f"Output file\n{destination}")
                valid = True
        self.process_button.setEnabled(valid and self.worker is None)

    def save_preferences(self) -> None:
        """Persist the folder and retain previous settings during invalid edits."""
        try:
            options = self.options.values()
        except ValueError as error:
            LOGGER.warning("Keeping previously valid settings: %s", error)
            options = self.preferences.options
        preferences = Preferences(
            str(self.folder), self.saveGeometry().toBase64().data().decode("ascii"), options
        )
        try:
            self.store.save(preferences)
        except (OSError, ValueError) as error:
            LOGGER.warning("Could not save preferences: %s", error)
        else:
            self.preferences = preferences

    def start_processing(self) -> None:
        """Validate a complete job, persist its choices, then launch the processing thread."""
        source = self.selected_source()
        if self.worker is not None or source is None:
            return
        try:
            options = self.options.values()
            destination = output_path(source, options)
            configuration = processing_config(options, self.project_root)
            self._check_dependencies(configuration)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot start processing", str(error))
            return
        self.preview_timer.stop()
        self.save_preferences()
        debug_dir = diagnostic_path(destination) if options.diagnostics else None
        worker = ProcessingWorker(
            source, destination, detection_config(options), configuration, debug_dir
        )
        self.worker = worker
        worker.progress.connect(self._job_progress)
        worker.preview.connect(self.preview.set_frame)
        worker.review_requested.connect(self._review_regions)
        worker.finished.connect(self._job_finished)
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.status.setText("Starting processing…")
        LOGGER.info("Processing %s → %s", source, destination)
        worker.start()

    @staticmethod
    def _check_dependencies(configuration: ProcessingConfig) -> None:
        """Report missing optional model files/packages before starting a long-running job."""
        if configuration.backend == "sam2":
            for module in ("torch", "sam2"):
                if importlib.util.find_spec(module) is None:
                    raise ValueError("SAM 2 is not installed. Run scripts/make-venv.sh first.")
            if not configuration.sam2_checkpoint.is_file():
                raise ValueError(
                    f"SAM 2 checkpoint not found: {configuration.sam2_checkpoint}. "
                    "Choose a checkpoint or run scripts/make-venv.sh."
                )
        if configuration.auto_roi and (
            importlib.util.find_spec("transformers") is None
            or not (configuration.grounding_model / "model.safetensors").is_file()
        ):
            raise ValueError(
                "Grounding DINO is not installed. Run scripts/make-venv.sh "
                "or choose manual selection."
            )

    def _set_busy(self, busy: bool) -> None:
        """Freeze job inputs while allowing cancellation and keeping the window responsive."""
        for widget in (
            self.folder_button,
            self.refresh_button,
            self.table,
            self.options,
            self.defaults_button,
        ):
            widget.setEnabled(not busy)
        self.process_button.setEnabled(not busy and self.selected_source() is not None)
        self.cancel_button.setEnabled(busy)

    def _job_progress(self, completed: int, total: int, stage: str) -> None:
        """Show stage-aware progress without inventing a percentage for model loading."""
        self.status.setText(stage if not total else f"{stage} • {completed}/{total} frames")
        self.progress.setRange(0, total if total else 0)
        self.progress.setValue(completed)

    def _review_regions(self, request: RegionRequest) -> None:
        """Run the region-review dialog on the main thread and release the waiting worker."""
        if self.worker is None or self.worker.cancel_event.is_set():
            request.ready.set()
            return
        dialog = RegionDialog(request.frame, request.initial, self)
        self.region_dialog = dialog
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                request.result = dialog.selected_regions()
        finally:
            request.ready.set()
            self.region_dialog = None
            dialog.deleteLater()

    def cancel_processing(self) -> None:
        """Request cancellation while allowing the engine to clean up partial output."""
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current processing step…")
            if self.region_dialog is not None:
                self.region_dialog.reject()

    def _job_finished(self) -> None:
        """Publish job disposition only after the processing thread has fully exited."""
        worker = self.worker
        if worker is None:
            return
        worker.wait()
        result = worker.result
        self.worker = None
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if result.status == "completed" else 0)
        if result.status == "completed":
            self.last_output = worker.output
            self.output_button.setEnabled(True)
            self.refresh()
            self.metadata_label.setText(f"Completed preview • {worker.output.name}")
            self.status.setText(f"Completed • {result.frames} frames • {worker.output.name}")
        elif result.status == "cancelled":
            self.status.setText("Processing cancelled")
        else:
            self.status.setText(f"Processing failed: {result.message}")
        worker.deleteLater()
        self._update_output()
        if self.close_requested:
            QTimer.singleShot(0, self.close)

    def _restore_defaults(self) -> None:
        """Reset processing controls from the immutable bundled JSON."""
        self.options.set_values(default_options())
        self._settings_changed()

    def _open_source(self, *arguments: object) -> None:
        """Open a double-clicked source in the operating system's video viewer."""
        source = self.selected_source()
        if source is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(source.resolve())))

    def _open_result(self) -> None:
        """Open the most recently completed annotated MP4 in the system video viewer."""
        if self.last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output.resolve())))

    def _open_logs(self) -> None:
        """Open the platform-specific application log directory."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(standard_log_directory())))

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancel active work and defer closing until worker-owned cleanup has completed."""
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Processing is running",
                "Cancel processing and close FishTrack?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.close_requested = True
                self.cancel_processing()
            event.ignore()
            return
        self.save_preferences()
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """Safely finish any active thread before destroying Qt objects during application exit."""
        if self.worker is not None:
            self.cancel_processing()
            self.worker.wait()
        logging.getLogger().removeHandler(self.log_handler)
