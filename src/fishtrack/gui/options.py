"""Grouped Qt controls translated into validated, persistable application options."""

from dataclasses import asdict

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fishtrack.gui.settings import UiOptions, validated_options


class OptionsPanel(QTabWidget):
    """Keep model, detection and output settings grouped as in the project wireframe."""

    changed = Signal()

    def __init__(self, options: UiOptions, parent: QWidget | None = None) -> None:
        """Create all controls from the shipped/validated settings."""
        super().__init__(parent)
        self.controls: dict[str, QWidget] = {}
        tracking, tracking_layout = self._tab("Tracking")
        form = QFormLayout()
        tracking_layout.addLayout(form)
        self._combo(
            form, "Method", "backend", [("SAM 2", "sam2"), ("Conventional CV", "conventional")]
        )
        self._combo(
            form,
            "Find fish",
            "selection",
            [("Automatic proposals", "automatic"), ("Select manually", "manual")],
        )
        self._number(form, "Start (seconds)", "start", 0, 864000, decimals=2)
        self._number(form, "Duration (0 = remainder)", "duration", 0, 864000, decimals=2)
        self._number(form, "Frame limit", "max_frames", 0, 10000000)
        self.sam_group = QGroupBox("SAM 2")
        sam = QFormLayout(self.sam_group)
        self._combo(
            sam,
            "Model size",
            "sam2_model",
            [
                (name, value)
                for name, value in [
                    ("Tiny", "tiny"),
                    ("Small", "small"),
                    ("Base plus", "base-plus"),
                    ("Large", "large"),
                ]
            ],
        )
        self._combo(sam, "Compute device", "sam2_device", self._devices())
        self._number(sam, "Minimum mask area (px)", "sam2_min_area", 1, 10000000)
        self._path(sam, "Checkpoint (blank = default)", "sam2_checkpoint", False)
        tracking_layout.addWidget(self.sam_group)
        self.limit_note = QLabel(
            "SAM 2 processes at most 240 frames per run. Output is video only."
        )
        self.limit_note.setWordWrap(True)
        tracking_layout.addWidget(self.limit_note)
        tracking_layout.addStretch()

        _, detection_layout = self._tab("Grounding DINO")
        detection = QFormLayout()
        detection_layout.addLayout(detection)
        self._check(detection, "Review proposed boxes", "review")
        self._line(detection, "Find objects described as", "grounding_prompt")
        self._number(detection, "Detection threshold", "grounding_threshold", 0.01, 1, decimals=2)
        self._number(detection, "Text threshold", "grounding_text_threshold", 0.01, 1, decimals=2)
        self._number(detection, "Maximum proposals", "max_auto_rois", 1, 100)
        self._combo(detection, "Compute device", "grounding_device", self._devices())
        self._path(detection, "Model folder (blank = default)", "grounding_model", True)
        note = QLabel(
            "Proposals are made on the starting frame. Review lets you remove incorrect boxes "
            "and add missed fish before tracking."
        )
        note.setWordWrap(True)
        detection_layout.addWidget(note)
        detection_layout.addStretch()

        _, motion_layout = self._tab("Motion")
        motion = QFormLayout()
        motion_layout.addLayout(motion)
        for label, key, low, high, decimals in (
            ("Minimum area (px)", "min_area", 1, 10000000, 0),
            ("Maximum area (px)", "max_area", 1, 100000000, 0),
            ("Motion threshold", "motion_threshold", 0.1, 255, 1),
            ("Local motion (px/frame)", "min_residual_motion", 0.1, 1000, 1),
            ("Tracking distance (px)", "max_distance", 1, 5000, 1),
            ("Missing frames tolerated", "max_missing", 0, 10000, None),
            ("Detections before tracking", "min_hits", 1, 1000, None),
            ("Appearance threshold", "tracking_threshold", 0.001, 1, 3),
        ):
            self._number(motion, label, key, low, high, decimals)
        self._check(motion, "Compensate camera motion", "compensate")
        self._check(motion, "Filter local background motion", "local_motion")
        motion_layout.addStretch()

        _, output_layout = self._tab("Output")
        output = QFormLayout()
        output_layout.addLayout(output)
        self._line(output, "Filename (blank = automatic)", "output_name")
        self._check(output, "Save beside the input", "output_beside_input")
        self._path(output, "Output folder", "output_folder", True)
        self._check(output, "Draw fish outlines", "outline")
        self._check(output, "Show tracking IDs", "track_ids")
        self._check(output, "Show bounding boxes", "bounding_boxes")
        self._check(output, "Save diagnostic masks and CSV", "diagnostics")
        note = QLabel(
            "Existing files are never overwritten. Automatic names receive a number when needed."
        )
        note.setWordWrap(True)
        output_layout.addWidget(note)
        output_layout.addStretch()
        self.controls["output_name"].setPlaceholderText("e.g. reef-tracked.mp4")
        self.controls["backend"].currentIndexChanged.connect(self._update_backend)
        self.controls["selection"].currentIndexChanged.connect(self._update_backend)
        self.controls["output_beside_input"].toggled.connect(self._update_output)
        self.set_values(options)

    @staticmethod
    def _devices() -> list[tuple[str, str]]:
        """Return human-readable device names and their processing values."""
        return [
            ("CPU", "cpu"),
            ("Automatic", "auto"),
            ("Apple GPU (MPS)", "mps"),
            ("NVIDIA GPU (CUDA)", "cuda"),
        ]

    def _tab(self, label: str) -> tuple[QWidget, QVBoxLayout]:
        """Add a scrolling options page so controls fit smaller desktop screens."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 14, 12, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.addTab(scroll, label)
        return content, layout

    def _register(self, key: str, widget: QWidget) -> None:
        """Register a named settings widget for persistence and tests."""
        widget.setObjectName(key)
        self.controls[key] = widget

    def _changed(self, *arguments: object) -> None:
        """Notify the controller that a presented setting has changed."""
        self.changed.emit()

    def _number(
        self,
        form: QFormLayout,
        label: str,
        key: str,
        low: float,
        high: float,
        decimals: int | None = None,
    ) -> None:
        """Add a bounded integer or floating-point control."""
        if decimals is None:
            widget = QSpinBox()
            widget.setRange(int(low), int(high))
        else:
            widget = QDoubleSpinBox()
            widget.setDecimals(decimals)
            widget.setRange(low, high)
            widget.setSingleStep(10**-decimals if decimals else 1)
        widget.valueChanged.connect(self._changed)
        form.addRow(label, widget)
        self._register(key, widget)

    def _combo(
        self, form: QFormLayout, label: str, key: str, choices: list[tuple[str, str]]
    ) -> None:
        """Add a combo whose display labels do not leak into processing configuration."""
        widget = QComboBox()
        for text, value in choices:
            widget.addItem(text, value)
        widget.currentIndexChanged.connect(self._changed)
        form.addRow(label, widget)
        self._register(key, widget)

    def _check(self, form: QFormLayout, label: str, key: str) -> None:
        """Add a persisted Boolean option."""
        widget = QCheckBox(label)
        widget.toggled.connect(self._changed)
        form.addRow(widget)
        self._register(key, widget)

    def _line(self, form: QFormLayout, label: str, key: str) -> None:
        """Add a persisted string option."""
        widget = QLineEdit()
        widget.textChanged.connect(self._changed)
        form.addRow(label, widget)
        self._register(key, widget)

    def _path(self, form: QFormLayout, label: str, key: str, directory: bool) -> None:
        """Add an editable filesystem path and an appropriate native picker."""
        row = QHBoxLayout()
        widget = QLineEdit()
        widget.textChanged.connect(self._changed)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(key, directory))
        row.addWidget(widget, 1)
        row.addWidget(button)
        form.addRow(label, row)
        self._register(key, widget)
        if key == "output_folder":
            self.output_browse = button

    def _browse(self, key: str, directory: bool) -> None:
        """Select a local directory or model checkpoint without downloading anything."""
        widget = self.controls[key]
        if directory:
            path = QFileDialog.getExistingDirectory(self, "Choose folder", widget.text())
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose checkpoint", widget.text(), "Model (*.pt)"
            )
        if path:
            widget.setText(path)

    def _update_backend(self, *arguments: object) -> None:
        """Enable relevant groups and keep acquisition choices valid for the selected method."""
        sam2 = self.controls["backend"].currentData() == "sam2"
        selection = self.controls["selection"]
        selected = selection.currentData()
        choices = (
            [("Automatic proposals", "automatic"), ("Select manually", "manual")]
            if sam2
            else [
                ("Automatic motion", "motion"),
                ("Automatic proposals", "automatic"),
                ("Select manually", "manual"),
            ]
        )
        selection.blockSignals(True)
        selection.clear()
        for label, value in choices:
            selection.addItem(label, value)
        index = selection.findData(selected)
        selection.setCurrentIndex(max(0, index))
        selection.blockSignals(False)
        self.sam_group.setEnabled(sam2)
        self.setTabEnabled(1, selection.currentData() == "automatic")
        self.setTabEnabled(2, not sam2)
        self.limit_note.setVisible(sam2)
        frame_limit = self.controls["max_frames"]
        previous_limit = frame_limit.value()
        frame_limit.setRange(1 if sam2 else 0, 240 if sam2 else 10000000)
        frame_limit.setSpecialValueText("" if sam2 else "Unlimited")
        if sam2 and (previous_limit == 0 or previous_limit > 240):
            frame_limit.setValue(240)

    def _update_output(self, *arguments: object) -> None:
        """Enable an explicit output folder only when not saving beside the input."""
        enabled = not self.controls["output_beside_input"].isChecked()
        self.controls["output_folder"].setEnabled(enabled)
        self.output_browse.setEnabled(enabled)

    def set_values(self, options: UiOptions) -> None:
        """Restore values without intermediate validation or processing side effects."""
        backend = self.controls["backend"]
        backend.blockSignals(True)
        backend.setCurrentIndex(backend.findData(options.backend))
        backend.blockSignals(False)
        self._update_backend()
        frame_limit = self.controls["max_frames"]
        frame_limit.blockSignals(True)
        frame_limit.setRange(
            1 if options.backend == "sam2" else 0, 240 if options.backend == "sam2" else 10000000
        )
        frame_limit.blockSignals(False)
        for key, value in asdict(options).items():
            widget = self.controls[key]
            widget.blockSignals(True)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            else:
                widget.setText(value)
            widget.blockSignals(False)
        self._update_backend()
        self._update_output()

    def values(self) -> UiOptions:
        """Return validated current settings or raise a user-readable ValueError."""
        values = {}
        for key, widget in self.controls.items():
            if isinstance(widget, QComboBox):
                values[key] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[key] = widget.value()
            else:
                values[key] = widget.text().strip()
        return validated_options(values)
