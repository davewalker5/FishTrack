"""Optional saved-run analysis and explicit manual comparison, without model loading."""

from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fishtrack.analysis import (
    TrackingRun,
    export_csv,
    load_run,
    save_run,
    track_events,
    track_statistics,
    validate_comparison,
)
from fishtrack.gui.canvas import FrameView


class TrajectoryView(FrameView):
    """Draw source-coordinate trajectories with separate segments across every gap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a frame-backed or dimension-only image-space plot."""
        super().__init__(parent)
        self.selected: list[tuple[TrackingRun, int]] = []
        self.frame = 0
        self.color = "#25edc5"
        self.placeholder = "Open tracking results to view image-space trajectories"

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint paths, start/end markers and selected-frame observations in source axes."""
        if not self.selected:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07323e"))
        if not self.image.isNull():
            painter.drawImage(self.image_rect(), self.image)
        run = self.selected[0][0]
        target = self.image_rect()
        if target.isEmpty():
            scale = min((self.width() - 60) / run.width, (self.height() - 60) / run.height)
            target = QRectF(
                (self.width() - run.width * scale) / 2,
                (self.height() - run.height * scale) / 2,
                run.width * scale,
                run.height * scale,
            )
            painter.fillRect(target, QColor("#123f49"))
        painter.setPen(QColor("white"))
        painter.drawRect(target)
        painter.drawText(
            int(target.right()) - 110, int(target.bottom()) + 20, f"{run.width} × {run.height} px"
        )
        for current, identity in self.selected:
            pen = QPen(QColor(self.color), 3)
            painter.setPen(pen)
            reports = [item for item in current.track(identity) if item.x is not None]
            previous = None
            for index, item in enumerate(reports):
                point = QPointF(
                    target.left() + item.x / run.width * target.width(),
                    target.top() + item.y / run.height * target.height(),
                )
                if previous is not None and item.frame - previous[0] == current.frame_step:
                    painter.drawLine(previous[1], point)
                if index in (0, len(reports) - 1):
                    painter.drawEllipse(point, 4, 4)
                    painter.drawText(point, " Start" if index == 0 else " End")
                if item.frame == self.frame:
                    painter.drawEllipse(point, 8, 8)
                previous = item.frame, point
        painter.end()


def observed_intervals(run: TrackingRun, identity: int) -> list[tuple[int, int]]:
    """Return half-open source-frame intervals with positions, preserving all internal gaps."""
    intervals: list[tuple[int, int]] = []
    for report in run.track(identity):
        if report.x is None:
            continue
        if intervals and report.frame == intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], report.frame + run.frame_step)
        else:
            intervals.append((report.frame, report.frame + run.frame_step))
    return intervals


class ContinuityView(QWidget):
    """Show observed frames against the full shared processing range, with genuine gaps."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an aligned two-row continuity chart using source-frame units."""
        super().__init__(parent)
        self.selected: list[tuple[TrackingRun, int] | None] = [None, None]
        self.frame = 0
        self.setFixedHeight(110)
        self.setAccessibleName("Track observation continuity in source frames")

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint aligned measured intervals and a shared current-frame cursor."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07323e"))
        painter.setPen(QColor("white"))
        painter.drawText(12, 22, "Track continuity · coloured = observed; dark = unavailable")
        selected = next((item for item in self.selected if item is not None), None)
        if selected is None:
            painter.end()
            return
        run = selected[0]
        start, count = run.start_frame, max(1, run.processed_frames)
        left, width = 160, max(1, self.width() - 180)
        for slot, item in enumerate(self.selected):
            top = 31 + slot * 29
            painter.setPen(QColor("white"))
            name = "No track selected"
            if item is not None:
                name = "Conventional CV" if item[0].backend == "conventional" else "SAM 2"
            painter.drawText(12, top + 18, name)
            painter.fillRect(QRectF(left, top, width, 19), QColor("#244650"))
            if item is None:
                continue
            color = "#e99f3d" if item[0].backend == "conventional" else "#25edc5"
            for first, end in observed_intervals(*item):
                painter.fillRect(
                    QRectF(
                        left + (first - start) / count * width,
                        top,
                        (end - first) / count * width,
                        19,
                    ),
                    QColor(color),
                )
        painter.setPen(QPen(QColor("white"), 1))
        cursor = left + (self.frame - start + 0.5) / count * width
        painter.drawLine(QPointF(cursor, 28), QPointF(cursor, 83))
        for fraction in (0, 0.5, 1):
            frame = start + round(fraction * max(0, run.processed_frames - 1))
            x = left + fraction * width
            painter.drawText(QPointF(min(x, self.width() - 65), 101), str(frame))
        painter.drawText(self.width() - 115, 22, "Source frames")
        painter.end()


class AnalysisDialog(QDialog):
    """Inspect standalone results, explicitly pair local tracks and export either run."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the optional analysis workflow with no tracking/model dependencies."""
        super().__init__(parent)
        self.setWindowTitle("FishTrack — Analysis")
        self.resize(1400, 900)
        self.runs: list[TrackingRun | None] = [None, None]
        self.source: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.next_source_frame: int | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.finished.connect(self.release_source)
        layout = QVBoxLayout(self)
        self.selectors = [QComboBox(), QComboBox()]
        self.names = [QLabel("No run loaded"), QLabel("No comparison loaded")]
        for slot in range(2):
            row = QHBoxLayout()
            button = QPushButton("Open results" if slot == 0 else "Open comparison")
            button.clicked.connect(lambda checked=False, index=slot: self.open_results(index))
            row.addWidget(button)
            row.addWidget(self.names[slot], 1)
            row.addWidget(self.selectors[slot])
            self.selectors[slot].currentIndexChanged.connect(self.refresh)
            for extension in ("csv", "json"):
                export = QPushButton(f"Export {extension.upper()}")
                export.clicked.connect(
                    lambda checked=False, index=slot, kind=extension: self.export(index, kind)
                )
                row.addWidget(export)
            layout.addLayout(row)
        pairing = QLabel(
            "Select corresponding tracks · Orange: CV · Green: SAM 2 · Playback is synchronized"
        )
        pairing.setToolTip(
            "Pair tracks manually: matching IDs do not identify the same animal. "
            "Both views share the source frame and playback controls."
        )
        pairing.setWordWrap(True)
        layout.addWidget(pairing)
        panels = QHBoxLayout()
        self.plots = [TrajectoryView(), TrajectoryView()]
        self.panel_titles = [QLabel("First run"), QLabel("Comparison run")]
        self.panel_status = [QLabel("No track selected"), QLabel("No track selected")]
        self.play_buttons = [QPushButton("Play"), QPushButton("Play")]
        self.plot = self.plots[0]
        for slot in range(2):
            panel = QVBoxLayout()
            panel.addWidget(self.panel_titles[slot])
            panel.addWidget(self.plots[slot], 1)
            panel.addWidget(self.panel_status[slot])
            self.play_buttons[slot].setEnabled(False)
            self.play_buttons[slot].clicked.connect(self.toggle_playback)
            panel.addWidget(self.play_buttons[slot])
            panels.addLayout(panel, 1)
        layout.addLayout(panels, 1)
        self.continuity = ContinuityView()
        layout.addWidget(self.continuity)
        controls = QHBoxLayout()
        source_button = QPushButton("Locate source video")
        source_button.clicked.connect(self.locate_source)
        controls.addWidget(source_button)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.inspect_frame)
        controls.addWidget(self.timeline, 1)
        self.frame_label = QLabel("Source frame")
        self.frame_label.setWordWrap(True)
        layout.addLayout(controls)
        layout.addWidget(self.frame_label)
        self.details_toggle = QToolButton()
        self.details_toggle.setText("Statistics and analysis settings")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(self.details_toggle)
        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_panel.setVisible(False)
        self.details_toggle.toggled.connect(self.toggle_details)
        layout.addWidget(self.details_panel)
        thresholds = QHBoxLayout()
        self.position_threshold = QDoubleSpinBox()
        self.position_threshold.setRange(1, 100000)
        self.position_threshold.setValue(100)
        self.area_threshold = QDoubleSpinBox()
        self.area_threshold.setRange(1.01, 1000)
        self.area_threshold.setValue(2)
        thresholds.addWidget(QLabel("Possible position jump (px)"))
        thresholds.addWidget(self.position_threshold)
        thresholds.addWidget(QLabel("Possible box-area change (ratio)"))
        thresholds.addWidget(self.area_threshold)
        for threshold in (self.position_threshold, self.area_threshold):
            threshold.valueChanged.connect(self.refresh)
        details_layout.addLayout(thresholds)
        statistics_layout = QHBoxLayout()
        self.detail_views = [QPlainTextEdit(), QPlainTextEdit()]
        self.details = self.detail_views[0]
        for view in self.detail_views:
            view.setReadOnly(True)
            view.setMaximumHeight(210)
            statistics_layout.addWidget(view)
        details_layout.addLayout(statistics_layout)
        note = QLabel(
            "Measurements describe tracking output, not accuracy or behaviour. Camera movement, "
            "perspective and tracker errors affect image-space distances. Missing positions are "
            "unavailable; path length excludes gaps. Video is optional for numerical analysis."
        )
        note.setWordWrap(True)
        details_layout.addWidget(note)

    def toggle_details(self, expanded: bool) -> None:
        """Reveal optional statistics/settings while giving previews the remaining space.

        :param expanded: Whether the details panel should be visible.
        """
        self.details_panel.setVisible(expanded)
        self.details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def open_results(self, slot: int) -> None:
        """Choose and validate a saved run, preserving current selections on failure."""
        name, _ = QFileDialog.getOpenFileName(self, "Open tracking results", "", "JSON (*.json)")
        if not name:
            return
        try:
            self.set_run(slot, load_run(Path(name)))
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot open results", str(error))

    def set_run(self, slot: int, run: TrackingRun) -> None:
        """Load a validated run and reject incompatible comparisons before changing state."""
        other = self.runs[1 - slot]
        if other is not None:
            validate_comparison(run, other)
        self.pause_playback()
        self.runs[slot] = run
        self.names[slot].setText(f"{run.backend} · {run.completion} · {run.run_id[:8]}")
        selector = self.selectors[slot]
        selector.blockSignals(True)
        selector.clear()
        for identity in run.track_ids:
            selector.addItem(f"Track {identity}", identity)
        selector.blockSignals(False)
        self.timeline.setRange(
            run.start_frame, max(run.start_frame, run.start_frame + run.processed_frames - 1)
        )
        if other is None:
            self.release_source()
        for plot in self.plots:
            if self.source is None:
                plot.clear("Source-frame coordinate representation")
        self.refresh()
        self.inspect_frame(self.timeline.value())

    def refresh(self, *arguments: object) -> None:
        """Refresh the explicitly selected tracks, descriptive statistics and event lists."""
        selected = []
        for slot, (run, selector, view) in enumerate(
            zip(self.runs, self.selectors, self.detail_views, strict=True)
        ):
            plot = self.plots[slot]
            plot.selected = []
            self.panel_status[slot].setText("No track selected")
            lines = []
            view.setPlainText("No observations in the selected results.")
            if run is not None:
                name = "Conventional CV" if run.backend == "conventional" else "SAM 2"
                self.panel_titles[slot].setText(f"{name} · {run.completion}")
                plot.color = "#e99f3d" if run.backend == "conventional" else "#25edc5"
            if run is None or selector.currentData() is None:
                selected.append(None)
                continue
            identity = selector.currentData()
            selected.append((run, identity))
            plot.selected = [(run, identity)]
            lines.append(f"{run.backend} / {run.run_id[:8]} / Track {identity} ({run.completion})")
            for key, value in track_statistics(run, identity).items():
                rendered = "Unavailable" if value is None else str(round(value, 4))
                lines.append(f"  {key.replace('_', ' ')}: {rendered}")
            events = track_events(
                run, identity, self.position_threshold.value(), self.area_threshold.value()
            )
            lines.append(
                "  Events: "
                + "; ".join(
                    f"{event.frame}: {event.kind}"
                    + (" (inferred)" if event.inferred else " (reported)")
                    for event in events
                )
            )
            view.setPlainText("\n".join(lines))
        self.continuity.selected = selected
        self.continuity.update()
        for plot in self.plots:
            plot.update()
        self.inspect_frame(self.timeline.value())

    def locate_source(self) -> None:
        """Locate a fingerprint-matching source video; never silently show unrelated footage."""
        name, _ = QFileDialog.getOpenFileName(self, "Locate source video", "", "Video (*.mp4)")
        if not name:
            return
        try:
            self.set_source(Path(name))
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot use source video", str(error))

    def set_source(self, path: Path) -> None:
        """Open one fingerprint-checked decoder shared by both trajectory views.

        :param path: Original source video, independent of filename or location.
        :raises ValueError: If no run is loaded, identity differs or the file cannot open.
        """
        from fishtrack.analysis import source_fingerprint

        run = next((run for run in self.runs if run is not None), None)
        if run is None:
            raise ValueError("Open tracking results before locating the source video")
        if source_fingerprint(path) != run.fingerprint:
            raise ValueError("Selected video does not match the saved source fingerprint")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("Cannot open source video")
        self.release_source()
        self.capture, self.source = capture, path
        for button in self.play_buttons:
            button.setEnabled(run.processed_frames > 0)
        self.inspect_frame(self.timeline.value())

    def pause_playback(self) -> None:
        """Pause both views together and reset their control labels."""
        self.timer.stop()
        for button in self.play_buttons:
            button.setText("Play")

    def toggle_playback(self) -> None:
        """Play or pause the common source timeline from either panel's control."""
        if self.timer.isActive():
            self.pause_playback()
            return
        run = next((run for run in self.runs if run is not None), None)
        if self.capture is None or run is None or not run.processed_frames:
            return
        if self.timeline.value() == self.timeline.maximum():
            self.timeline.setValue(self.timeline.minimum())
        self.timer.start(max(1, round(1000 / run.fps)))
        for button in self.play_buttons:
            button.setText("Pause")

    def advance_frame(self) -> None:
        """Advance both copies once per source-frame interval and pause at the clip end."""
        frame = self.timeline.value()
        if frame >= self.timeline.maximum():
            self.pause_playback()
            return
        self.timeline.setValue(frame + 1)
        if self.timeline.value() == self.timeline.maximum():
            self.pause_playback()

    def release_source(self, *arguments: object) -> None:
        """Stop playback and release the shared decoder when replacing footage or closing."""
        self.pause_playback()
        if self.capture is not None:
            self.capture.release()
        self.capture, self.source, self.next_source_frame = None, None, None
        for button in self.play_buttons:
            button.setEnabled(False)

    def inspect_frame(self, frame: int) -> None:
        """Decode once and display the identical source frame behind each selected track."""
        reports = []
        for slot, plot in enumerate(self.plots):
            plot.frame = frame
            text = "No track selected"
            for run, identity in plot.selected:
                item = next((item for item in run.track(identity) if item.frame == frame), None)
                text = f"{run.backend}: {item.status if item else 'no report'}"
                if item and item.x is not None:
                    text += f" ({item.x}, {item.y})"
                reports.append(text)
            self.panel_status[slot].setText(text)
        run = next((run for run in self.runs if run is not None), None)
        seconds = frame / run.fps if run else 0
        self.frame_label.setText(f"Frame {frame} · {seconds:.3f} s · " + " / ".join(reports))
        if self.capture is not None:
            if self.next_source_frame != frame:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = self.capture.read()
            if success:
                self.next_source_frame = frame + 1
                for plot in self.plots:
                    plot.set_frame(image)
            else:
                self.next_source_frame = None
                self.pause_playback()
                for plot in self.plots:
                    plot.clear("Source frame cannot be decoded")
        for plot in self.plots:
            plot.update()
        self.continuity.frame = frame
        self.continuity.update()

    def export(self, slot: int, kind: str) -> None:
        """Export either loaded run without overwriting existing files."""
        run = self.runs[slot]
        if run is None:
            return
        name, _ = QFileDialog.getSaveFileName(
            self, "Export tracking data", "", f"{kind.upper()} (*.{kind})"
        )
        if not name:
            return
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix(f".{kind}")
        try:
            (export_csv if kind == "csv" else save_run)(run, path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot export results", str(error))
