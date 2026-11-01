"""Aspect-preserving preview and editable source-coordinate region selection."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from fishtrack.control import Regions
from fishtrack.detection import Image
from fishtrack.selection import region_at_point, selection_box


class FrameView(QWidget):
    """Show BGR video frames without stretching or retaining borrowed NumPy memory."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty, resizable video preview."""
        super().__init__(parent)
        self.image = QImage()
        self.placeholder = "Choose a folder and select an MP4 to preview"
        self.setMinimumSize(320, 180)
        self.setAccessibleName("Video preview")

    def set_frame(self, frame: Image) -> None:
        """Copy a BGR image into Qt-owned memory and repaint."""
        array = np.ascontiguousarray(frame)
        height, width = array.shape[:2]
        self.image = QImage(
            array.data, width, height, array.strides[0], QImage.Format.Format_BGR888
        ).copy()
        self.update()

    def clear(self, message: str) -> None:
        """Replace any previous image with an explanatory placeholder."""
        self.image = QImage()
        self.placeholder = message
        self.update()

    def image_rect(self) -> QRectF:
        """Return the fitted image rectangle in widget coordinates, including letterboxing."""
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def source_point(self, point: QPointF, clamp: bool = False) -> tuple[float, float] | None:
        """Map a widget position to source pixels, rejecting letterbox clicks unless clamped."""
        rectangle = self.image_rect()
        if rectangle.isEmpty() or (not clamp and not rectangle.contains(point)):
            return None
        x = (point.x() - rectangle.left()) * self.image.width() / rectangle.width()
        y = (point.y() - rectangle.top()) * self.image.height() / rectangle.height()
        return max(0.0, min(float(self.image.width()), x)), max(
            0.0, min(float(self.image.height()), y)
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint a fitted frame on an underwater-coloured background."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07323e"))
        if self.image.isNull():
            painter.setPen(QColor("#b7d8dc"))
            painter.drawText(
                self.rect().adjusted(20, 20, -20, -20),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.placeholder,
            )
        else:
            painter.drawImage(self.image_rect(), self.image)
        painter.end()


class RegionCanvas(FrameView):
    """Edit rectangles stored in original video coordinates, independent of preview size."""

    changed = Signal()

    def __init__(self, frame: Image, regions: Regions = (), parent: QWidget | None = None) -> None:
        """Display an initial frame and any proposed regions."""
        super().__init__(parent)
        self.set_frame(frame)
        self.regions = list(regions)
        self.drag_start: tuple[float, float] | None = None
        self.drag_end: tuple[float, float] | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Fish selection canvas")

    def _widget_rectangle(self, box: tuple[float, float, float, float]) -> QRectF:
        """Map one source rectangle into the currently fitted preview area."""
        target = self.image_rect()
        scale = target.width() / self.image.width()
        x, y, width, height = box
        return QRectF(
            target.left() + x * scale, target.top() + y * scale, width * scale, height * scale
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        """Overlay selected regions and the in-progress drag without modifying the frame."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#ffe16b"), 2))
        for identifier, box in enumerate(self.regions, start=1):
            rectangle = self._widget_rectangle(box)
            painter.drawRect(rectangle)
            painter.drawText(rectangle.topLeft() + QPointF(4, 17), str(identifier))
        if self.drag_start is not None and self.drag_end is not None:
            x, y = self.drag_start
            end_x, end_y = self.drag_end
            painter.drawRect(
                self._widget_rectangle(
                    (min(x, end_x), min(y, end_y), abs(end_x - x), abs(end_y - y))
                )
            )
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start drawing or remove the smallest region containing a right/control click."""
        point = self.source_point(event.position())
        if point is None:
            return
        remove = event.button() == Qt.MouseButton.RightButton or bool(
            event.modifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )
        if remove:
            index = region_at_point(self.regions, point)
            if index is not None:
                self.regions.pop(index)
                self.changed.emit()
                self.update()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = self.drag_end = point
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update a drag in source coordinates, including drags outside the preview edge."""
        if self.drag_start is not None:
            self.drag_end = self.source_point(event.position(), clamp=True)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Accept a sufficiently large rectangle and discard accidental tiny drags."""
        if event.button() != Qt.MouseButton.LeftButton or self.drag_start is None:
            return
        end = self.source_point(event.position(), clamp=True)
        if end is not None:
            try:
                region = selection_box(
                    self.drag_start, end, 1.0, self.image.width(), self.image.height()
                )
            except ValueError:
                # A click or tiny drag is not a new selection.
                region = None
            if region is not None:
                self.regions.append(region)
                self.changed.emit()
        self.drag_start = self.drag_end = None
        self.update()

    def undo(self) -> None:
        """Remove the most recently accepted rectangle."""
        if self.regions:
            self.regions.pop()
            self.changed.emit()
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle Backspace locally and leave dialog navigation to Qt."""
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.undo()
        else:
            super().keyPressEvent(event)


class RegionDialog(QDialog):
    """Review automatic proposals or manually select fish using the same Qt interface."""

    def __init__(self, frame: Image, regions: Regions = (), parent: QWidget | None = None) -> None:
        """Build a modal selection dialog which never accepts an empty selection."""
        super().__init__(parent)
        self.setWindowTitle("FishTrack • Review fish selections")
        self.resize(1060, 720)
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "Right-click a box to remove it. Drag to add a fish. "
            "Backspace undoes the last selection. Enter starts tracking."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.canvas = RegionCanvas(frame, regions)
        layout.addWidget(self.canvas, 1)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.accept_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.accept_button.setText("Track selected fish")
        self.accept_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.canvas.changed.connect(self._update_count)
        self._update_count()

    def _update_count(self) -> None:
        """Keep acceptance disabled until at least one fish has been selected."""
        count = len(self.canvas.regions)
        self.count_label.setText(
            f"{count} fish selected" if count else "Select at least one fish to continue"
        )
        self.accept_button.setEnabled(count > 0)

    def accept(self) -> None:
        """Confirm only nonempty regions, including keyboard-triggered acceptance."""
        if self.canvas.regions:
            super().accept()

    def selected_regions(self) -> Regions:
        """Return the accepted source-coordinate rectangles in displayed ID order."""
        return tuple(self.canvas.regions)
