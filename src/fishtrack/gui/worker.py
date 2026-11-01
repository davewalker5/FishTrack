"""Background processing with queued previews and main-thread region-review requests."""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from fishtrack.control import ProcessingCancelled, ProcessingHooks, Regions
from fishtrack.detection import DetectionConfig, Image
from fishtrack.video import ProcessingConfig, process_video

LOGGER = logging.getLogger(__name__)


@dataclass
class RegionRequest:
    """One synchronized request for the GUI thread to review a starting frame."""

    frame: Image
    initial: Regions
    ready: threading.Event = field(default_factory=threading.Event)
    result: Regions | None = None


@dataclass(frozen=True)
class JobResult:
    """Completed worker disposition consumed only after its thread has stopped."""

    status: str
    frames: int = 0
    message: str = ""


class ProcessingWorker(QThread):
    """Run the existing processing engine without blocking the desktop event loop."""

    progress = Signal(int, int, str)
    preview = Signal(object)
    review_requested = Signal(object)

    def __init__(
        self,
        source: Path,
        output: Path,
        detection: DetectionConfig,
        options: ProcessingConfig,
        debug_dir: Path | None = None,
    ) -> None:
        """Capture one immutable job; start is invoked by the main-window controller."""
        super().__init__()
        self.source, self.output = source, output
        self.detection, self.options, self.debug_dir = detection, options, debug_dir
        self.cancel_event = threading.Event()
        self.result = JobResult("pending")
        self.last_preview = 0.0
        self.last_frame: Image | None = None

    def cancel(self) -> None:
        """Request cooperative cancellation, including while a region review is pending."""
        self.cancel_event.set()

    def _preview(self, frame: Image) -> None:
        """Throttle copied frames so a fast conventional job cannot flood the UI queue."""
        self.last_frame = frame
        now = time.monotonic()
        if now - self.last_preview >= 0.2:
            self.preview.emit(frame.copy())
            self.last_preview = now

    def _review(self, frame: Image, regions: Regions) -> Regions:
        """Wait for a main-thread dialog without blocking cancellation or using Tk."""
        request = RegionRequest(frame.copy(), regions)
        self.review_requested.emit(request)
        while not request.ready.wait(0.1):
            if self.cancel_event.is_set():
                raise ProcessingCancelled("Processing cancelled during fish selection")
        if self.cancel_event.is_set() or request.result is None:
            raise ProcessingCancelled("Fish selection cancelled")
        return request.result

    def run(self) -> None:
        """Report all job failures at this thread boundary and always allow engine cleanup."""
        hooks = ProcessingHooks(
            cancelled=self.cancel_event.is_set,
            progress=self.progress.emit,
            preview=self._preview,
            select=self._review,
        )
        try:
            count = process_video(
                self.source, self.output, self.detection, self.options, self.debug_dir, hooks
            )
        except ProcessingCancelled as error:
            LOGGER.info("%s", error)
            self.result = JobResult("cancelled", message=str(error))
        except Exception as error:
            # A thread-boundary failure must be surfaced, with traceback in the application log.
            LOGGER.exception("Processing failed")
            self.result = JobResult("failed", message=str(error))
        else:
            if self.last_frame is not None:
                self.preview.emit(self.last_frame.copy())
            self.result = JobResult("completed", count)
