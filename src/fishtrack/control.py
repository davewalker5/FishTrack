"""Optional processing callbacks shared by command-line and desktop callers."""

from collections.abc import Callable
from dataclasses import dataclass

from fishtrack.detection import Image

Regions = tuple[tuple[int, int, int, int], ...]


class ProcessingCancelled(Exception):
    """Signal a requested cancellation which must unwind normal resource cleanup."""


@dataclass
class ProcessingHooks:
    """Thread-neutral callbacks; UI implementations marshal work to the main thread."""

    cancelled: Callable[[], bool] | None = None
    progress: Callable[[int, int, str], None] | None = None
    preview: Callable[[Image], None] | None = None
    select: Callable[[Image, Regions], Regions] | None = None

    def check(self) -> None:
        """Raise before further processing when the caller has requested cancellation."""
        if self.cancelled is not None and self.cancelled():
            raise ProcessingCancelled("Processing cancelled")

    def report(self, completed: int, total: int, stage: str) -> None:
        """Check cancellation and optionally report stage progress to the caller."""
        self.check()
        if self.progress is not None:
            self.progress(completed, total, stage)

    def show(self, frame: Image) -> None:
        """Check cancellation and optionally publish an annotated frame."""
        self.check()
        if self.preview is not None:
            self.preview(frame)
