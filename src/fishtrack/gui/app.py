"""FishTrack desktop entry point and application-wide identity/logging setup."""

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from fishtrack.gui.main_window import MainWindow
from fishtrack.gui.settings import standard_log_directory


def configure_logging() -> RotatingFileHandler:
    """Install a bounded persistent log in Qt's application-local data location.

    :return: Installed handler, removed and closed during desktop shutdown.
    :raises OSError: If the standard logging directory cannot be created or written.
    """
    directory = standard_log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        directory / "fishtrack.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    return handler


def configure_style(application: QApplication) -> None:
    """Match ReefFrame's charcoal form and recessed control backgrounds.

    :param application: Desktop application receiving the shared Fusion palette.
    """
    application.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    application.setStyle("Fusion")
    palette = QPalette(application.palette())
    colors = {
        QPalette.ColorRole.Window: "#1e1e1e",
        QPalette.ColorRole.WindowText: "#f0f0f0",
        QPalette.ColorRole.Base: "#171717",
        QPalette.ColorRole.AlternateBase: "#1b1b1b",
        QPalette.ColorRole.Text: "#f0f0f0",
        QPalette.ColorRole.Button: "#242424",
        QPalette.ColorRole.ButtonText: "#dedede",
        QPalette.ColorRole.Light: "#414141",
        QPalette.ColorRole.Midlight: "#333333",
        QPalette.ColorRole.Mid: "#3a3a3a",
        QPalette.ColorRole.Dark: "#121212",
        QPalette.ColorRole.Shadow: "#0f0f0f",
        QPalette.ColorRole.Highlight: "#126d78",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#25c6bb",
        QPalette.ColorRole.ToolTipBase: "#242424",
        QPalette.ColorRole.ToolTipText: "#f0f0f0",
        QPalette.ColorRole.PlaceholderText: "#8a8a8a",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#666666"))
    application.setPalette(palette)


def main(argv: list[str] | None = None) -> int:
    """Open the desktop application with optional explicit starting folder.

    :param argv: Optional argument list, otherwise use process arguments.
    :return: Qt event-loop exit status.
    """
    parser = argparse.ArgumentParser(description="FishTrack desktop interface")
    parser.add_argument(
        "--folder", type=Path, help="Open a footage folder instead of the remembered folder"
    )
    args = parser.parse_args(argv)
    application = QApplication(sys.argv[:1])
    application.setApplicationName("FishTrack")
    application.setOrganizationName("FishTrack")
    configure_style(application)
    handler = configure_logging()
    window = MainWindow(folder=args.folder)
    application.setWindowIcon(window.windowIcon())
    application.aboutToQuit.connect(window.shutdown)
    window.show()
    try:
        return application.exec()
    finally:
        window.shutdown()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        logging.getLogger().removeHandler(handler)
        handler.close()
