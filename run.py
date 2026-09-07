"""
Entry Point for the Application
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from frontend.main_window import DeskCompanionWindow
from frontend.theme import APP_STYLESHEET

from backend.services.webcam_service import WebcamService
from backend.slots import SlotController

def create_application(argv: list[str] | None = None) -> QApplication:
    """Create and style the Qt application."""

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Kibo Desk Companion")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    return app


def main() -> int:
    app = create_application()

    # Created Services
    webcam = WebcamService(camera_index=0, target_fps=24)

    window = DeskCompanionWindow()
    slot_controller = SlotController(webcam)
    slot_controller.connect_window_to_slots(window)


    window.showMaximized()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())