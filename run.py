"""
Entry Point for the Application
"""
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from frontend.main_window import DeskCompanionWindow
from frontend.theme import APP_STYLESHEET

from backend.services.webcam_service import WebcamService
from backend.services.face_recognition_service import FaceRecognitionService

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
    project_root = Path(__file__).resolve().parent

    # Created Services
    webcam = WebcamService(camera_index=0, target_fps=24, analysis_fps=4)
    face_recognition = FaceRecognitionService(
        detector_model_path=Path().cwd().joinpath("models", "face_detection_yunet_2023mar.onnx"), 
        recognizer_model_path= Path().cwd().joinpath("models", "face_recognition_sface_2021dec.onnx"),  
        expression_model_path= Path().cwd().joinpath("models", "facial_expression_recognition_mobilefacenet_2022july.onnx"),
        owner_embedding_path=Path().cwd().joinpath("data", "vision", "owner_embedding.npy") 
    )

    window = DeskCompanionWindow()
    slot_controller = SlotController(webcam, face_recognition)
    slot_controller.connect_window_to_slots(window)
    app.aboutToQuit.connect(slot_controller.shutdown)
    face_recognition.start()


    window.showMaximized()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())