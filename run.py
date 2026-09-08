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
from backend.services.activity_monitor_service import ActivityMonitorService
from backend.services.google_calendar_service import GoogleCalendarService
from backend.services.llm_service import LLMService

from backend.slots import SlotController
from backend.state_manager import CurrentStateManager
from backend.database import Database
from backend.demo_clock import DemoClock

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
    database = Database()
    database.create_tables()
    profile = database.load_or_create_profile()
    clock = DemoClock(speed=1.0)
    session_id = database.start_app_session(profile["user_id"], clock.speed)
    state_manager = CurrentStateManager(
        clock=clock,
        session_id=session_id,
        user_name=profile["display_name"],
        goal=profile["goal"],
        automatic_nudges=(
            profile["automatic_nudges"]
        ),
        # Five simulated minutes.
        inactivity_threshold_s=300.0,
    )
    
    # Created Services
    webcam = WebcamService(camera_index=0, target_fps=24, analysis_fps=4)
    face_recognition = FaceRecognitionService(
        detector_model_path=Path().cwd().joinpath("models", "face_detection_yunet_2023mar.onnx"), 
        recognizer_model_path= Path().cwd().joinpath("models", "face_recognition_sface_2021dec.onnx"),  
        expression_model_path= Path().cwd().joinpath("models", "facial_expression_recognition_mobilefacenet_2022july.onnx"),
        owner_embedding_path=Path().cwd().joinpath("data", "vision", "owner_embedding.npy") 
    )
    activity_monitor = ActivityMonitorService(throttle_seconds=0.25)
    calendar = GoogleCalendarService(clock=clock, timezone_name="Pacific/Auckland")
    llm = LLMService()

    window = DeskCompanionWindow()
    window.load_profile(profile["display_name"], profile["goal"], profile["automatic_nudges"])
    slot_controller = SlotController(
        webcam=webcam,
        face_recognition=face_recognition,
        activity_monitor=activity_monitor,
        calendar=calendar,
        llm=llm,
        clock=clock,
        state_manager=state_manager,
        database=database,
        user_id=profile["user_id"],
        session_id=session_id,
    )

    slot_controller.connect_window_to_slots(window)
    app.aboutToQuit.connect(slot_controller.shutdown)
    face_recognition.start()
    state_manager.start()
    activity_monitor.start()
    calendar.start()


    window.showMaximized()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())