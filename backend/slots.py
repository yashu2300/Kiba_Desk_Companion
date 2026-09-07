"""Qt slot controller connecting frontend signals to backend services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, pyqtSlot


# Import all services such as demo clock and etc. 
from backend.services.webcam_service import WebcamService
from backend.services.face_recognition_service import FaceRecognitionService
from backend.services.activity_monitor_service import ActivityMonitorService

from backend.state_manager import CurrentStateManager
from backend.database import Database
from backend.demo_clock import DemoClock

from frontend.main_window import DeskCompanionWindow


class SlotController(QObject):
    def __init__(
        self,
        webcam: WebcamService,
        face_recognition: FaceRecognitionService,
        activity_monitor: ActivityMonitorService,
        clock: DemoClock,
        state_manager: CurrentStateManager,
        database: Database,
        user_id: int,
        session_id: int,
        parent: QObject | None = None) -> None:

        super().__init__(parent)
        self.webcam = webcam
        self.face_recognition = face_recognition
        self.activity_monitor = activity_monitor
        self.clock = clock
        self.state_manager = state_manager
        self.database = database
        self.user_id = user_id
        self.session_id = session_id


    def connect_window_to_slots(self, window:DeskCompanionWindow):
        """ Wire every frontend/pyqtSignals and service result signals in ONE PLACE"""

        # Connect Main Window Signals
        window.camera_toggled.connect(self.on_camera_toggled)
        window.face_enrollment_requested.connect(self.on_face_enrollment_requested)
        window.demo_clock_changed.connect(self.on_demo_clock_changed)
        window.goal_saved.connect(self.on_goal_saved)
        window.user_activity_detected.connect(self.state_manager.record_user_activity)


        # Connect WebcamService Signals
        self.webcam.frame_ready.connect(window.set_camera_frame)
        self.webcam.camera_opened.connect(window.show_camera_opened)
        self.webcam.error.connect(window.show_camera_error)
        self.webcam.error.connect(self._on_webcam_error)
        self.webcam.stopped.connect(window.show_camera_stopped)
        self.webcam.stopped.connect(self._on_webcam_stopped)

        # Connect Face Recongition Signals
        self.face_recognition.result_ready.connect(window.show_vision_result)
        self.face_recognition.enrollment_busy_changed.connect(window.set_face_enrollment_busy)
        self.face_recognition.enrollment_finished.connect(window.show_face_enrollment_result)
        self.face_recognition.error.connect(window.show_vision_error)
        self.face_recognition.error.connect(self._on_vision_error)
        self.face_recognition.initialized.connect(self._on_vision_initialized)

        # Activity Monitors
        self.activity_monitor.started.connect(self._on_activity_monitor_started)
        self.activity_monitor.stopped.connect(self._on_activity_monitor_stopped)
        self.activity_monitor.error.connect(self._on_activity_monitor_error)

        # State Related
        self.state_manager.state_changed.connect(window.show_current_state)
        self.state_manager.snapshot_created.connect(self._persist_state_snapshot)


        # Interconnected Serivce Signals
        self.webcam.analysis_frame_ready.connect(self.face_recognition.submit_frame)
        self.face_recognition.result_ready.connect(self.state_manager.update_vision_result)
        self.webcam.stopped.connect(self.state_manager.mark_vision_unavailable)
        self.activity_monitor.activity_detected.connect(self.state_manager.record_user_activity)

    # Webcam Services
    @pyqtSlot(bool)
    def on_camera_toggled(self, camera_active: bool) -> None:
        print(f"[SLOT] on_camera_toggled(camera_active={camera_active})")
        if camera_active:
            self.webcam.start()
        else:
            self.webcam.stop()

    @pyqtSlot(str)
    def _on_webcam_error(self, message: str) -> None:
        print(f"[WEBCAM ERROR] {message}")

    @pyqtSlot()
    def _on_webcam_stopped(self) -> None:
        print("[WEBCAM] Capture stopped and the camera was released.")

    # Vision Face Recognition Service
    @pyqtSlot()
    def on_face_enrollment_requested(self) -> None:
        print("[SLOT] on_face_enrollment_requested()")
        self.face_recognition.enroll_owner()

    @pyqtSlot(bool)
    def _on_vision_initialized(
        self,
        owner_enrolled: bool,
    ) -> None:
        enrolled = "yes" if owner_enrolled else "no"

        print(
            "[VISION] Models ready. "
            f"Owner enrolled: {enrolled}."
        )

    @pyqtSlot(str)
    def _on_vision_error(
        self,
        message: str,
    ) -> None:
        print(f"[VISION ERROR] {message}")

    # Activity Monitor Services
    @pyqtSlot()
    def _on_activity_monitor_started(self) -> None:
        print("[ACTIVITY] Global mouse and keyboard monitoring started.")

    @pyqtSlot()
    def _on_activity_monitor_stopped(self) -> None:
        print("[ACTIVITY] Global input monitoring stopped.")


    @pyqtSlot(str)
    def _on_activity_monitor_error(self, message: str) -> None:
        print(f"[ACTIVITY ERROR] {message}")

    # DEMO & STATE RELATED
    @pyqtSlot(str, float)
    def on_demo_clock_changed(self, label: str, speed: float) -> None:
        print(f"[CLOCK] {label} ({speed:g}x)")
        self.state_manager.set_clock_speed(label, speed)

    @pyqtSlot(str, str, bool)
    def on_goal_saved(self, display_name: str, goal: str, automatic_nudges: bool) -> None:
        self.database.save_profile(self.user_id, display_name, goal, automatic_nudges)

        self.state_manager.update_profile(display_name, goal, automatic_nudges)

        print("[DATABASE] User profile and goal saved.")


    @pyqtSlot(dict)
    def _persist_state_snapshot(self, snapshot: dict,) -> None:
        try:
            snapshot_id = (self.database.save_state_snapshot(snapshot))

            print(
                f"[STATE] Snapshot {snapshot_id} saved: "
                f"{snapshot['trigger']} -> "
                f"{snapshot['changed_fields']}"
            )

        except Exception as error:
            print(
                "[DATABASE ERROR] State snapshot "
                f"was not saved: {error}"
            )





    @pyqtSlot()
    def shutdown(self) -> None:
        self.webcam.stop()
        self.face_recognition.stop()
        self.activity_monitor.stop()

        self.state_manager.capture_snapshot("session_ended")
        self.state_manager.stop()

        self.database.finish_app_session(self.session_id)
        self.database.close()