"""Qt slot controller connecting frontend signals to backend services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, pyqtSlot


# Import all services such as demo clock and etc. 
from backend.services.webcam_service import WebcamService
from backend.services.face_recognition_service import FaceRecognitionService

from frontend.main_window import DeskCompanionWindow


class SlotController(QObject):
    def __init__(self, webcam:WebcamService, face_recognition:FaceRecognitionService, parent:QObject | None = None): # Add in Service Instance Here

        super().__init__(parent)
        self.webcam = webcam
        self.face_recognition = face_recognition


    def connect_window_to_slots(self, window:DeskCompanionWindow):
        """ Wire every frontend/pyqtSignals and service result signals in ONE PLACE"""

        # Connect Main Window Signals
        window.camera_toggled.connect(self.on_camera_toggled)
        window.face_enrollment_requested.connect(self.on_face_enrollment_requested)

        # Connect WebcamService Signals
        self.webcam.frame_ready.connect(window.set_camera_frame)
        self.webcam.camera_opened.connect(window.show_camera_opened)
        self.webcam.error.connect(window.show_camera_error)
        self.webcam.error.connect(self._on_webcam_error)
        self.webcam.stopped.connect(window.show_camera_stopped)
        self.webcam.stopped.connect(self._on_webcam_stopped)
        self.webcam.analysis_frame_ready.connect(self.face_recognition.submit_frame)

        # Connect Face Recongition Signals
        self.face_recognition.result_ready.connect(window.show_vision_result)
        self.face_recognition.enrollment_busy_changed.connect(window.set_face_enrollment_busy)
        self.face_recognition.enrollment_finished.connect(window.show_face_enrollment_result)
        self.face_recognition.error.connect(window.show_vision_error)
        self.face_recognition.error.connect(self._on_vision_error)
        self.face_recognition.initialized.connect(self._on_vision_initialized)


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



    @pyqtSlot()
    def shutdown(self) -> None:
        """Release the webcam and stop the vision thread."""

        self.webcam.stop()
        self.face_recognition.stop()