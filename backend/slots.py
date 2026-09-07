"""Qt slot controller connecting frontend signals to backend services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import QObject, pyqtSlot


# Import all services such as demo clock and etc. 
from backend.services.webcam_service import WebcamService

from frontend.main_window import DeskCompanionWindow


class SlotController(QObject):
    def __init__(self, webcam:WebcamService, parent:QObject | None = None): # Add in Service Instance Here

        super().__init__(parent)
        self.webcam = webcam


    def connect_window_to_slots(self, window:DeskCompanionWindow):
        """ Wire every frontend/pyqtSignals and service result signals in ONE PLACE"""

        # Connect Main Window Signals
        window.camera_toggled.connect(self.on_camera_toggled)

        # Connect WebcamService Signals
        self.webcam.frame_ready.connect(window.set_camera_frame)
        self.webcam.camera_opened.connect(window.show_camera_opened)
        self.webcam.error.connect(window.show_camera_error)
        self.webcam.error.connect(self._on_webcam_error)
        self.webcam.stopped.connect(window.show_camera_stopped)
        self.webcam.stopped.connect(self._on_webcam_stopped)


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
