
import os

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage

import time

def bgr_frame_to_qimage(frame: np.ndarray, *, mirror: bool = True) -> QImage:
    """Convert one OpenCV BGR frame into an independently owned RGB QImage."""

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (height, width, 3)")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if mirror:
        rgb = cv2.flip(rgb, 1)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    bytes_per_line = width * channels

    # copy() is essential: OpenCV will reuse its frame buffer on the next read.
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()


class _WebcamWorker(QThread):
    """Read frames away from the main Qt event loop."""

    frame_ready = pyqtSignal(QImage)
    analysis_frame_ready = pyqtSignal(object)
    camera_opened = pyqtSignal(int, int, int, str)
    capture_error = pyqtSignal(str)
    capture_stopped = pyqtSignal()

    def __init__(
        self,
        camera_index: int,
        target_fps: int,
        analysis_fps:int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self.target_fps = max(1, target_fps)
        self.analysis_fps = max(1, min(analysis_fps, self.target_fps))

    def _open_capture(self):
        attempts: list[tuple[int, str]] = []
        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            attempts.append((cv2.CAP_DSHOW, "DirectShow"))
        attempts.append((cv2.CAP_ANY, "automatic"))

        for api_preference, label in attempts:
            capture = cv2.VideoCapture(self.camera_index, api_preference)
            if capture.isOpened():
                return capture, label
            capture.release()
        return None, "unavailable"

    def run(self) -> None:
        capture = None
        try:
            capture, backend_name = self._open_capture()
            if capture is None:
                self.capture_error.emit(
                    f"Could not open webcam index {self.camera_index}. "
                    "Check Windows camera permission and whether another app is using it."
                )
                return

            capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_opened.emit(
                self.camera_index,
                actual_width,
                actual_height,
                backend_name,
            )

            failed_reads = 0
            frame_delay_ms = max(1, int(1000 / self.target_fps))
            analysis_interval = 1.0 / self.analysis_fps
            last_analysis_time = 0.0

            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 30:
                        self.capture_error.emit(
                            "The webcam opened but stopped returning frames."
                        )
                        break
                    self.msleep(20)
                    continue

                failed_reads = 0
                try:
                    self.frame_ready.emit(bgr_frame_to_qimage(frame, mirror=True))
                    
                except (ValueError, cv2.error) as error:
                    self.capture_error.emit(f"Webcam frame conversion failed: {error}")
                    break
                now = time.monotonic()

                if now - last_analysis_time >= analysis_interval:
                    # Own the array because VideoCapture reuses its buffer.
                    self.analysis_frame_ready.emit(frame.copy())
                    last_analysis_time = now
    
                self.msleep(frame_delay_ms)
        except Exception as error:  # Hardware drivers can raise backend-specific errors.
            self.capture_error.emit(f"Unexpected webcam error: {error}")
        finally:
            if capture is not None:
                capture.release()
            self.capture_stopped.emit()

class WebcamService(QObject):
    """Small Wrapper around the threaded capture worker."""

    frame_ready = pyqtSignal(QImage)
    analysis_frame_ready = pyqtSignal(object)
    camera_opened = pyqtSignal(int, int, int, str)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 24,
        analysis_fps: int = 4,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.analysis_fps = analysis_fps
        self._worker: _WebcamWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @pyqtSlot()
    def start(self) -> None:
        if self.is_running:
            return

        worker = _WebcamWorker(
            self.camera_index,
            self.target_fps,
            self.analysis_fps,
            self,
        )
        worker.frame_ready.connect(self.frame_ready)
        worker.analysis_frame_ready.connect(self.analysis_frame_ready)
        worker.camera_opened.connect(self.camera_opened)
        worker.capture_error.connect(self.error)
        worker.capture_stopped.connect(self.stopped)
        worker.finished.connect(self._worker_finished)
        self._worker = worker
        worker.start()

    @pyqtSlot()
    def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.isRunning():
            worker.requestInterruption()
            worker.wait(2500)

    @pyqtSlot()
    def _worker_finished(self) -> None:
        finished_worker = self.sender()
        if finished_worker is self._worker:
            self._worker = None
        if isinstance(finished_worker, _WebcamWorker):
            finished_worker.deleteLater()
