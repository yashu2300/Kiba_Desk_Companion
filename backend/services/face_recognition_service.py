"""Threaded YuNet + SFace recognition, with local owner enrollment."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from backend.services.emotion_recognition_service import (
    EmotionRecognitionService,
)


class _FaceVisionWorker(QObject):
    """Own and run all OpenCV DNN objects on one background thread."""

    initialized = pyqtSignal(bool)
    analysis_completed = pyqtSignal(dict)
    enrollment_completed = pyqtSignal(bool, str)
    operation_failed = pyqtSignal(str)

    COSINE_THRESHOLD = 0.363
    MIN_ENROLLMENT_FACE_SIZE = 80
    MAX_FACES = 5

    def __init__(
        self,
        detector_model_path: Path,
        recognizer_model_path: Path,
        expression_model_path: Path,
        owner_embedding_path: Path,
    ) -> None:
        super().__init__()

        self.detector_model_path = detector_model_path
        self.recognizer_model_path = recognizer_model_path
        self.expression_model_path = expression_model_path
        self.owner_embedding_path = owner_embedding_path

        self._detector = None
        self._recognizer = None
        self._expression: EmotionRecognitionService | None = None
        self._owner_embedding: np.ndarray | None = None

    @pyqtSlot()
    def initialize(self) -> None:
        """Load all models inside the worker thread."""

        try:
            for path in (
                self.detector_model_path,
                self.recognizer_model_path,
                self.expression_model_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Required vision model was not found: {path}"
                    )

            if not hasattr(cv2, "FaceDetectorYN"):
                raise RuntimeError(
                    "This OpenCV build does not contain FaceDetectorYN."
                )

            if not hasattr(cv2, "FaceRecognizerSF"):
                raise RuntimeError(
                    "This OpenCV build does not contain FaceRecognizerSF."
                )

            self._detector = cv2.FaceDetectorYN.create(
                str(self.detector_model_path),
                "",
                (320, 320),
                0.90,   # Detection confidence threshold
                0.30,   # Non-maximum suppression threshold
                5000,
            )

            self._recognizer = cv2.FaceRecognizerSF.create(
                str(self.recognizer_model_path),
                "",
            )

            self._expression = EmotionRecognitionService(
                self.expression_model_path
            )

            self._owner_embedding = self._load_owner_embedding()

            self.initialized.emit(
                self._owner_embedding is not None
            )

        except Exception as error:
            self.operation_failed.emit(
                f"Vision model initialization failed: {error}"
            )

    def _load_owner_embedding(self) -> np.ndarray | None:
        if not self.owner_embedding_path.is_file():
            return None

        embedding = np.load(
            self.owner_embedding_path,
            allow_pickle=False,
        )

        return self._normalise_embedding(embedding)

    @staticmethod
    def _normalise_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        norm = float(np.linalg.norm(embedding))

        if norm <= 1e-12:
            raise ValueError(
                "The face embedding has zero length."
            )

        return embedding / norm

    def _detect_faces(
        self,
        frame: np.ndarray,
    ) -> list[np.ndarray]:
        if self._detector is None:
            raise RuntimeError(
                "The face detector has not been initialized."
            )

        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))

        _, detected = self._detector.detect(frame)

        if detected is None:
            return []

        faces = [
            np.asarray(face, dtype=np.float32)
            for face in detected
        ]

        # Largest face becomes the primary person.
        faces.sort(
            key=lambda face: float(face[2] * face[3]),
            reverse=True,
        )

        return faces[: self.MAX_FACES]

    def _extract_embedding_and_face(
        self,
        frame: np.ndarray,
        face: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._recognizer is None:
            raise RuntimeError(
                "The face recognizer has not been initialized."
            )

        aligned_face = self._recognizer.alignCrop(
            frame,
            face,
        )

        embedding = self._recognizer.feature(
            aligned_face
        )

        return (
            self._normalise_embedding(embedding),
            aligned_face,
        )

    def _identify(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, float | None]:
        if self._owner_embedding is None:
            return "Not enrolled", None

        similarity = float(
            np.dot(
                embedding.ravel(),
                self._owner_embedding.ravel(),
            )
        )

        identity = (
            "Owner"
            if similarity >= self.COSINE_THRESHOLD
            else "Unknown"
        )

        return identity, similarity

    @staticmethod
    def _normalised_mirrored_box(
        face: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> list[float]:
        """Return an overlay box matching the mirrored webcam display."""

        x, y, width, height = (
            float(value) for value in face[:4]
        )

        mirrored_x = frame_width - x - width

        return [
            max(0.0, min(1.0, mirrored_x / frame_width)),
            max(0.0, min(1.0, y / frame_height)),
            max(0.0, min(1.0, width / frame_width)),
            max(0.0, min(1.0, height / frame_height)),
        ]

    @pyqtSlot(object)
    def analyse(self, frame: object) -> None:
        try:
            image = np.asarray(frame)

            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    "Vision analysis expected one BGR colour frame."
                )

            if self._expression is None:
                raise RuntimeError(
                    "The expression model has not been initialized."
                )

            frame_height, frame_width = image.shape[:2]
            face_rows = self._detect_faces(image)
            detected_faces: list[dict] = []

            for face in face_rows:
                embedding, aligned_face = (
                    self._extract_embedding_and_face(
                        image,
                        face,
                    )
                )

                identity, similarity = self._identify(
                    embedding
                )

                expression, expression_confidence = (
                    self._expression.predict(aligned_face)
                )

                detected_faces.append(
                    {
                        "bbox_norm": self._normalised_mirrored_box(
                            face,
                            frame_width,
                            frame_height,
                        ),
                        "identity": identity,
                        "similarity": (
                            round(similarity, 3)
                            if similarity is not None
                            else None
                        ),
                        "expression": expression,
                        "expression_confidence": round(
                            expression_confidence,
                            3,
                        ),
                        "detection_confidence": round(
                            float(face[-1]),
                            3,
                        ),
                    }
                )

            primary = next(
                (
                    face
                    for face in detected_faces
                    if face["identity"] == "Owner"
                ),
                detected_faces[0]
                if detected_faces
                else None,
            )

            if not detected_faces:
                identity_summary = "No face"
                expression_summary = "No face"

            elif self._owner_embedding is None:
                identity_summary = "Not enrolled"
                expression_summary = primary["expression"]

            else:
                owner_count = sum(
                    face["identity"] == "Owner"
                    for face in detected_faces
                )

                unknown_count = (
                    len(detected_faces) - owner_count
                )

                if owner_count and unknown_count:
                    identity_summary = (
                        f"Owner + {unknown_count} unknown"
                    )
                elif owner_count:
                    identity_summary = "Owner"
                elif len(detected_faces) == 1:
                    identity_summary = "Unknown"
                else:
                    identity_summary = (
                        f"{unknown_count} unknown"
                    )

                expression_summary = primary["expression"]

            result = {
                "face_present": bool(detected_faces),
                "face_count": len(detected_faces),
                "presence": "At desk" if detected_faces else "Away",
                "identity": identity_summary,
                "expression": expression_summary,
                "identity_confidence": primary.get("similarity") if primary else None,
                "expression_confidence": primary.get("expression_confidence", 0.0) if primary else 0.0,
                "detection_confidence": primary.get("detection_confidence", 0.0) if primary else 0.0,
                "owner_enrolled": self._owner_embedding is not None,
                "faces": detected_faces,
            }

            self.analysis_completed.emit(result)

        except Exception as error:
            self.operation_failed.emit(
                f"Vision analysis failed: {error}"
            )

    @pyqtSlot(object)
    def enroll_owner(self, frame: object) -> None:
        """Create and store one local owner embedding."""

        try:
            image = np.asarray(frame)
            faces = self._detect_faces(image)

            if len(faces) != 1:
                raise ValueError(
                    "Enrollment needs exactly one visible face. "
                    f"Currently detected: {len(faces)}."
                )

            face = faces[0]

            if min(
                float(face[2]),
                float(face[3]),
            ) < self.MIN_ENROLLMENT_FACE_SIZE:
                raise ValueError(
                    "Move closer to the camera and try "
                    "enrollment again."
                )

            embedding, _ = (
                self._extract_embedding_and_face(
                    image,
                    face,
                )
            )

            self.owner_embedding_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = (
                self.owner_embedding_path.with_suffix(".tmp")
            )

            with temporary_path.open("wb") as file:
                np.save(
                    file,
                    embedding,
                    allow_pickle=False,
                )

            os.replace(
                temporary_path,
                self.owner_embedding_path,
            )

            self._owner_embedding = embedding

            self.enrollment_completed.emit(
                True,
                "Owner face enrolled locally. "
                "Recognition is now active.",
            )

        except Exception as error:
            self.enrollment_completed.emit(
                False,
                str(error),
            )


class FaceRecognitionService(QObject):
    """Qt facade preventing OpenCV inference from blocking the GUI."""

    initialized = pyqtSignal(bool)
    result_ready = pyqtSignal(dict)
    enrollment_busy_changed = pyqtSignal(bool)
    enrollment_finished = pyqtSignal(bool, str)
    error = pyqtSignal(str)

    _analysis_requested = pyqtSignal(object)
    _enrollment_requested = pyqtSignal(object)

    def __init__(
        self,
        detector_model_path: str | Path,
        recognizer_model_path: str | Path,
        expression_model_path: str | Path,
        owner_embedding_path: str | Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self._ready = False
        self._busy = False
        self._enrollment_pending = False
        self._latest_frame: np.ndarray | None = None

        self._thread = QThread(self)

        self._worker = _FaceVisionWorker(
            Path(detector_model_path),
            Path(recognizer_model_path),
            Path(expression_model_path),
            Path(owner_embedding_path),
        )

        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.initialize)
        self._analysis_requested.connect(self._worker.analyse)

        self._enrollment_requested.connect(
            self._worker.enroll_owner
        )

        self._worker.initialized.connect(
            self._on_initialized
        )

        self._worker.analysis_completed.connect(
            self._on_analysis_completed
        )

        self._worker.enrollment_completed.connect(
            self._on_enrollment_completed
        )

        self._worker.operation_failed.connect(
            self._on_operation_failed
        )

        self._thread.finished.connect(
            self._worker.deleteLater
        )

    @pyqtSlot()
    def start(self) -> None:
        if not self._thread.isRunning():
            self._thread.start()

    @pyqtSlot()
    def stop(self) -> None:
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)

    @pyqtSlot(object)
    def submit_frame(self, frame: object) -> None:
        """Receive an analysis frame from WebcamService."""

        image = np.asarray(frame)
        first_frame = self._latest_frame is None
        self._latest_frame = image.copy()

        if first_frame:
            print(
                "[VISION] First analysis frame received: "
                f"{image.shape}"
            )

        if not self._ready:
            return

        # The user pressed Enroll before the first frame arrived.
        # Enroll now using this newly received frame.
        if self._enrollment_pending and not self._busy:
            self._enrollment_pending = False
            self._dispatch_enrollment()
            return

        if self._busy:
            return

        self._busy = True

        self._analysis_requested.emit(
            self._latest_frame.copy()
        )

    @pyqtSlot()
    def enroll_owner(self) -> None:
        if not self._ready:
            self.error.emit(
                "Vision models are still loading."
            )
            return

        self.enrollment_busy_changed.emit(True)
        self._enrollment_pending = True

        # A frame has not arrived yet. submit_frame() will
        # automatically continue enrollment when one arrives.
        if self._latest_frame is None:
            print(
                "[VISION] Enrollment queued while waiting "
                "for the first analysis frame."
            )
            return

        # Let the current analysis finish before enrolling.
        if self._busy:
            return

        self._enrollment_pending = False
        self._dispatch_enrollment()

    def _dispatch_enrollment(self) -> None:
        if self._latest_frame is None:
            self.enrollment_busy_changed.emit(False)
            return

        self._busy = True

        self._enrollment_requested.emit(
            self._latest_frame.copy()
        )

    @pyqtSlot(bool)
    def _on_initialized(
        self,
        owner_enrolled: bool,
    ) -> None:
        self._ready = True
        self.initialized.emit(owner_enrolled)

    @pyqtSlot(dict)
    def _on_analysis_completed(
        self,
        result: dict,
    ) -> None:
        self._busy = False
        self.result_ready.emit(result)

        if self._enrollment_pending:
            self._enrollment_pending = False
            self._dispatch_enrollment()

    @pyqtSlot(bool, str)
    def _on_enrollment_completed(
        self,
        success: bool,
        message: str,
    ) -> None:
        self._busy = False
        self.enrollment_busy_changed.emit(False)
        self.enrollment_finished.emit(
            success,
            message,
        )

    @pyqtSlot(str)
    def _on_operation_failed(
        self,
        message: str,
    ) -> None:
        self._busy = False
        self.enrollment_busy_changed.emit(False)
        self.error.emit(message)

        if self._enrollment_pending:
            self._enrollment_pending = False
            self._dispatch_enrollment()
