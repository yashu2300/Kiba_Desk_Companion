from pathlib import Path

import cv2
import numpy as np


class EmotionRecognitionService:
    """Classify a 112x112 aligned face into one of seven expressions.

    This is a facial-expression estimate, not a measurement of internal emotion.
    The class contains no Qt code, allowing it to be reused by FastAPI later.
    """

    LABELS = (
        "Angry",
        "Disgust",
        "Fearful",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised",
    )

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Expression model was not found: {self.model_path}"
            )

        self._network = cv2.dnn.readNet(str(self.model_path))
        self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, aligned_face_bgr: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence) for one aligned BGR face."""

        if aligned_face_bgr is None or aligned_face_bgr.size == 0:
            raise ValueError(
                "Expression recognition received an empty face image."
            )

        face = cv2.resize(aligned_face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(face)
        self._network.setInput(blob)

        scores = np.asarray(
            self._network.forward(),
            dtype=np.float32,
        ).reshape(-1)

        if scores.size != len(self.LABELS):
            raise RuntimeError(
                "Unexpected expression-model output shape: "
                f"expected {len(self.LABELS)} values, received {scores.size}."
            )

        # Convert logits into displayable probabilities.
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)

        index = int(np.argmax(probabilities))
        return self.LABELS[index], float(probabilities[index])