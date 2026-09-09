"""Kokoro-ONNX speech generation with one replaceable WAV buffer."""

from __future__ import annotations

import io
import os
import queue
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _project_path(
    root: Path,
    environment_name: str,
    default: str,
) -> Path:
    configured = os.getenv(
        environment_name,
        "",
    ).strip()

    path = (
        Path(configured).expanduser()
        if configured
        else Path(default)
    )

    return path if path.is_absolute() else root / path


class _KokoroWorkerThread(QThread):
    initialized = pyqtSignal()
    initialization_failed = pyqtSignal(str)
    audio_created = pyqtSignal(dict)
    request_finished = pyqtSignal(dict)
    request_failed = pyqtSignal(str, str)

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.model_path = model_path
        self.voices_path = voices_path

        self._requests: queue.Queue[
            dict[str, Any] | None
        ] = queue.Queue()

        self._stop_requested = (
            threading.Event()
        )

        self._sounddevice: Any | None = None

    def enqueue(
        self,
        request: dict[str, Any],
    ) -> None:
        self._requests.put(request)

    def stop(self) -> None:
        self._stop_requested.set()
        self.requestInterruption()

        # Remove queued speech so the application
        # does not continue speaking while closing.
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break

        self._requests.put(None)

        if self._sounddevice is not None:
            try:
                self._sounddevice.stop()
            except Exception:
                pass

    def run(self) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf

            from kokoro_onnx import Kokoro

            self._sounddevice = sd

            # Kokoro is loaded once and remains in
            # this background thread.
            kokoro = Kokoro(
                str(self.model_path),
                str(self.voices_path),
            )

            self.initialized.emit()

        except Exception as error:
            self.initialization_failed.emit(
                "Kokoro could not start: "
                f"{type(error).__name__}: {error}"
            )
            return

        while (
            not self._stop_requested.is_set()
            and not self.isInterruptionRequested()
        ):
            request = self._requests.get()

            if (
                request is None
                or self._stop_requested.is_set()
                or self.isInterruptionRequested()
            ):
                break

            turn_id = str(
                request["turn_id"]
            )

            try:
                samples, sample_rate = (
                    kokoro.create(
                        request["text"],
                        voice=request["voice"],
                        speed=request["speed"],
                        lang=request["language"],
                    )
                )

                # Create one complete WAV in memory.
                wav_buffer = io.BytesIO()

                sf.write(
                    wav_buffer,
                    samples,
                    samplerate=sample_rate,
                    subtype="PCM_16",
                    format="WAV",
                )

                # bytes are immutable, so FastAPI
                # can safely read the current value.
                wav_bytes = (
                    wav_buffer.getvalue()
                )

                metadata = {
                    "turn_id": turn_id,
                    "persona": request["persona"],
                    "text": request["text"],
                    "voice": request["voice"],
                    "sample_rate": int(
                        sample_rate
                    ),
                    "duration_s": (
                        len(samples)
                        / float(sample_rate)
                    ),
                    "size_bytes": len(
                        wav_bytes
                    ),
                    "played_locally": False,
                }

                # Send the fully generated WAV to
                # TextToSpeechService before playback.
                self.audio_created.emit(
                    {
                        **metadata,
                        "wav_bytes": wav_bytes,
                    }
                )

                if (
                    request["autoplay"]
                    and not self
                    ._stop_requested
                    .is_set()
                ):
                    sd.play(
                        samples,
                        sample_rate,
                    )

                    sd.wait()

                    metadata[
                        "played_locally"
                    ] = (
                        not self
                        ._stop_requested
                        .is_set()
                    )

                self.request_finished.emit(
                    metadata
                )

            except Exception as error:
                if not self._stop_requested.is_set():
                    self.request_failed.emit(
                        turn_id,
                        "Speech generation failed: "
                        f"{type(error).__name__}: "
                        f"{error}",
                    )

        try:
            sd.stop()
        except Exception:
            pass


class TextToSpeechService(QObject):
    ready_changed = pyqtSignal(bool)
    busy_changed = pyqtSignal(bool)

    # Emitted when the latest complete WAV has
    # replaced the previous WAV.
    audio_ready = pyqtSignal(dict)

    playback_finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        load_dotenv(root / ".env")

        self.model_path = _project_path(
            root,
            "KOKORO_MODEL_PATH",
            "models/kokoro-v1.0.onnx",
        )

        self.voices_path = _project_path(
            root,
            "KOKORO_VOICES_PATH",
            "models/voices-v1.0.bin",
        )

        self.voice = os.getenv(
            "KOKORO_VOICE",
            "af_heart",
        ).strip()

        self.language = os.getenv(
            "KOKORO_LANGUAGE",
            "en-us",
        ).strip()

        self.speed = float(
            os.getenv(
                "KOKORO_SPEED",
                "1.0",
            )
        )

        self.autoplay = _environment_bool(
            "TTS_AUTOPLAY",
            True,
        )

        self._worker: (
            _KokoroWorkerThread | None
        ) = None

        self._started = False
        self._ready = False
        self._failed = False
        self._stopping = False
        self._pending_requests = 0
        self._revision = 0

        # Only one WAV is retained.
        self._latest_wav: (
            bytes | None
        ) = None

        self._latest_metadata: (
            dict[str, Any] | None
        ) = None

        self._audio_lock = (
            threading.Lock()
        )

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def has_audio(self) -> bool:
        with self._audio_lock:
            return (
                self._latest_wav
                is not None
            )

    def latest_wav_bytes(
        self,
    ) -> bytes | None:
        """
        Return the current WAV.

        The future FastAPI /tts.wav endpoint
        will call this method.
        """

        with self._audio_lock:
            return self._latest_wav

    def latest_audio_metadata(
        self,
    ) -> dict[str, Any] | None:
        with self._audio_lock:
            if self._latest_metadata is None:
                return None

            return dict(
                self._latest_metadata
            )

    @pyqtSlot()
    def start(self) -> None:
        if (
            self._started
            or self._stopping
        ):
            return

        missing = [
            str(path)
            for path in (
                self.model_path,
                self.voices_path,
            )
            if not path.is_file()
        ]

        if missing:
            self._failed = True

            self.error.emit(
                "Kokoro model file(s) "
                "missing: "
                + ", ".join(missing)
            )

            return

        self._worker = (
            _KokoroWorkerThread(
                model_path=self.model_path,
                voices_path=(
                    self.voices_path
                ),
                parent=self,
            )
        )

        self._worker.initialized.connect(
            self._on_initialized
        )

        self._worker.initialization_failed.connect(
            self._on_initialization_failed
        )

        self._worker.audio_created.connect(
            self._on_audio_created
        )

        self._worker.request_finished.connect(
            self._on_request_finished
        )

        self._worker.request_failed.connect(
            self._on_request_failed
        )

        self._started = True
        self._worker.start()

    def speak(
        self,
        text: str,
        turn_id: str = "",
        persona: str = "unknown",
    ) -> None:
        clean_text = text.strip()

        if (
            not clean_text
            or self._stopping
        ):
            return

        if (
            not self._started
            and not self._failed
        ):
            self.start()

        if (
            self._failed
            or self._worker is None
        ):
            return

        request = {
            "text": clean_text,
            "turn_id": (
                turn_id
                or str(uuid4())
            ),
            "persona": persona,
            "voice": self.voice,
            "language": self.language,
            "speed": self.speed,
            "autoplay": self.autoplay,
        }

        self._pending_requests += 1

        if self._pending_requests == 1:
            self.busy_changed.emit(True)

        self._worker.enqueue(request)

    @pyqtSlot()
    def _on_initialized(self) -> None:
        self._ready = True
        self.ready_changed.emit(True)

    @pyqtSlot(str)
    def _on_initialization_failed(
        self,
        message: str,
    ) -> None:
        self._failed = True
        self._ready = False
        self._pending_requests = 0

        self.ready_changed.emit(False)
        self.busy_changed.emit(False)
        self.error.emit(message)

    @pyqtSlot(dict)
    def _on_audio_created(
        self,
        result: dict,
    ) -> None:
        wav_bytes = result.pop(
            "wav_bytes"
        )

        self._revision += 1

        metadata = {
            **result,
            "revision": self._revision,
        }

        # Replace the old WAV only after the new
        # WAV has been completely generated.
        with self._audio_lock:
            self._latest_wav = (
                wav_bytes
            )

            self._latest_metadata = (
                metadata
            )

        # Later this is where the Petoi service
        # can be told that /tts.wav is ready.
        self.audio_ready.emit(
            dict(metadata)
        )

    @pyqtSlot(dict)
    def _on_request_finished(
        self,
        result: dict,
    ) -> None:
        self._pending_requests = max(
            0,
            self._pending_requests - 1,
        )

        self.playback_finished.emit(
            result
        )

        if self._pending_requests == 0:
            self.busy_changed.emit(
                False
            )

    @pyqtSlot(str, str)
    def _on_request_failed(
        self,
        turn_id: str,
        message: str,
    ) -> None:
        self._pending_requests = max(
            0,
            self._pending_requests - 1,
        )

        self.error.emit(
            f"turn_id={turn_id}: {message}"
        )

        if self._pending_requests == 0:
            self.busy_changed.emit(
                False
            )

    @pyqtSlot()
    def stop(self) -> None:
        if self._stopping:
            return

        self._stopping = True
        worker = self._worker

        if (
            worker is not None
            and worker.isRunning()
        ):
            worker.stop()

            # Prevent:
            # QThread destroyed while running.
            worker.wait()

        self._ready = False
        self._pending_requests = 0

        self.ready_changed.emit(False)
        self.busy_changed.emit(False)