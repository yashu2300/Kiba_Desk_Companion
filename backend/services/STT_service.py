"""Always-listening Sherpa-ONNX wake word and Faster-Whisper STT service."""

from __future__ import annotations

import os
import re
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)


def _project_path(root: Path, environment_name: str, default: str) -> Path:
    configured = os.getenv(environment_name, "").strip()
    path = Path(configured).expanduser() if configured else Path(default)
    return path if path.is_absolute() else root / path


def _microphone_device(value: str) -> int | str | None:
    value = value.strip()

    if not value:
        return None

    return int(value) if value.isdigit() else value


class _WakeWordThread(QThread):
    initialized = pyqtSignal()
    initialization_failed = pyqtSignal(str)
    wake_word_detected = pyqtSignal(str)
    command_captured = pyqtSignal(object)
    capture_cancelled = pyqtSignal(str)

    def __init__(
        self,
        model_dir: Path,
        keywords_path: Path,
        microphone_device: int | str | None,
        sample_rate: int,
        chunk_ms: int,
        num_threads: int,
        keywords_score: float,
        keywords_threshold: float,
        silence_rms: float,
        speech_wait_s: float,
        end_silence_s: float,
        min_command_s: float,
        max_command_s: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.model_dir = model_dir
        self.keywords_path = keywords_path
        self.microphone_device = microphone_device
        self.sample_rate = sample_rate

        self.frames_per_chunk = max(
            160,
            int(sample_rate * chunk_ms / 1000),
        )

        self.chunk_seconds = (
            self.frames_per_chunk
            / float(sample_rate)
        )

        self.num_threads = num_threads
        self.keywords_score = keywords_score
        self.keywords_threshold = keywords_threshold
        self.silence_rms = silence_rms
        self.speech_wait_s = speech_wait_s
        self.end_silence_s = end_silence_s
        self.min_command_s = min_command_s
        self.max_command_s = max_command_s

        self._stop_event = threading.Event()
        self._enabled_event = threading.Event()
        self._stream: Any | None = None

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled_event.set()
        else:
            self._enabled_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self._enabled_event.set()
        self.requestInterruption()

        stream = self._stream

        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

    def _model_file(self, filename: str) -> str:
        path = self.model_dir / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        return str(path)

    def _record_command(
        self,
        microphone: Any,
        pre_roll: list[np.ndarray],
    ) -> np.ndarray | None:
        frames = [
            frame.copy()
            for frame in pre_roll
        ]

        elapsed = 0.0
        silence_after_speech = 0.0
        speech_started = False

        while (
            not self._stop_event.is_set()
            and not self.isInterruptionRequested()
        ):
            samples, _overflowed = microphone.read(
                self.frames_per_chunk
            )

            chunk = (
                np.asarray(
                    samples,
                    dtype=np.float32,
                )
                .reshape(-1)
                .copy()
            )

            frames.append(chunk)
            elapsed += self.chunk_seconds

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(chunk),
                        dtype=np.float64,
                    )
                )
            )

            if rms >= self.silence_rms:
                speech_started = True
                silence_after_speech = 0.0

            elif speech_started:
                silence_after_speech += (
                    self.chunk_seconds
                )

            if (
                not speech_started
                and elapsed >= self.speech_wait_s
            ):
                return None

            if (
                speech_started
                and elapsed >= self.min_command_s
                and silence_after_speech >= self.end_silence_s
            ):
                return np.concatenate(
                    frames
                ).astype(
                    np.float32,
                    copy=False,
                )

            if elapsed >= self.max_command_s:
                if speech_started:
                    return np.concatenate(
                        frames
                    ).astype(
                        np.float32,
                        copy=False,
                    )

                return None

        return None

    def run(self) -> None:
        try:
            import sherpa_onnx
            import sounddevice as sd

            if not self.keywords_path.is_file():
                raise FileNotFoundError(
                    self.keywords_path
                )

            spotter = sherpa_onnx.KeywordSpotter(
                tokens=self._model_file(
                    "tokens.txt"
                ),
                encoder=self._model_file(
                    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
                ),
                decoder=self._model_file(
                    "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
                ),
                joiner=self._model_file(
                    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
                ),
                num_threads=self.num_threads,
                max_active_paths=4,
                keywords_file=str(
                    self.keywords_path
                ),
                keywords_score=(
                    self.keywords_score
                ),
                keywords_threshold=(
                    self.keywords_threshold
                ),
                num_trailing_blanks=1,
                provider="cpu",
            )

            keyword_stream = (
                spotter.create_stream()
            )

            pre_roll_chunks = max(
                1,
                int(
                    400
                    / (
                        self.chunk_seconds
                        * 1000
                    )
                ),
            )

            recent_audio: deque[
                np.ndarray
            ] = deque(
                maxlen=pre_roll_chunks
            )

            self.initialized.emit()

            with sd.InputStream(
                device=self.microphone_device,
                channels=1,
                dtype="float32",
                samplerate=self.sample_rate,
                blocksize=self.frames_per_chunk,
            ) as microphone:
                self._stream = microphone
                was_enabled = False

                while (
                    not self._stop_event.is_set()
                    and not self.isInterruptionRequested()
                ):
                    samples, _overflowed = (
                        microphone.read(
                            self.frames_per_chunk
                        )
                    )

                    chunk = (
                        np.asarray(
                            samples,
                            dtype=np.float32,
                        )
                        .reshape(-1)
                        .copy()
                    )

                    if not self._enabled_event.is_set():
                        if was_enabled:
                            spotter.reset_stream(
                                keyword_stream
                            )

                            recent_audio.clear()

                        was_enabled = False
                        continue

                    if not was_enabled:
                        spotter.reset_stream(
                            keyword_stream
                        )

                        recent_audio.clear()
                        was_enabled = True

                    recent_audio.append(chunk)

                    keyword_stream.accept_waveform(
                        self.sample_rate,
                        chunk,
                    )

                    detected_keyword = ""

                    while spotter.is_ready(
                        keyword_stream
                    ):
                        spotter.decode_stream(
                            keyword_stream
                        )

                        result = spotter.get_result(
                            keyword_stream
                        )

                        if result:
                            detected_keyword = str(
                                getattr(
                                    result,
                                    "keyword",
                                    "",
                                )
                                or result
                            )
                            break

                    if not detected_keyword:
                        continue

                    self.wake_word_detected.emit(
                        detected_keyword
                    )

                    spotter.reset_stream(
                        keyword_stream
                    )

                    command = self._record_command(
                        microphone,
                        list(recent_audio),
                    )

                    recent_audio.clear()

                    if (
                        command is None
                        or command.size == 0
                    ):
                        self.capture_cancelled.emit(
                            "No command was heard after the wake phrase."
                        )
                        continue

                    self._enabled_event.clear()
                    self.command_captured.emit(
                        command
                    )

                    was_enabled = False

        except Exception as error:
            if not self._stop_event.is_set():
                self.initialization_failed.emit(
                    (
                        "Speech listener could not start: "
                        f"{type(error).__name__}: {error}"
                    )
                )

        finally:
            self._stream = None


class _WhisperRuntime:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        cpu_threads: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads

        self.model: Any | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return

            from faster_whisper import (
                WhisperModel,
            )

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=1,
            )

    def transcribe(
        self,
        samples: np.ndarray,
    ) -> str:
        self.load()

        segments, _info = (
            self.model.transcribe(
                samples,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
            )
        )

        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()


class _PreloadSignals(QObject):
    ready = pyqtSignal()
    failed = pyqtSignal(str)


class _PreloadWhisperTask(QRunnable):
    def __init__(
        self,
        runtime: _WhisperRuntime,
    ) -> None:
        super().__init__()

        self.runtime = runtime
        self.signals = _PreloadSignals()

    def run(self) -> None:
        try:
            self.runtime.load()
            self.signals.ready.emit()

        except Exception as error:
            self.signals.failed.emit(
                (
                    "Whisper could not load: "
                    f"{type(error).__name__}: {error}"
                )
            )


class _TranscriptionSignals(QObject):
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)


class _TranscriptionTask(QRunnable):
    def __init__(
        self,
        runtime: _WhisperRuntime,
        samples: np.ndarray,
    ) -> None:
        super().__init__()

        self.runtime = runtime
        self.samples = samples
        self.signals = (
            _TranscriptionSignals()
        )

    def run(self) -> None:
        try:
            transcript = (
                self.runtime.transcribe(
                    self.samples
                )
            )

            self.signals.completed.emit(
                transcript
            )

        except Exception as error:
            self.signals.failed.emit(
                (
                    "Speech transcription failed: "
                    f"{type(error).__name__}: {error}"
                )
            )


class SpeechToTextService(QObject):
    ready_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    capture_busy_changed = pyqtSignal(bool)
    wake_word_detected = pyqtSignal(str)
    transcript_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    _WAKE_PREFIX = re.compile(
        (
            r"^\s*"
            r"(?:(?:hey|hi|okay|ok)[\s,.-]+)?"
            r"(?:kibo|keebo|kebo)\b"
            r"[\s,:;.!?-]*"
        ),
        re.IGNORECASE,
    )

    def __init__(
        self,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")

        default_model_dir = (
            "models/"
            "sherpa-onnx-kws-zipformer-"
            "gigaspeech-3.3M-2024-01-01"
        )

        self.model_dir = _project_path(
            root,
            "SHERPA_KWS_MODEL_DIR",
            default_model_dir,
        )

        self.keywords_path = _project_path(
            root,
            "SHERPA_KWS_KEYWORDS_PATH",
            "config/keywords.txt",
        )

        self.microphone_device = (
            _microphone_device(
                os.getenv(
                    "STT_MIC_DEVICE",
                    "",
                )
            )
        )

        self.sample_rate = int(
            os.getenv(
                "STT_SAMPLE_RATE",
                "16000",
            )
        )

        self.chunk_ms = int(
            os.getenv(
                "STT_CHUNK_MS",
                "100",
            )
        )

        self.num_threads = int(
            os.getenv(
                "SHERPA_KWS_NUM_THREADS",
                "1",
            )
        )

        self.keywords_score = float(
            os.getenv(
                "SHERPA_KWS_SCORE",
                "1.5",
            )
        )

        self.keywords_threshold = float(
            os.getenv(
                "SHERPA_KWS_THRESHOLD",
                "0.35",
            )
        )

        self.silence_rms = float(
            os.getenv(
                "STT_SILENCE_RMS",
                "0.015",
            )
        )

        self.speech_wait_s = float(
            os.getenv(
                "STT_SPEECH_WAIT_SECONDS",
                "3.0",
            )
        )

        self.end_silence_s = float(
            os.getenv(
                "STT_END_SILENCE_SECONDS",
                "1.0",
            )
        )

        self.min_command_s = float(
            os.getenv(
                "STT_MIN_COMMAND_SECONDS",
                "0.4",
            )
        )

        self.max_command_s = float(
            os.getenv(
                "STT_MAX_COMMAND_SECONDS",
                "10.0",
            )
        )

        self._runtime = _WhisperRuntime(
            model_name=os.getenv(
                "STT_WHISPER_MODEL",
                "tiny.en",
            ).strip(),
            device=os.getenv(
                "STT_WHISPER_DEVICE",
                "cpu",
            ).strip(),
            compute_type=os.getenv(
                "STT_WHISPER_COMPUTE_TYPE",
                "int8",
            ).strip(),
            cpu_threads=int(
                os.getenv(
                    "STT_WHISPER_CPU_THREADS",
                    "2",
                )
            ),
        )

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self._listener: (
            _WakeWordThread | None
        ) = None

        self._preload_task: (
            _PreloadWhisperTask | None
        ) = None

        self._transcription_task: (
            _TranscriptionTask | None
        ) = None

        self._started = False
        self._stopping = False
        self._wake_ready = False
        self._whisper_ready = False
        self._interaction_busy = False
        self._awaiting_interaction = False
        self._guard_suspended = False

    @property
    def is_ready(self) -> bool:
        return (
            self._wake_ready
            and self._whisper_ready
        )

    @pyqtSlot()
    def start(self) -> None:
        if self._started or self._stopping:
            return

        self._started = True

        self.status_changed.emit(
            "Loading voice recognition…"
        )

        self._start_listener()

        self._preload_task = (
            _PreloadWhisperTask(
                self._runtime
            )
        )

        self._preload_task.signals.ready.connect(
            self._on_whisper_ready
        )

        self._preload_task.signals.failed.connect(
            self._on_whisper_failed
        )

        self._pool.start(
            self._preload_task
        )

    def _start_listener(self) -> None:
        if (
            self._stopping
            or self._guard_suspended
        ):
            return

        if (
            self._listener is not None
            and self._listener.isRunning()
        ):
            return

        self._listener = _WakeWordThread(
            model_dir=self.model_dir,
            keywords_path=self.keywords_path,
            microphone_device=self.microphone_device,
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            num_threads=self.num_threads,
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
            silence_rms=self.silence_rms,
            speech_wait_s=self.speech_wait_s,
            end_silence_s=self.end_silence_s,
            min_command_s=self.min_command_s,
            max_command_s=self.max_command_s,
            parent=self,
        )

        self._listener.initialized.connect(
            self._on_wake_ready
        )

        self._listener.initialization_failed.connect(
            self._on_listener_failed
        )

        self._listener.wake_word_detected.connect(
            self._on_wake_word_detected
        )

        self._listener.command_captured.connect(
            self._on_command_captured
        )

        self._listener.capture_cancelled.connect(
            self._on_capture_cancelled
        )

        self._listener.start()

    def _stop_listener(self) -> None:
        listener = self._listener
        self._listener = None
        self._wake_ready = False

        if (
            listener is not None
            and listener.isRunning()
        ):
            listener.stop()
            listener.wait()

        if listener is not None:
            listener.deleteLater()

    @pyqtSlot(bool)
    def set_guard_mode(
        self,
        active: bool,
    ) -> None:
        if self._guard_suspended == active:
            return

        self._guard_suspended = active
        self._set_awaiting_interaction(False)

        if active:
            self._stop_listener()
            self.ready_changed.emit(False)

            self.status_changed.emit(
                "Voice input disabled in guard mode"
            )

        elif self._started and not self._stopping:
            self.status_changed.emit(
                "Restarting wake-word listener…"
            )

            self._start_listener()

    @pyqtSlot(bool)
    def set_interaction_busy(
        self,
        busy: bool,
    ) -> None:
        was_busy = self._interaction_busy
        self._interaction_busy = busy

        if (
            was_busy
            and not busy
            and self._awaiting_interaction
        ):
            self._set_awaiting_interaction(
                False
            )

        self._apply_listening_state()

    def _set_awaiting_interaction(
        self,
        waiting: bool,
    ) -> None:
        if self._awaiting_interaction == waiting:
            return

        self._awaiting_interaction = waiting

        self.capture_busy_changed.emit(
            waiting
        )

    def _apply_listening_state(self) -> None:
        enabled = (
            self._started
            and not self._stopping
            and self.is_ready
            and not self._guard_suspended
            and not self._interaction_busy
            and not self._awaiting_interaction
        )

        if self._listener is not None:
            self._listener.set_enabled(
                enabled
            )

        if enabled:
            self.status_changed.emit(
                "Say ‘Hey Kibo’ to speak"
            )

        elif self._guard_suspended:
            self.status_changed.emit(
                "Voice input disabled in guard mode"
            )

        elif self._interaction_busy:
            self.status_changed.emit(
                "Voice input paused while Kibo responds"
            )

        elif (
            not self.is_ready
            and not self._stopping
        ):
            self.status_changed.emit(
                "Loading voice recognition…"
            )

    @pyqtSlot()
    def _on_wake_ready(self) -> None:
        self._wake_ready = True
        self._publish_ready()

    @pyqtSlot()
    def _on_whisper_ready(self) -> None:
        self._whisper_ready = True
        self._preload_task = None
        self._publish_ready()

    def _publish_ready(self) -> None:
        self.ready_changed.emit(
            self.is_ready
        )

        self._apply_listening_state()

    @pyqtSlot(str)
    def _on_listener_failed(
        self,
        message: str,
    ) -> None:
        self._wake_ready = False

        self.ready_changed.emit(False)

        self.status_changed.emit(
            "Voice input unavailable"
        )

        self.error.emit(message)

    @pyqtSlot(str)
    def _on_whisper_failed(
        self,
        message: str,
    ) -> None:
        self._whisper_ready = False
        self._preload_task = None

        self.ready_changed.emit(False)

        self.status_changed.emit(
            "Voice transcription unavailable"
        )

        self.error.emit(message)

    @pyqtSlot(str)
    def _on_wake_word_detected(
        self,
        keyword: str,
    ) -> None:
        self._set_awaiting_interaction(
            True
        )

        self._apply_listening_state()

        self.status_changed.emit(
            "Wake phrase heard — listening for your command…"
        )

        self.wake_word_detected.emit(
            keyword
        )

    @pyqtSlot(object)
    def _on_command_captured(
        self,
        samples: object,
    ) -> None:
        if self._stopping:
            return

        self._set_awaiting_interaction(
            True
        )

        self._apply_listening_state()

        self.status_changed.emit(
            "Transcribing…"
        )

        self._transcription_task = (
            _TranscriptionTask(
                self._runtime,
                np.asarray(
                    samples,
                    dtype=np.float32,
                ),
            )
        )

        self._transcription_task.signals.completed.connect(
            self._on_transcription_completed
        )

        self._transcription_task.signals.failed.connect(
            self._on_transcription_failed
        )

        self._pool.start(
            self._transcription_task
        )

    @pyqtSlot(str)
    def _on_capture_cancelled(
        self,
        message: str,
    ) -> None:
        self._set_awaiting_interaction(
            False
        )

        self.status_changed.emit(
            message
            + " Say ‘Hey Kibo’ to try again."
        )

        QTimer.singleShot(
            1500,
            self._apply_listening_state,
        )

    @pyqtSlot(str)
    def _on_transcription_completed(
        self,
        transcript: str,
    ) -> None:
        self._transcription_task = None

        clean_transcript = (
            self._WAKE_PREFIX.sub(
                "",
                transcript,
            )
            .strip()
        )

        if not clean_transcript:
            self._set_awaiting_interaction(
                False
            )

            self.status_changed.emit(
                "I did not catch the command. "
                "Say ‘Hey Kibo’ to try again."
            )

            QTimer.singleShot(
                1500,
                self._apply_listening_state,
            )
            return

        self.status_changed.emit(
            "Sending your message to Kibo…"
        )

        self.transcript_ready.emit(
            clean_transcript
        )

        QTimer.singleShot(
            250,
            self._release_if_no_interaction_started,
        )

    def _release_if_no_interaction_started(
        self,
    ) -> None:
        if (
            self._awaiting_interaction
            and not self._interaction_busy
        ):
            self._set_awaiting_interaction(
                False
            )

            self._apply_listening_state()

    @pyqtSlot(str)
    def _on_transcription_failed(
        self,
        message: str,
    ) -> None:
        self._transcription_task = None

        self._set_awaiting_interaction(
            False
        )

        self.error.emit(message)
        self._apply_listening_state()

    @pyqtSlot()
    def stop(self) -> None:
        if self._stopping:
            return

        self._stopping = True

        self._stop_listener()

        self._pool.clear()
        self._pool.waitForDone()

        self._listener = None
        self._runtime.model = None

        self._set_awaiting_interaction(
            False
        )

        self._wake_ready = False
        self._whisper_ready = False

        self.ready_changed.emit(False)