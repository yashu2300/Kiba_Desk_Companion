"""Embedded FastAPI server exposing the latest replaceable TTS WAV."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from backend.services.TTS_service import TextToSpeechService


def create_fastapi_app(tts: TextToSpeechService) -> FastAPI:
    app = FastAPI(title="Kiba Petoi Audio API", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        metadata = tts.latest_audio_metadata()
        return {
            "status": "ok",
            "audio_available": tts.has_audio,
            "revision": metadata.get("revision") if metadata else None,
        }

    @app.get("/tts/status")
    def tts_status() -> dict[str, Any]:
        metadata = tts.latest_audio_metadata()
        return {"audio_available": metadata is not None, "audio": metadata}

    @app.get("/tts.wav")
    def latest_tts_wav() -> Response:
        latest_audio = tts.latest_audio()
        if latest_audio is None:
            raise HTTPException(status_code=404, detail="No TTS audio is available yet.")

        wav_bytes, metadata = latest_audio
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Disposition": 'inline; filename="tts.wav"',
                "X-TTS-Revision": str(metadata.get("revision", "")),
                "X-TTS-Duration": str(metadata.get("duration_s", "")),
            },
        )

    return app


class _UvicornServerThread(QThread):
    server_failed = pyqtSignal(str)

    def __init__(self, app: FastAPI, host: str, port: int, log_level: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level
        self.server: uvicorn.Server | None = None

    def run(self) -> None:
        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level=self.log_level,
                access_log=False,
            )
            self.server = uvicorn.Server(config)
            self.server.run()
        except BaseException as error:
            self.server_failed.emit(f"FastAPI server failed: {type(error).__name__}: {error}")

    def request_stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True


class FastAPIServerService(QObject):
    server_started = pyqtSignal(str)
    server_stopped = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, tts: TextToSpeechService, parent: QObject | None = None) -> None:
        super().__init__(parent)
        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")

        self.host = os.getenv("FASTAPI_HOST", "0.0.0.0").strip()
        self.port = int(os.getenv("FASTAPI_PORT", "8000"))
        self.log_level = os.getenv("FASTAPI_LOG_LEVEL", "warning").strip()
        self.app = create_fastapi_app(tts)
        self._worker: _UvicornServerThread | None = None
        self._announced_started = False
        self._stopping = False
        self._startup_timer = QTimer(self)
        self._startup_timer.setInterval(100)
        self._startup_timer.timeout.connect(self._check_started)

    @property
    def local_url(self) -> str:
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}"

    @pyqtSlot()
    def start(self) -> None:
        if self._worker is not None or self._stopping:
            return

        self._worker = _UvicornServerThread(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            parent=self,
        )
        self._worker.server_failed.connect(self.error.emit)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self._startup_timer.start()

    @pyqtSlot()
    def _check_started(self) -> None:
        worker = self._worker
        if worker is None or worker.server is None:
            return
        if worker.server.started and not self._announced_started:
            self._announced_started = True
            self._startup_timer.stop()
            self.server_started.emit(self.local_url)

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        self._startup_timer.stop()
        self.server_stopped.emit()

    @pyqtSlot()
    def stop(self) -> None:
        if self._stopping:
            return

        self._stopping = True
        self._startup_timer.stop()
        worker = self._worker

        if worker is not None and worker.isRunning():
            worker.request_stop()
            if not worker.wait(10000):
                if worker.server is not None:
                    worker.server.force_exit = True
                worker.wait()

        self._worker = None
