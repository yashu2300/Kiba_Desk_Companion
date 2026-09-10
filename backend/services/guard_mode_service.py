"""Deterministic guard mode driven only by local face recognition."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from PyQt5.QtCore import (
    QObject,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)


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

    return (
        path
        if path.is_absolute()
        else root / path
    )


class GuardModeService(QObject):
    mode_changed = pyqtSignal(str)
    warning_requested = pyqtSignal(str)
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

        self.warning_wav_path = (
            _project_path(
                root,
                "GUARD_WARNING_WAV_PATH",
                (
                    "assets/audio/"
                    "guard_warning.wav"
                ),
            )
        )

        self.repeat_interval_ms = max(
            1000,
            int(
                os.getenv(
                    "GUARD_WARNING_REPEAT_MS",
                    "5000",
                )
            ),
        )

        self.intruder_confirm_frames = max(
            1,
            int(
                os.getenv(
                    "GUARD_INTRUDER_CONFIRM_FRAMES",
                    "2",
                )
            ),
        )

        self.owner_confirm_frames = max(
            1,
            int(
                os.getenv(
                    "GUARD_OWNER_CONFIRM_FRAMES",
                    "2",
                )
            ),
        )

        self._active = False
        self._owner_enrolled = False
        self._intruder_present = False
        self._intruder_frames = 0
        self._owner_frames = 0

        self._warning_timer = QTimer(self)
        self._warning_timer.setInterval(
            self.repeat_interval_ms
        )
        self._warning_timer.timeout.connect(
            self._repeat_warning
        )

    @property
    def is_active(self) -> bool:
        return self._active

    @pyqtSlot(bool)
    def set_owner_enrolled(
        self,
        enrolled: bool,
    ) -> None:
        self._owner_enrolled = enrolled

    @pyqtSlot(bool, str)
    def on_enrollment_finished(
        self,
        success: bool,
        _message: str,
    ) -> None:
        if success:
            self._owner_enrolled = True

    def enter(self) -> bool:
        if self._active:
            return True

        if not self._owner_enrolled:
            self.error.emit(
                "Guard mode requires an "
                "enrolled owner face."
            )
            return False

        if not self.warning_wav_path.is_file():
            self.error.emit(
                "Guard warning WAV was not "
                f"found: {self.warning_wav_path}"
            )
            return False

        self._active = True
        self._intruder_present = False
        self._intruder_frames = 0
        self._owner_frames = 0
        self._warning_timer.stop()

        self.mode_changed.emit("guard")
        return True

    @pyqtSlot(dict)
    def observe_vision(
        self,
        result: dict,
    ) -> None:
        self._owner_enrolled = bool(
            result.get(
                "owner_enrolled",
                self._owner_enrolled,
            )
        )

        if not self._active:
            return

        if not self._owner_enrolled:
            self.error.emit(
                "Owner enrollment became "
                "unavailable; guard mode stopped."
            )
            self._return_to_normal()
            return

        faces = list(
            result.get("faces", [])
        )

        owner_detected = any(
            face.get("identity") == "Owner"
            for face in faces
        )

        intruder_detected = any(
            face.get("identity") == "Unknown"
            for face in faces
        )

        if owner_detected:
            self._owner_frames += 1
            self._intruder_frames = 0

            if (
                self._owner_frames
                >= self.owner_confirm_frames
            ):
                self._return_to_normal()

            return

        self._owner_frames = 0

        if intruder_detected:
            self._intruder_frames += 1

            if (
                self._intruder_frames
                >= self.intruder_confirm_frames
                and not self._intruder_present
            ):
                self._intruder_present = True

                self.warning_requested.emit(
                    str(self.warning_wav_path)
                )

                self._warning_timer.start()

        else:
            self._intruder_frames = 0
            self._intruder_present = False
            self._warning_timer.stop()

    @pyqtSlot()
    def stop(self) -> None:
        self._warning_timer.stop()
        self._active = False
        self._intruder_present = False

    def _repeat_warning(self) -> None:
        if (
            self._active
            and self._intruder_present
        ):
            self.warning_requested.emit(
                str(self.warning_wav_path)
            )

    def _return_to_normal(self) -> None:
        if not self._active:
            return

        self._warning_timer.stop()
        self._active = False
        self._intruder_present = False
        self._intruder_frames = 0
        self._owner_frames = 0

        self.mode_changed.emit("normal")