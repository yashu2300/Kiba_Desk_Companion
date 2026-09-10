"""Background owner notifications for guard-mode intruder alerts."""

from __future__ import annotations

import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)


def _environment_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _EmailTaskSignals(QObject):
    sent = pyqtSignal(str)
    failed = pyqtSignal(str)


class _EmailTask(QRunnable):
    def __init__(
        self,
        host: str,
        port: int,
        use_ssl: bool,
        username: str,
        password: str,
        sender: str,
        recipient: str,
        subject: str,
        body: str,
        image_bytes: bytes | None,
        timeout_seconds: float,
    ) -> None:
        super().__init__()

        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.username = username
        self.password = password
        self.sender = sender
        self.recipient = recipient
        self.subject = subject
        self.body = body
        self.image_bytes = image_bytes
        self.timeout_seconds = timeout_seconds

        self.signals = _EmailTaskSignals()

    def _build_message(
        self,
    ) -> EmailMessage:
        message = EmailMessage()

        message["From"] = self.sender
        message["To"] = self.recipient
        message["Subject"] = self.subject

        message.set_content(
            self.body
        )

        if self.image_bytes:
            message.add_attachment(
                self.image_bytes,
                maintype="image",
                subtype="jpeg",
                filename=(
                    "kibo_guard_intruder.jpg"
                ),
            )

        return message

    def _send_once(self) -> None:
        message = self._build_message()
        tls_context = ssl.create_default_context()

        if self.use_ssl:
            with smtplib.SMTP_SSL(
                self.host,
                self.port,
                timeout=self.timeout_seconds,
                context=tls_context,
            ) as smtp:
                smtp.login(
                    self.username,
                    self.password,
                )

                smtp.send_message(
                    message
                )

            return

        with smtplib.SMTP(
            self.host,
            self.port,
            timeout=self.timeout_seconds,
        ) as smtp:
            smtp.ehlo()

            smtp.starttls(
                context=tls_context
            )

            smtp.ehlo()

            smtp.login(
                self.username,
                self.password,
            )

            smtp.send_message(
                message
            )

    def run(self) -> None:
        last_error: Exception | None = None

        # Retry once for temporary network issues.
        for attempt in range(2):
            try:
                self._send_once()

                self.signals.sent.emit(
                    self.recipient
                )
                return

            except Exception as error:
                last_error = error

                if attempt == 0:
                    time.sleep(2.0)

        if last_error is None:
            return

        self.signals.failed.emit(
            "Guard email could not be sent: "
            f"{type(last_error).__name__}: "
            f"{last_error}"
        )


class OwnerNotificationService(QObject):
    busy_changed = pyqtSignal(bool)
    notification_sent = pyqtSignal(str)
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

        self.enabled = _environment_bool(
            "GUARD_EMAIL_ENABLED",
            False,
        )

        self.host = os.getenv(
            "GUARD_EMAIL_SMTP_HOST",
            "smtp.gmail.com",
        ).strip()

        self.port = int(
            os.getenv(
                "GUARD_EMAIL_SMTP_PORT",
                "465",
            )
        )

        self.use_ssl = _environment_bool(
            "GUARD_EMAIL_USE_SSL",
            True,
        )

        self.username = os.getenv(
            "GUARD_EMAIL_USERNAME",
            "",
        ).strip()

        self.password = (
            os.getenv(
                "GUARD_EMAIL_APP_PASSWORD",
                "",
            )
            .replace(" ", "")
            .strip()
        )

        self.sender = os.getenv(
            "GUARD_EMAIL_FROM",
            self.username,
        ).strip()

        self.recipient = os.getenv(
            "GUARD_EMAIL_TO",
            "",
        ).strip()

        self.cooldown_seconds = max(
            0.0,
            float(
                os.getenv(
                    "GUARD_EMAIL_COOLDOWN_SECONDS",
                    "300",
                )
            ),
        )

        self.timeout_seconds = max(
            3.0,
            float(
                os.getenv(
                    "GUARD_EMAIL_TIMEOUT_SECONDS",
                    "10",
                )
            ),
        )

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self._task: _EmailTask | None = None
        self._last_sent_at: float | None = None
        self._stopping = False

    @property
    def is_configured(self) -> bool:
        return bool(
            self.enabled
            and self.host
            and self.port
            and self.username
            and self.password
            and self.sender
            and self.recipient
        )

    @property
    def is_busy(self) -> bool:
        return self._task is not None

    def notify_intruder(
        self,
        details: dict,
    ) -> bool:
        if (
            self._stopping
            or not self.enabled
        ):
            return False

        if not self.is_configured:
            self.error.emit(
                "Guard email is enabled but "
                "its sender, App Password, "
                "or recipient is missing "
                "from .env."
            )
            return False

        if self.is_busy:
            return False

        now = time.monotonic()

        if (
            self._last_sent_at is not None
            and now - self._last_sent_at
            < self.cooldown_seconds
        ):
            return False

        owner_name = str(
            details.get(
                "owner_name"
            )
            or "Owner"
        )

        detected_at = str(
            details.get(
                "detected_at_utc"
            )
            or "Unknown time"
        )

        unknown_face_count = int(
            details.get(
                "unknown_face_count",
                1,
            )
        )

        frame_jpeg = details.get(
            "frame_jpeg"
        )

        if not isinstance(
            frame_jpeg,
            bytes,
        ):
            frame_jpeg = None

        subject = (
            "Kibo Guard Alert: "
            "Unknown person detected"
        )

        attachment_line = (
            "A camera frame is attached "
            "to this alert.\n\n"
            if frame_jpeg
            else ""
        )

        body = (
            f"Hello {owner_name},\n\n"
            "Kibo detected an unknown person "
            "near your desk while guard mode "
            "was active.\n\n"
            f"Detected at: {detected_at}\n"
            "Unknown faces detected: "
            f"{unknown_face_count}\n\n"
            f"{attachment_line}"
            "The local guard warning is active. "
            "Please check the area when it is "
            "safe to do so.\n\n"
            "— Kibo Desk Companion"
        )

        task = _EmailTask(
            host=self.host,
            port=self.port,
            use_ssl=self.use_ssl,
            username=self.username,
            password=self.password,
            sender=self.sender,
            recipient=self.recipient,
            subject=subject,
            body=body,
            image_bytes=frame_jpeg,
            timeout_seconds=(
                self.timeout_seconds
            ),
        )

        task.signals.sent.connect(
            self._on_sent
        )

        task.signals.failed.connect(
            self._on_failed
        )

        self._task = task
        self.busy_changed.emit(True)
        self._pool.start(task)

        return True

    @pyqtSlot(str)
    def _on_sent(
        self,
        recipient: str,
    ) -> None:
        self._last_sent_at = (
            time.monotonic()
        )

        self._task = None
        self.busy_changed.emit(False)

        self.notification_sent.emit(
            recipient
        )

    @pyqtSlot(str)
    def _on_failed(
        self,
        message: str,
    ) -> None:
        self._task = None
        self.busy_changed.emit(False)
        self.error.emit(message)

    @pyqtSlot()
    def stop(self) -> None:
        if self._stopping:
            return

        self._stopping = True

        self._pool.clear()
        self._pool.waitForDone()

        self._task = None
        self.busy_changed.emit(False)