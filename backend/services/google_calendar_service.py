"""Read-only Google Calendar service with DemoClock monitoring."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PyQt5.QtCore import (
    QObject,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)

from backend.demo_clock import DemoClock


SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly"
]


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m"

    if minutes:
        return f"{minutes}m"

    return f"{seconds}s"


def _format_time(value: datetime) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


class _CalendarFetchThread(QThread):
    """Perform OAuth and API calls outside the GUI thread."""

    events_loaded = pyqtSignal(object)
    fetch_failed = pyqtSignal(str)

    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        calendar_id: str,
        timezone_name: str,
        day_start_utc: datetime,
        day_end_utc: datetime,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.credentials_path = credentials_path
        self.token_path = token_path
        self.calendar_id = calendar_id
        self.timezone_name = timezone_name
        self.day_start_utc = day_start_utc
        self.day_end_utc = day_end_utc

    def _load_credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import (
            InstalledAppFlow,
        )

        credentials = None

        if self.token_path.exists():
            credentials = (
                Credentials.from_authorized_user_file(
                    str(self.token_path),
                    SCOPES,
                )
            )

        if (
            credentials
            and credentials.expired
            and credentials.refresh_token
        ):
            credentials.refresh(Request())

        if not credentials or not credentials.valid:
            if not self.credentials_path.exists():
                raise FileNotFoundError(
                    "Google OAuth credentials were not found at "
                    f"{self.credentials_path}"
                )

            flow = (
                InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path),
                    SCOPES,
                )
            )

            credentials = flow.run_local_server(
                port=0
            )

        self.token_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.token_path.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

        return credentials

    def run(self) -> None:
        stage = "starting calendar request"

        try:
            from googleapiclient.discovery import build
            import httplib2
            from google_auth_httplib2 import (
                AuthorizedHttp,
            )

            stage = "loading or refreshing OAuth credentials"
            credentials = self._load_credentials()
            if self.isInterruptionRequested():
                return

            stage = "building Google Calendar client"

            authorized_http = AuthorizedHttp(
                credentials,
                http=httplib2.Http(
                    timeout=20
                ),
            )

            api = build(
                "calendar",
                "v3",
                http=authorized_http,
                cache_discovery=False,
                static_discovery=True,
            )

            stage = "fetching calendar events"

            response = (
                api.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=(
                        self.day_start_utc.isoformat()
                    ),
                    timeMax=(
                        self.day_end_utc.isoformat()
                    ),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=2500,
                    timeZone=self.timezone_name,
                )
                .execute(num_retries=2)
            )

            if self.isInterruptionRequested():
                return

            local_zone = ZoneInfo(
                self.timezone_name
            )

            events: list[dict[str, Any]] = []

            for item in response.get("items", []):
                if item.get("status") == "cancelled":
                    continue

                start_data = item.get("start", {})
                end_data = item.get("end", {})
                all_day = "date" in start_data

                if all_day:
                    start_local = datetime.combine(
                        date.fromisoformat(
                            start_data["date"]
                        ),
                        time.min,
                        tzinfo=local_zone,
                    )

                    end_local = datetime.combine(
                        date.fromisoformat(
                            end_data["date"]
                        ),
                        time.min,
                        tzinfo=local_zone,
                    )
                else:
                    start_local = _parse_datetime(
                        start_data["dateTime"]
                    ).astimezone(local_zone)

                    end_local = _parse_datetime(
                        end_data["dateTime"]
                    ).astimezone(local_zone)

                events.append(
                    {
                        "id": str(item.get("id", "")),
                        "summary": str(
                            item.get(
                                "summary",
                                "Untitled event",
                            )
                        ),
                        "location": str(
                            item.get("location", "")
                        ),
                        "start_utc": (
                            start_local.astimezone(
                                timezone.utc
                            ).isoformat()
                        ),
                        "end_utc": (
                            end_local.astimezone(
                                timezone.utc
                            ).isoformat()
                        ),
                        "all_day": all_day,
                        "html_link": str(
                            item.get("htmlLink", "")
                        ),
                    }
                )

            calendar_day = (
                self.day_start_utc
                .astimezone(local_zone)
                .date()
                .isoformat()
            )

            self.events_loaded.emit(
                {
                    "events": events,
                    "calendar_day": calendar_day,
                    "synced_at_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                }
            )

        except ImportError:
            self.fetch_failed.emit(
                "Google Calendar packages are missing. "
                "Install google-api-python-client, "
                "google-auth-httplib2, "
                "google-auth-oauthlib and tzdata."
            )

        except Exception as error:
            if self.isInterruptionRequested():
                return

            self.fetch_failed.emit(
                f"{stage} failed: {type(error).__name__}: {error}"
            )


class GoogleCalendarService(QObject):
    """Fetch and monitor today's Google Calendar events."""

    calendar_updated = pyqtSignal(dict)
    busy_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(
        self,
        clock: DemoClock,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        calendar_id: str = "primary",
        timezone_name: str = "Pacific/Auckland",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        root = Path(__file__).resolve().parents[2]

        self.clock = clock

        self.credentials_path = (
            credentials_path
            or root
            / "credentials"
            / "google_calendar_credentials.json"
        )

        self.token_path = (
            token_path
            or root
            / "data"
            / "google_calendar_token.json"
        )

        self.calendar_id = calendar_id
        self.timezone_name = timezone_name
        self.local_zone = ZoneInfo(
            timezone_name
        )

        self._events: list[dict[str, Any]] = []
        self._calendar_day = ""
        self._synced_at_utc: str | None = None
        self._connected = False

        self._worker: (
            _CalendarFetchThread | None
        ) = None

        self._pending_refresh = False
        self._connection_retry_count = 0

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(
            self._tick
        )

    @pyqtSlot()
    def start(self) -> None:
        self._stopping = False
        self._status_timer.start()
        self.refresh()


    @pyqtSlot()
    def stop(self) -> None:
        self._stopping = True
        self._pending_refresh = False
        self._status_timer.stop()

        worker = self._worker

        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(5000)

    @pyqtSlot()
    def refresh(self) -> None:
        if self._stopping:
            return
        if (
            self._worker is not None
            and self._worker.isRunning()
        ):
            self._pending_refresh = True
            return

        simulated_now = (
            self.clock.now_utc()
            .astimezone(self.local_zone)
        )

        day_start_local = datetime.combine(
            simulated_now.date(),
            time.min,
            tzinfo=self.local_zone,
        )

        day_end_local = (
            day_start_local
            + timedelta(days=1)
        )

        self._worker = _CalendarFetchThread(
            credentials_path=(
                self.credentials_path
            ),
            token_path=self.token_path,
            calendar_id=self.calendar_id,
            timezone_name=self.timezone_name,
            day_start_utc=(
                day_start_local.astimezone(
                    timezone.utc
                )
            ),
            day_end_utc=(
                day_end_local.astimezone(
                    timezone.utc
                )
            ),
            parent=self,
        )

        self._worker.events_loaded.connect(
            self._on_events_loaded
        )

        self._worker.fetch_failed.connect(
            self._on_fetch_failed
        )

        self._worker.finished.connect(
            self._on_worker_finished
        )

        self.busy_changed.emit(True)
        self._worker.start()

    @pyqtSlot(object)
    def _on_events_loaded(self, result: object) -> None:
        if self._stopping:
            return
        
        self._connection_retry_count = 0
        payload = dict(result)

        self._events = list(
            payload.get("events", [])
        )

        self._calendar_day = str(
            payload.get("calendar_day", "")
        )

        self._synced_at_utc = payload.get(
            "synced_at_utc"
        )

        self._connected = True
        self._publish()

    @pyqtSlot(str)
    def _on_fetch_failed(
        self,
        message: str,
    ) -> None:
        if self._stopping:
            return
        self._connected = False
        self.error.emit(message)
        self._publish()
        if self._connection_retry_count < 1:
            self._connection_retry_count += 1
            print(
                "[CALENDAR RETRY] "
                "Retrying in 5 seconds."
            )

            QTimer.singleShot(
                5000,
                self.refresh,
            )           

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None

        self.busy_changed.emit(False)

        if worker is not None:
            worker.deleteLater()

        if self._pending_refresh and not self._stopping:
            self._pending_refresh = False

            QTimer.singleShot(
                0,
                self.refresh,
            )

    @pyqtSlot()
    def _tick(self) -> None:
        simulated_day = (
            self.clock.now_utc()
            .astimezone(self.local_zone)
            .date()
            .isoformat()
        )

        if (
            self._calendar_day
            and simulated_day
            != self._calendar_day
        ):
            self.refresh()
            return

        self._publish()

    def _publish(self) -> None:
        now_utc = self.clock.now_utc()
        classified: list[dict[str, Any]] = []

        for source in self._events:
            event = dict(source)

            start_utc = _parse_datetime(
                event["start_utc"]
            )

            end_utc = _parse_datetime(
                event["end_utc"]
            )

            start_local = start_utc.astimezone(
                self.local_zone
            )

            end_local = end_utc.astimezone(
                self.local_zone
            )

            if now_utc >= end_utc:
                status = "past"
                status_text = "PAST"

            elif now_utc >= start_utc:
                status = "happening_now"

                if event["all_day"]:
                    status_text = "TODAY"
                else:
                    remaining = (
                        end_utc - now_utc
                    ).total_seconds()

                    status_text = (
                        "NOW · "
                        f"{_format_duration(remaining)} "
                        "left"
                    )

            else:
                status = "upcoming"

                until_start = (
                    start_utc - now_utc
                ).total_seconds()

                status_text = (
                    "IN "
                    f"{_format_duration(until_start)}"
                )

            if event["all_day"]:
                time_text = "All day"
            else:
                time_text = (
                    f"{_format_time(start_local)} – "
                    f"{_format_time(end_local)}"
                )

            event.update(
                {
                    "status": status,
                    "status_text": status_text,
                    "time_text": time_text,
                    "seconds_until_start": max(
                        0.0,
                        (
                            start_utc - now_utc
                        ).total_seconds(),
                    ),
                    "seconds_until_end": max(
                        0.0,
                        (
                            end_utc - now_utc
                        ).total_seconds(),
                    ),
                }
            )

            classified.append(event)

        current_events = [
            event
            for event in classified
            if event["status"]
            == "happening_now"
        ]

        upcoming_events = [
            event
            for event in classified
            if event["status"]
            == "upcoming"
        ]

        self.calendar_updated.emit(
            {
                "connected": self._connected,
                "calendar_day": (
                    self._calendar_day
                ),
                "events": classified,
                "current_event": (
                    current_events[0]
                    if current_events
                    else None
                ),
                "next_event": (
                    upcoming_events[0]
                    if upcoming_events
                    else None
                ),
                "past_count": sum(
                    event["status"] == "past"
                    for event in classified
                ),
                "current_count": len(
                    current_events
                ),
                "upcoming_count": len(
                    upcoming_events
                ),
                "synced_at_utc": (
                    self._synced_at_utc
                ),
                "simulated_now_utc": (
                    now_utc.isoformat()
                ),
            }
        )