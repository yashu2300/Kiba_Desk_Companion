"""A monotonic clock that can be accelerated for demonstrations."""

import time
from datetime import datetime, timedelta, timezone
from threading import RLock


class DemoClock:
    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError(
                "DemoClock speed must be greater than zero."
            )

        self._lock = RLock()
        self._speed = float(speed)
        self._real_anchor = time.monotonic()
        self._simulated_anchor = 0.0
        self._utc_anchor = datetime.now(timezone.utc)

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    def elapsed_seconds(self) -> float:
        """Return elapsed simulated seconds."""

        with self._lock:
            real_elapsed = (
                time.monotonic() - self._real_anchor
            )

            return (
                self._simulated_anchor
                + real_elapsed * self._speed
            )

    def now_utc(self) -> datetime:
        """Return the current simulated UTC date and time."""

        return self._utc_anchor + timedelta(
            seconds=self.elapsed_seconds()
        )

    def set_speed(self, speed: float) -> None:
        """Change speed without causing elapsed time to jump."""

        if speed <= 0:
            raise ValueError(
                "DemoClock speed must be greater than zero."
            )

        with self._lock:
            now = time.monotonic()

            self._simulated_anchor += (
                now - self._real_anchor
            ) * self._speed

            self._real_anchor = now
            self._speed = float(speed)