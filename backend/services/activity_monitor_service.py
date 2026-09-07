"""Global, privacy-preserving mouse and keyboard activity monitor."""

from __future__ import annotations

import time
from threading import Lock

from pynput import keyboard, mouse
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class ActivityMonitorService(QObject):
    """Emit activity pulses without retaining keys or coordinates."""

    activity_detected = pyqtSignal(str)
    started = pyqtSignal()
    stopped = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        throttle_seconds: float = 0.25,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.throttle_seconds = max(
            0.1,
            float(throttle_seconds),
        )

        self._mouse_listener: (
            mouse.Listener | None
        ) = None

        self._keyboard_listener: (
            keyboard.Listener | None
        ) = None

        self._running = False
        self._last_emitted_at = float("-inf")
        self._lock = Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def _report_activity(
        self,
        source: str,
    ) -> None:
        """Called by pynput's native listener threads."""

        now = time.monotonic()

        with self._lock:
            if not self._running:
                return

            if (
                now - self._last_emitted_at
                < self.throttle_seconds
            ):
                return

            self._last_emitted_at = now

        # Qt safely queues this signal into the
        # CurrentStateManager's thread.
        self.activity_detected.emit(source)

    def _on_mouse_move(
        self,
        _x: int,
        _y: int,
    ) -> None:
        # Coordinates are deliberately discarded.
        self._report_activity("mouse")

    def _on_mouse_click(
        self,
        _x: int,
        _y: int,
        _button: mouse.Button,
        pressed: bool,
    ) -> None:
        # Count only the press, not both press and release.
        if pressed:
            self._report_activity("mouse")

    def _on_mouse_scroll(
        self,
        _x: int,
        _y: int,
        _dx: int,
        _dy: int,
    ) -> None:
        self._report_activity("mouse")

    def _on_key_press(
        self,
        _key: object,
    ) -> None:
        # The key value is deliberately ignored.
        self._report_activity("keyboard")

    @pyqtSlot()
    def start(self) -> None:
        if self._running:
            return

        try:
            with self._lock:
                self._running = True
                self._last_emitted_at = float("-inf")

            # pynput listeners cannot be restarted after stopping.
            # Therefore, create new instances on every start.
            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll,
                suppress=False,
            )

            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                suppress=False,
            )

            self._mouse_listener.start()
            self._keyboard_listener.start()

            self.started.emit()

        except Exception as error:
            with self._lock:
                self._running = False

            self._stop_listeners()

            self.error.emit(
                "Global activity monitor could not "
                f"start: {error}"
            )

    def _stop_listeners(self) -> None:
        listeners = (
            self._mouse_listener,
            self._keyboard_listener,
        )

        self._mouse_listener = None
        self._keyboard_listener = None

        for listener in listeners:
            if listener is not None:
                listener.stop()

        for listener in listeners:
            if (
                listener is not None
                and listener.is_alive()
            ):
                listener.join(timeout=1.0)

    @pyqtSlot()
    def stop(self) -> None:
        with self._lock:
            was_running = self._running
            self._running = False

        self._stop_listeners()

        if was_running:
            self.stopped.emit()