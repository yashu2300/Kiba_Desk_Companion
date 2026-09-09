"""Evaluate contextual triggers without calling an LLM directly."""

from __future__ import annotations

import time
from typing import Any

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class EventEvaluatorService(QObject):
    """Emit each trigger once per underlying period or event."""

    trigger_ready = pyqtSignal(str, dict)

    def __init__(
        self,
        inactivity_trigger_s: float = 300.0,
        active_desk_trigger_s: float = 2700.0,
        calendar_lead_s: float = 600.0,
        negative_emotion_trigger_s: float = 180.0,
        minimum_nudge_interval_real_s: float = 15.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.inactivity_trigger_s = float(inactivity_trigger_s)
        self.active_desk_trigger_s = float(active_desk_trigger_s)
        self.calendar_lead_s = float(calendar_lead_s)
        self.negative_emotion_trigger_s = float(
            negative_emotion_trigger_s
        )
        self.minimum_nudge_interval_real_s = float(
            minimum_nudge_interval_real_s
        )

        self._busy = False
        self._seen_keys: set[str] = set()
        self._last_nudge_real_s = float("-inf")

    @pyqtSlot(bool)
    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    @pyqtSlot()
    def reset_session(self) -> None:
        self._seen_keys.clear()
        self._last_nudge_real_s = float("-inf")

    def evaluate(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        if self._busy:
            return

        if state.get("automatic_nudges") is not True:
            return

        elapsed_real_s = (
            time.monotonic() - self._last_nudge_real_s
        )

        if elapsed_real_s < self.minimum_nudge_interval_real_s:
            return

        # Priority order when several conditions become true together.
        if self._evaluate_calendar(state):
            return

        if self._evaluate_negative_emotion(state, metrics):
            return

        if self._evaluate_inactivity(state, metrics):
            return

        self._evaluate_active_desk(state, metrics)

    def _emit_once(
        self,
        key: str,
        trigger_name: str,
        details: dict[str, Any],
    ) -> bool:
        if key in self._seen_keys:
            return False

        self._seen_keys.add(key)
        self._last_nudge_real_s = time.monotonic()

        print(f"[CONTEXT TRIGGER] {trigger_name}: {details}")
        self.trigger_ready.emit(trigger_name, details)
        return True

    def _evaluate_calendar(
        self,
        state: dict[str, Any],
    ) -> bool:
        if state.get("owner_at_desk") is False:
            return False    

        if not state.get("calendar_connected"):
            return False

        event = state.get("next_calendar_event")

        if not event or bool(event.get("all_day", False)):
            return False

        try:
            seconds_until_start = float(
                event.get("seconds_until_start")
            )
        except (TypeError, ValueError):
            return False

        if not 0.0 < seconds_until_start <= self.calendar_lead_s:
            return False

        event_key = (
            event.get("id")
            or f"{event.get('summary')}:{event.get('start_utc')}"
        )

        return self._emit_once(
            f"calendar_event_soon:{event_key}",
            "calendar_event_soon",
            {"event": dict(event)},
        )

    def _evaluate_inactivity(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        # False means the camera explicitly determined that
        # the owner is not at the desk. None means unknown,
        # usually because the camera is disabled.
        if state.get("owner_at_desk") is False:
            return False

        if not bool(state.get("is_inactive", False)):
            return False

        duration_s = float(
            metrics.get("current_qualified_inactive_s", 0.0)
        )
        period = metrics.get("current_inactivity_period")

        if duration_s < self.inactivity_trigger_s or not period:
            return False

        start = period.get("started_elapsed_s", "unknown")

        return self._emit_once(
            f"computer_input_inactive:{start}",
            "computer_input_inactive",
            {"current_inactivity_period": dict(period)},
        )

    def _evaluate_active_desk(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        if state.get("owner_at_desk") is not True:
            return False

        # Do not call this "active" while input inactivity is active.
        if bool(state.get("is_inactive", False)):
            return False

        duration_s = float(
            metrics.get("continuous_owner_at_desk_s", 0.0)
        )
        start = metrics.get(
            "continuous_owner_at_desk_started_elapsed_s"
        )

        if duration_s < self.active_desk_trigger_s:
            return False

        if start is None:
            return False

        return self._emit_once(
            f"active_at_desk_too_long:{start}",
            "active_at_desk_too_long",
            {"continuous_owner_at_desk_s": duration_s},
        )

    def _evaluate_negative_emotion(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        if state.get("owner_at_desk") is not True:
            return False

        if not metrics.get("negative_emotion_active"):
            return False

        duration_s = float(
            metrics.get("negative_emotion_duration_s", 0.0)
        )
        period = metrics.get("current_negative_emotion")

        if duration_s < self.negative_emotion_trigger_s:
            return False

        if not period:
            return False

        start = period.get("started_elapsed_s", "unknown")

        return self._emit_once(
            f"negative_emotion_sustained:{start}",
            "negative_emotion_sustained",
            {"current_negative_emotion": dict(period)},
        )