"""Evaluate automatic contextual triggers without calling an LLM directly."""

from __future__ import annotations

import time
from typing import Any

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class EventEvaluatorService(QObject):
    """Emit each meaningful trigger once per underlying session or event."""

    trigger_ready = pyqtSignal(str, dict)

    def __init__(
        self,
        inactivity_trigger_s: float = 300.0,
        active_desk_trigger_s: float = 2700.0,
        calendar_lead_s: float = 600.0,
        negative_emotion_trigger_s: float = 180.0,
        minimum_nudge_interval_real_s: float = 15.0,
        post_break_nudge_s: float = 600.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.inactivity_trigger_s = float(
            inactivity_trigger_s
        )
        self.active_desk_trigger_s = float(
            active_desk_trigger_s
        )
        self.calendar_lead_s = float(
            calendar_lead_s
        )
        self.negative_emotion_trigger_s = float(
            negative_emotion_trigger_s
        )
        self.minimum_nudge_interval_real_s = float(
            minimum_nudge_interval_real_s
        )
        self.post_break_nudge_s = float(
            post_break_nudge_s
        )

        self._busy = False
        self._seen_keys: set[str] = set()
        self._last_nudge_real_s = float("-inf")
        self._suppress_inactivity_until_elapsed_s = float(
            "-inf"
        )

    @pyqtSlot(bool)
    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    @pyqtSlot()
    def reset_session(self) -> None:
        self._seen_keys.clear()
        self._last_nudge_real_s = float("-inf")
        self._suppress_inactivity_until_elapsed_s = float(
            "-inf"
        )

    def evaluate(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        if self._busy:
            return

        if state.get("automatic_nudges") is not True:
            return

        real_seconds_since_last_nudge = (
            time.monotonic()
            - self._last_nudge_real_s
        )

        if (
            real_seconds_since_last_nudge
            < self.minimum_nudge_interval_real_s
        ):
            return

        # Trigger priority when multiple conditions
        # become true during the same state update.
        if self._evaluate_calendar(state):
            return

        if self._evaluate_negative_emotion(
            state,
            metrics,
        ):
            return

        if self._evaluate_inactivity(
            state,
            metrics,
        ):
            return

        self._evaluate_active_desk(
            state,
            metrics,
        )

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

        print(
            f"[CONTEXT TRIGGER] "
            f"{trigger_name}: {details}"
        )

        self.trigger_ready.emit(
            trigger_name,
            details,
        )

        return True

    def _evaluate_calendar(
        self,
        state: dict[str, Any],
    ) -> bool:
        # Explicit False means the owner is known
        # to be away. None means camera status unknown.
        if state.get("owner_at_desk") is False:
            return False

        if not state.get("calendar_connected"):
            return False

        event = state.get(
            "next_calendar_event"
        )

        if not event:
            return False

        if bool(event.get("all_day", False)):
            return False

        try:
            seconds_until_start = float(
                event.get("seconds_until_start")
            )
        except (TypeError, ValueError):
            return False

        if not (
            0.0
            < seconds_until_start
            <= self.calendar_lead_s
        ):
            return False

        event_key = (
            event.get("id")
            or (
                f"{event.get('summary')}:"
                f"{event.get('start_utc')}"
            )
        )

        return self._emit_once(
            f"calendar_event_soon:{event_key}",
            "calendar_event_soon",
            {
                "event": dict(event),
            },
        )

    def _evaluate_inactivity(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        owner_at_desk = state.get(
            "owner_at_desk"
        )

        # Explicit False means no owner is present.
        # Do not speak to an empty desk or unknown person.
        if owner_at_desk is False:
            return False

        if not bool(
            state.get("is_inactive", False)
        ):
            return False

        elapsed_s = float(
            state.get(
                "simulated_elapsed_s",
                0.0,
            )
        )

        # An active-at-desk nudge normally suggests
        # a break. Do not immediately interpret that
        # break as procrastination.
        if (
            elapsed_s
            < self._suppress_inactivity_until_elapsed_s
        ):
            return False

        if owner_at_desk is True:
            # Confirmed owner-at-desk inactivity.
            duration_s = float(
                metrics.get(
                    "current_qualified_inactive_s",
                    0.0,
                )
            )

            period = metrics.get(
                "current_inactivity_period"
            )

            if (
                duration_s
                < self.inactivity_trigger_s
            ):
                return False

            if not period:
                return False

            trigger_name = (
                "owner_inactive_at_desk"
            )

        else:
            # Camera unavailable. Use raw input-idle
            # time but phrase the result conditionally.
            duration_s = float(
                metrics.get(
                    "current_input_idle_s",
                    0.0,
                )
            )

            period = None

            if (
                duration_s
                < self.inactivity_trigger_s
            ):
                return False

            trigger_name = (
                "computer_input_inactive"
            )

        input_idle_started_s = max(
            0.0,
            elapsed_s
            - float(
                metrics.get(
                    "current_input_idle_s",
                    duration_s,
                )
            ),
        )

        # Use the raw input-idle start for both
        # inactivity trigger types. This prevents
        # camera changes from causing two nudges
        # during the same inactivity period.
        trigger_key = (
            f"input_inactive:"
            f"{round(input_idle_started_s)}"
        )

        return self._emit_once(
            trigger_key,
            trigger_name,
            {
                "owner_confirmed_at_desk": (
                    owner_at_desk is True
                ),
                "current_inactivity_period": (
                    dict(period)
                    if period
                    else None
                ),
            },
        )

    def _evaluate_active_desk(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        # This trigger requires confirmed presence.
        if state.get("owner_at_desk") is not True:
            return False

        # Never issue an active-work break nudge
        # while the user is currently inactive.
        if bool(
            state.get("is_inactive", False)
        ):
            return False

        # Effective time excludes qualifying
        # inactivity during the desk period.
        duration_s = float(
            metrics.get(
                "effective_active_at_desk_s",
                0.0,
            )
        )

        desk_started_s = metrics.get(
            "continuous_owner_at_desk_started_elapsed_s"
        )

        if (
            duration_s
            < self.active_desk_trigger_s
        ):
            return False

        if desk_started_s is None:
            return False

        emitted = self._emit_once(
            (
                "active_at_desk_too_long:"
                f"{desk_started_s}"
            ),
            "active_at_desk_too_long",
            {
                "effective_active_at_desk_s": (
                    duration_s
                ),
                "continuous_owner_at_desk_s": float(
                    metrics.get(
                        "continuous_owner_at_desk_s",
                        0.0,
                    )
                ),
                "inactive_during_current_desk_s": float(
                    metrics.get(
                        "inactive_during_current_desk_s",
                        0.0,
                    )
                ),
            },
        )

        if emitted:
            elapsed_s = float(
                state.get(
                    "simulated_elapsed_s",
                    0.0,
                )
            )

            # Allow the suggested break to happen
            # without producing an inactivity nudge.
            self._suppress_inactivity_until_elapsed_s = (
                elapsed_s
                + self.post_break_nudge_s
            )

        return emitted

    def _evaluate_negative_emotion(
        self,
        state: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        # Expression estimates are only relevant
        # when the owner is positively identified.
        if state.get("owner_at_desk") is not True:
            return False

        if not metrics.get(
            "negative_emotion_active"
        ):
            return False

        duration_s = float(
            metrics.get(
                "negative_emotion_duration_s",
                0.0,
            )
        )

        period = metrics.get(
            "current_negative_emotion"
        )

        if (
            duration_s
            < self.negative_emotion_trigger_s
        ):
            return False

        if not period:
            return False

        started_s = period.get(
            "started_elapsed_s",
            "unknown",
        )

        return self._emit_once(
            (
                "negative_emotion_sustained:"
                f"{started_s}"
            ),
            "negative_emotion_sustained",
            {
                "current_negative_emotion": dict(
                    period
                )
            },
        )