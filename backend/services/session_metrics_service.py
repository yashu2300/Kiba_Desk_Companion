"""Track simulated away, input-inactivity and expression periods."""

from __future__ import annotations

from collections import deque
from datetime import timedelta
from typing import Any

from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
    pyqtSlot,
)

from backend.demo_clock import DemoClock


class SessionMetricsService(QObject):
    """Build session-scoped metrics from CurrentState updates."""

    metrics_changed = pyqtSignal(dict)
    period_completed = pyqtSignal(dict)

    NEGATIVE_EXPRESSIONS = {
        "Angry",
        "Disgust",
        "Fearful",
        "Sad",
    }

    def __init__(
        self,
        clock: DemoClock,
        session_id: int,
        minimum_break_s: float = 60.0,
        inactivity_threshold_s: float = 300.0,
        negative_confidence_threshold: float = 0.70,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.clock = clock
        self.session_id = session_id

        self.minimum_break_s = max(
            0.0,
            float(minimum_break_s),
        )

        self.inactivity_threshold_s = max(
            0.0,
            float(inactivity_threshold_s),
        )

        self.negative_confidence_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    negative_confidence_threshold
                ),
            ),
        )

        self._latest_state: dict[
            str,
            Any,
        ] = {}

        self._session_open = True

        self._previous_owner_at_desk: (
            bool | None
        ) = None

        self._away_started: (
            dict[str, Any] | None
        ) = None

        self._inactive_started: (
            dict[str, Any] | None
        ) = None

        self._negative_emotion_started: (
            dict[str, Any] | None
        ) = None

        self._continuous_desk_started_s: (
            float | None
        ) = None

        self._times_left_desk = 0
        self._break_count = 0
        self._inactivity_session_count = 0

        self._completed_away_s = 0.0
        self._completed_inactive_s = 0.0

        self._last_break: (
            dict[str, Any] | None
        ) = None

        self._away_periods: deque[
            dict[str, Any]
        ] = deque(maxlen=50)

        self._inactive_periods: deque[
            dict[str, Any]
        ] = deque(maxlen=50)

        self._negative_emotion_periods: deque[
            dict[str, Any]
        ] = deque(maxlen=50)

    @staticmethod
    def _new_period(
        started_elapsed_s: float,
        started_at_simulated_utc: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "started_elapsed_s": float(
                started_elapsed_s
            ),
            "started_at_simulated_utc": (
                started_at_simulated_utc
            ),
            "details": details or {},
        }

    def _complete_period(
        self,
        period_type: str,
        started: dict[str, Any],
        ended_elapsed_s: float,
        ended_at_simulated_utc: str,
        extra_details: (
            dict[str, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        duration_s = max(
            0.0,
            ended_elapsed_s
            - float(
                started[
                    "started_elapsed_s"
                ]
            ),
        )

        details = dict(
            started.get("details", {})
        )

        details.update(
            extra_details or {}
        )

        period = {
            "session_id": self.session_id,
            "period_type": period_type,
            "started_elapsed_s": round(
                float(
                    started[
                        "started_elapsed_s"
                    ]
                ),
                3,
            ),
            "ended_elapsed_s": round(
                float(ended_elapsed_s),
                3,
            ),
            "started_at_simulated_utc": (
                started[
                    "started_at_simulated_utc"
                ]
            ),
            "ended_at_simulated_utc": (
                ended_at_simulated_utc
            ),
            "duration_s": round(
                duration_s,
                3,
            ),
            "details": details,
        }

        if period_type == "away":
            self._away_periods.append(
                period
            )

            self._completed_away_s += (
                duration_s
            )

            if (
                duration_s
                >= self.minimum_break_s
            ):
                self._break_count += 1
                self._last_break = period

        elif period_type == "computer_inactive":
            self._inactive_periods.append(
                period
            )

            self._completed_inactive_s += (
                duration_s
            )

        elif period_type == "negative_emotion":
            self._negative_emotion_periods.append(
                period
            )

        self.period_completed.emit(
            period
        )

        return period

    @pyqtSlot(dict)
    def observe_state(
        self,
        state: dict,
    ) -> None:
        if not self._session_open:
            return

        self._latest_state = dict(
            state
        )

        elapsed_s = float(
            state.get(
                "simulated_elapsed_s",
                self.clock.elapsed_seconds(),
            )
        )

        simulated_utc = (
            self.clock.now_utc().isoformat()
        )

        self._observe_owner_presence(
            state,
            elapsed_s,
            simulated_utc,
        )

        self._observe_inactivity(
            state,
            elapsed_s,
            simulated_utc,
        )

        self._observe_emotion(
            state,
            elapsed_s,
            simulated_utc,
        )

        self.metrics_changed.emit(
            self.summary()
        )

    def _observe_owner_presence(
        self,
        state: dict,
        elapsed_s: float,
        simulated_utc: str,
    ) -> None:
        owner_at_desk = state.get(
            "owner_at_desk"
        )

        previous_owner = (
            self._previous_owner_at_desk
        )

        if (
            previous_owner is True
            and owner_at_desk is False
        ):
            self._times_left_desk += 1

            self._away_started = (
                self._new_period(
                    elapsed_s,
                    simulated_utc,
                )
            )

            self._continuous_desk_started_s = (
                None
            )

        elif (
            previous_owner is False
            and owner_at_desk is True
        ):
            if self._away_started is not None:
                self._complete_period(
                    "away",
                    self._away_started,
                    elapsed_s,
                    simulated_utc,
                )

            self._away_started = None
            self._continuous_desk_started_s = (
                elapsed_s
            )

        elif (
            owner_at_desk is True
            and self._continuous_desk_started_s
            is None
        ):
            self._continuous_desk_started_s = (
                elapsed_s
            )

        elif owner_at_desk is None:
            if self._away_started is not None:
                self._complete_period(
                    "away",
                    self._away_started,
                    elapsed_s,
                    simulated_utc,
                    {
                        "end_reason": (
                            "vision_unavailable"
                        )
                    },
                )

            self._away_started = None
            self._continuous_desk_started_s = (
                None
            )

        self._previous_owner_at_desk = (
            owner_at_desk
        )

    def _observe_inactivity(
        self,
        state: dict,
        elapsed_s: float,
        simulated_utc: str,
    ) -> None:
        is_inactive = bool(
            state.get(
                "is_inactive",
                False,
            )
        )

        inactive_s = float(
            state.get(
                "inactive_seconds",
                0.0,
            )
        )

        owner_at_desk = state.get(
            "owner_at_desk"
        )

        desk_started_s = (
            self._continuous_desk_started_s
        )

        # Raw simulated time of the last input.
        possible_start_s = max(
            0.0,
            elapsed_s - inactive_s,
        )

        # If the owner has only just returned,
        # do not backdate desk inactivity through
        # the period when they were away.
        if desk_started_s is not None:
            possible_start_s = max(
                possible_start_s,
                desk_started_s,
            )

        owner_idle_s = max(
            0.0,
            elapsed_s - possible_start_s,
        )

        qualifying = (
            owner_at_desk is True
            and is_inactive
            and owner_idle_s
            >= self.inactivity_threshold_s
        )

        if (
            qualifying
            and self._inactive_started is None
        ):
            started_utc = (
                self.clock.now_utc()
                - timedelta(
                    seconds=owner_idle_s
                )
            )

            self._inactivity_session_count += 1

            self._inactive_started = (
                self._new_period(
                    possible_start_s,
                    started_utc.isoformat(),
                    {
                        "threshold_crossed_at_simulated_utc": (
                            simulated_utc
                        )
                    },
                )
            )

        elif (
            not qualifying
            and self._inactive_started is not None
        ):
            if owner_at_desk is False:
                end_reason = (
                    "owner_left_desk"
                )

            elif owner_at_desk is None:
                end_reason = (
                    "vision_unavailable"
                )

            else:
                end_reason = (
                    "user_active"
                )

            self._complete_period(
                "computer_inactive",
                self._inactive_started,
                elapsed_s,
                simulated_utc,
                {
                    "end_reason": end_reason
                },
            )

            self._inactive_started = None

    def _observe_emotion(
        self,
        state: dict,
        elapsed_s: float,
        simulated_utc: str,
    ) -> None:
        expression = str(
            state.get(
                "expression",
                "Unknown",
            )
        )

        confidence = float(
            state.get(
                "expression_confidence",
                0.0,
            )
        )

        owner_at_desk = state.get(
            "owner_at_desk"
        )

        qualifying = (
            owner_at_desk is True
            and expression
            in self.NEGATIVE_EXPRESSIONS
            and confidence
            >= self.negative_confidence_threshold
        )

        if (
            qualifying
            and self._negative_emotion_started
            is None
        ):
            self._negative_emotion_started = (
                self._new_period(
                    elapsed_s,
                    simulated_utc,
                    {
                        "starting_expression": (
                            expression
                        ),
                        "starting_confidence": (
                            confidence
                        ),
                    },
                )
            )

        elif (
            not qualifying
            and self._negative_emotion_started
            is not None
        ):
            self._complete_period(
                "negative_emotion",
                self._negative_emotion_started,
                elapsed_s,
                simulated_utc,
                {
                    "ending_expression": (
                        expression
                    ),
                    "ending_confidence": (
                        confidence
                    ),
                },
            )

            self._negative_emotion_started = (
                None
            )

    @staticmethod
    def _running_duration(
        started: dict[str, Any] | None,
        elapsed_s: float,
    ) -> float:
        if started is None:
            return 0.0

        return max(
            0.0,
            elapsed_s
            - float(
                started[
                    "started_elapsed_s"
                ]
            ),
        )

    @staticmethod
    def _overlap_seconds(
        period_start_s: float,
        period_end_s: float,
        window_start_s: float,
        window_end_s: float,
    ) -> float:
        return max(
            0.0,
            min(
                period_end_s,
                window_end_s,
            )
            - max(
                period_start_s,
                window_start_s,
            ),
        )

    def _inactivity_during_current_desk(
        self,
        desk_started_s: float,
        elapsed_s: float,
    ) -> float:
        total_s = 0.0

        for period in self._inactive_periods:
            total_s += self._overlap_seconds(
                float(
                    period[
                        "started_elapsed_s"
                    ]
                ),
                float(
                    period[
                        "ended_elapsed_s"
                    ]
                ),
                desk_started_s,
                elapsed_s,
            )

        if self._inactive_started is not None:
            total_s += self._overlap_seconds(
                float(
                    self._inactive_started[
                        "started_elapsed_s"
                    ]
                ),
                elapsed_s,
                desk_started_s,
                elapsed_s,
            )

        return total_s

    def summary(self) -> dict[str, Any]:
        elapsed_s = (
            self.clock.elapsed_seconds()
        )

        current_away_s = (
            self._running_duration(
                self._away_started,
                elapsed_s,
            )
        )

        current_qualified_inactive_s = (
            self._running_duration(
                self._inactive_started,
                elapsed_s,
            )
        )

        negative_emotion_s = (
            self._running_duration(
                self._negative_emotion_started,
                elapsed_s,
            )
        )

        current_input_idle_s = float(
            self._latest_state.get(
                "inactive_seconds",
                0.0,
            )
        )

        owner_at_desk = (
            self._latest_state.get(
                "owner_at_desk"
            )
        )

        continuous_desk_s = 0.0

        if (
            owner_at_desk is True
            and self._continuous_desk_started_s
            is not None
        ):
            continuous_desk_s = max(
                0.0,
                elapsed_s
                - self._continuous_desk_started_s,
            )

        inactive_during_current_desk_s = (
            0.0
        )

        if (
            owner_at_desk is True
            and self._continuous_desk_started_s
            is not None
        ):
            inactive_during_current_desk_s = (
                self._inactivity_during_current_desk(
                    self._continuous_desk_started_s,
                    elapsed_s,
                )
            )

        effective_active_at_desk_s = max(
            0.0,
            continuous_desk_s
            - inactive_during_current_desk_s,
        )

        current_inactivity_period = None

        if self._inactive_started is not None:
            current_inactivity_period = {
                "started_elapsed_s": round(
                    float(
                        self._inactive_started[
                            "started_elapsed_s"
                        ]
                    ),
                    3,
                ),
                "started_at_simulated_utc": (
                    self._inactive_started[
                        "started_at_simulated_utc"
                    ]
                ),
                "duration_s": round(
                    current_qualified_inactive_s,
                    3,
                ),
            }

        current_negative_emotion = None

        if (
            self._negative_emotion_started
            is not None
        ):
            current_negative_emotion = {
                "started_elapsed_s": round(
                    float(
                        self._negative_emotion_started[
                            "started_elapsed_s"
                        ]
                    ),
                    3,
                ),
                "started_at_simulated_utc": (
                    self._negative_emotion_started[
                        "started_at_simulated_utc"
                    ]
                ),
                "duration_s": round(
                    negative_emotion_s,
                    3,
                ),
                "expression": str(
                    self._latest_state.get(
                        "expression",
                        "Unknown",
                    )
                ),
                "confidence": float(
                    self._latest_state.get(
                        "expression_confidence",
                        0.0,
                    )
                ),
            }

        return {
            "session_id": self.session_id,
            "current_away_s": round(
                current_away_s,
                3,
            ),
            "total_away_s": round(
                self._completed_away_s
                + current_away_s,
                3,
            ),
            "times_left_desk": (
                self._times_left_desk
            ),
            "break_count": (
                self._break_count
            ),
            "last_break": (
                self._last_break
            ),
            "continuous_owner_at_desk_s": round(
                continuous_desk_s,
                3,
            ),
            "inactive_during_current_desk_s": round(
                inactive_during_current_desk_s,
                3,
            ),
            "effective_active_at_desk_s": round(
                effective_active_at_desk_s,
                3,
            ),
            "continuous_owner_at_desk_started_elapsed_s": (
                round(
                    self._continuous_desk_started_s,
                    3,
                )
                if self._continuous_desk_started_s
                is not None
                else None
            ),
            "current_input_idle_s": round(
                current_input_idle_s,
                3,
            ),
            "current_qualified_inactive_s": round(
                current_qualified_inactive_s,
                3,
            ),
            "current_inactivity_period": (
                current_inactivity_period
            ),
            "total_qualified_inactive_s": round(
                self._completed_inactive_s
                + current_qualified_inactive_s,
                3,
            ),
            "inactivity_session_count": (
                self._inactivity_session_count
            ),
            "recent_away_periods": list(
                self._away_periods
            ),
            "recent_inactivity_periods": list(
                self._inactive_periods
            ),
            "negative_emotion_active": (
                self._negative_emotion_started
                is not None
            ),
            "negative_emotion_duration_s": round(
                negative_emotion_s,
                3,
            ),
            "current_negative_emotion": (
                current_negative_emotion
            ),
            "recent_negative_emotion_periods": list(
                self._negative_emotion_periods
            ),
        }

    @pyqtSlot()
    def finish_session(self) -> None:
        if not self._session_open:
            return

        elapsed_s = (
            self.clock.elapsed_seconds()
        )

        simulated_utc = (
            self.clock.now_utc().isoformat()
        )

        if self._away_started is not None:
            self._complete_period(
                "away",
                self._away_started,
                elapsed_s,
                simulated_utc,
                {
                    "end_reason": (
                        "session_ended"
                    )
                },
            )

        if self._inactive_started is not None:
            self._complete_period(
                "computer_inactive",
                self._inactive_started,
                elapsed_s,
                simulated_utc,
                {
                    "end_reason": (
                        "session_ended"
                    )
                },
            )

        if (
            self._negative_emotion_started
            is not None
        ):
            self._complete_period(
                "negative_emotion",
                self._negative_emotion_started,
                elapsed_s,
                simulated_utc,
                {
                    "end_reason": (
                        "session_ended"
                    )
                },
            )

        self._session_open = False
        self._away_started = None
        self._inactive_started = None
        self._negative_emotion_started = None
        self._continuous_desk_started_s = (
            None
        )

    @pyqtSlot(int)
    def reset_session(
        self,
        session_id: int,
    ) -> None:
        self.session_id = session_id
        self._session_open = True
        self._latest_state = {}
        self._previous_owner_at_desk = None
        self._away_started = None
        self._inactive_started = None
        self._negative_emotion_started = None
        self._continuous_desk_started_s = None

        self._times_left_desk = 0
        self._break_count = 0
        self._inactivity_session_count = 0

        self._completed_away_s = 0.0
        self._completed_inactive_s = 0.0
        self._last_break = None

        self._away_periods.clear()
        self._inactive_periods.clear()
        self._negative_emotion_periods.clear()