"""Current contextual state and bounded short-term state memory."""

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from backend.demo_clock import DemoClock


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CurrentState:
    """One complete description of the current world context."""

    session_id: int
    observed_at_utc: datetime
    simulated_elapsed_s: float
    clock_speed: float

    person_at_desk: bool | None = None
    owner_at_desk: bool | None = None
    unknown_person_present: bool = False

    face_count: int = 0
    identity: str = "Unknown"
    expression: str = "Unknown"

    away_seconds: float = 0.0
    inactive_seconds: float = 0.0
    is_inactive: bool = False

    user_name: str = ""
    goal: str = ""
    automatic_nudges: bool = True

    # Future calendar, posture, speech and environmental values
    # can be placed here without immediately changing the database.
    extra_context: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["observed_at_utc"] = (
            self.observed_at_utc.isoformat()
        )

        result["simulated_elapsed_s"] = round(
            self.simulated_elapsed_s,
            3,
        )

        result["away_seconds"] = round(
            self.away_seconds,
            3,
        )

        result["inactive_seconds"] = round(
            self.inactive_seconds,
            3,
        )

        return result


@dataclass(frozen=True)
class StateSnapshot:
    """An immutable historical copy of CurrentState."""

    sequence: int
    trigger: str
    changed_fields: tuple[str, ...]
    state: CurrentState

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "trigger": self.trigger,
            "changed_fields": list(
                self.changed_fields
            ),
            "state": self.state.to_dict(),
        }


class ShortTermStateMemory:
    """Bounded in-memory history for future LLM context."""

    def __init__(
        self,
        maximum_snapshots: int = 100,
    ) -> None:
        self._snapshots: deque[StateSnapshot] = deque(
            maxlen=maximum_snapshots
        )

        self._next_sequence = 1

    def append(
        self,
        state: CurrentState,
        trigger: str,
        changed_fields: tuple[str, ...],
    ) -> StateSnapshot:
        snapshot = StateSnapshot(
            sequence=self._next_sequence,
            trigger=trigger,
            changed_fields=changed_fields,
            state=state,
        )

        self._next_sequence += 1
        self._snapshots.append(snapshot)

        return snapshot

    def recent(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        snapshots = list(self._snapshots)[
            -max(0, limit):
        ]

        return [
            snapshot.to_dict()
            for snapshot in snapshots
        ]


class CurrentStateManager(QObject):
    """Maintain current state and record meaningful transitions."""

    state_changed = pyqtSignal(dict)
    snapshot_created = pyqtSignal(dict)

    def __init__(
        self,
        clock: DemoClock,
        session_id: int,
        user_name: str = "",
        goal: str = "",
        automatic_nudges: bool = True,
        inactivity_threshold_s: float = 300.0,
        memory_size: int = 100,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.clock = clock
        self.memory = ShortTermStateMemory(
            memory_size
        )

        self.inactivity_threshold_s = (
            inactivity_threshold_s
        )

        elapsed = clock.elapsed_seconds()

        self.current = CurrentState(
            session_id=session_id,
            observed_at_utc=utc_now(),
            simulated_elapsed_s=elapsed,
            clock_speed=clock.speed,
            user_name=user_name,
            goal=goal,
            automatic_nudges=automatic_nudges,
        )

        self._last_activity_at = elapsed
        self._owner_away_started_at: (
            float | None
        ) = None

        # Used to prevent one bad vision frame from changing state.
        self._candidates: dict[
            str,
            tuple[Any, int],
        ] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    @pyqtSlot()
    def start(self) -> None:
        self._timer.start()

        self.capture_snapshot(
            "session_started",
            ("session_id",),
        )

    @pyqtSlot()
    def stop(self) -> None:
        self._timer.stop()

    def _debounced(
        self,
        field_name: str,
        value: Any,
        samples: int,
    ) -> Any:
        """Accept a changed value after repeated observations."""

        current_value = getattr(
            self.current,
            field_name,
        )

        if value == current_value:
            self._candidates.pop(
                field_name,
                None,
            )

            return current_value

        candidate, count = self._candidates.get(
            field_name,
            (None, 0),
        )

        if candidate == value:
            count += 1
        else:
            candidate = value
            count = 1

        self._candidates[field_name] = (
            candidate,
            count,
        )

        # An uninitialised Boolean can accept its first value
        # immediately. Later changes are debounced.
        required = (
            1
            if current_value is None
            else samples
        )

        if count >= required:
            self._candidates.pop(
                field_name,
                None,
            )

            return value

        return current_value

    def _refresh_durations(
        self,
        state: CurrentState,
    ) -> CurrentState:
        elapsed = self.clock.elapsed_seconds()

        away_seconds = 0.0

        if (
            state.owner_at_desk is False
            and self._owner_away_started_at is not None
        ):
            away_seconds = max(
                0.0,
                elapsed - self._owner_away_started_at,
            )

        inactive_seconds = max(
            0.0,
            elapsed - self._last_activity_at,
        )

        return replace(
            state,
            observed_at_utc=utc_now(),
            simulated_elapsed_s=elapsed,
            clock_speed=self.clock.speed,
            away_seconds=away_seconds,
            inactive_seconds=inactive_seconds,
            is_inactive=(
                inactive_seconds
                >= self.inactivity_threshold_s
            ),
        )

    def _commit(
        self,
        trigger: str,
        **updates: Any,
    ) -> None:
        previous = self.current
        proposed = replace(
            previous,
            **updates,
        )

        if (
            proposed.owner_at_desk
            != previous.owner_at_desk
        ):
            if proposed.owner_at_desk is False:
                self._owner_away_started_at = (
                    self.clock.elapsed_seconds()
                )
            else:
                self._owner_away_started_at = None

        proposed = self._refresh_durations(
            proposed
        )

        changed_fields = tuple(
            field_name
            for field_name in updates
            if getattr(previous, field_name)
            != getattr(proposed, field_name)
        )

        self.current = proposed
        self.state_changed.emit(
            proposed.to_dict()
        )

        if changed_fields:
            self.capture_snapshot(
                trigger,
                changed_fields,
            )

    @pyqtSlot(dict)
    def update_vision_result(
        self,
        result: dict,
    ) -> None:
        """Convert the vision result into contextual state."""

        faces = result.get("faces", [])
        face_count = int(
            result.get("face_count", 0)
        )

        owner_enrolled = bool(
            result.get(
                "owner_enrolled",
                False,
            )
        )

        person_at_desk = face_count > 0

        if owner_enrolled:
            owner_at_desk: bool | None = any(
                face.get("identity") == "Owner"
                for face in faces
            )
        else:
            owner_at_desk = None

        unknown_present = any(
            face.get("identity") == "Unknown"
            for face in faces
        )

        updates = {
            "person_at_desk": self._debounced(
                "person_at_desk",
                person_at_desk,
                2,
            ),
            "owner_at_desk": self._debounced(
                "owner_at_desk",
                owner_at_desk,
                2,
            ),
            "unknown_person_present": self._debounced(
                "unknown_person_present",
                unknown_present,
                2,
            ),
            "face_count": self._debounced(
                "face_count",
                face_count,
                2,
            ),
            "identity": self._debounced(
                "identity",
                str(
                    result.get(
                        "identity",
                        "Unknown",
                    )
                ),
                2,
            ),
            "expression": self._debounced(
                "expression",
                str(
                    result.get(
                        "expression",
                        "Unknown",
                    )
                ),
                3,
            ),
        }

        self._commit(
            "vision_changed",
            **updates,
        )

    @pyqtSlot()
    def mark_vision_unavailable(self) -> None:
        """Clear vision state when the camera is stopped."""

        self._candidates.clear()
        self._owner_away_started_at = None

        # We use None instead of False because a disabled camera
        # cannot tell whether the owner is away.
        self._commit(
            "vision_unavailable",
            person_at_desk=None,
            owner_at_desk=None,
            unknown_person_present=False,
            face_count=0,
            identity="Unknown",
            expression="Unknown",
        )

    @pyqtSlot(str)
    def record_user_activity(
        self,
        source: str = "mouse",
    ) -> None:
        """Reset inactivity when an activity service reports input."""

        previous = self._refresh_durations(
            self.current
        )

        was_inactive = previous.is_inactive
        self._last_activity_at = (
            self.clock.elapsed_seconds()
        )

        self.current = self._refresh_durations(
            previous
        )

        self.state_changed.emit(
            self.current.to_dict()
        )

        # Ordinary mouse movements should not create thousands
        # of database rows. Store only inactive -> active.
        if was_inactive:
            self.capture_snapshot(
                "user_active",
                (
                    "is_inactive",
                    "inactive_seconds",
                ),
            )

    @pyqtSlot(str, float)
    def set_clock_speed(
        self,
        label: str,
        speed: float,
    ) -> None:
        self.clock.set_speed(speed)

        self._commit(
            "clock_speed_changed",
            clock_speed=speed,
        )

    @pyqtSlot(str, str, bool)
    def update_profile(
        self,
        user_name: str,
        goal: str,
        automatic_nudges: bool,
    ) -> None:
        self._commit(
            "profile_changed",
            user_name=user_name,
            goal=goal,
            automatic_nudges=automatic_nudges,
        )

    @pyqtSlot()
    def _tick(self) -> None:
        previous_inactive = (
            self.current.is_inactive
        )

        self.current = self._refresh_durations(
            self.current
        )

        self.state_changed.emit(
            self.current.to_dict()
        )

        if (
            self.current.is_inactive
            != previous_inactive
        ):
            self.capture_snapshot(
                "inactivity_changed",
                (
                    "is_inactive",
                    "inactive_seconds",
                ),
            )

    def capture_snapshot(
        self,
        trigger: str,
        changed_fields: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        self.current = self._refresh_durations(
            self.current
        )

        snapshot = self.memory.append(
            self.current,
            trigger,
            changed_fields,
        )

        payload = snapshot.to_dict()
        self.snapshot_created.emit(payload)

        return payload

    def context_payload(
        self,
        history_limit: int = 10,
    ) -> dict[str, Any]:
        """Return future LLM contextual input."""

        return {
            "current_state": (
                self._refresh_durations(
                    self.current
                ).to_dict()
            ),
            "recent_state_transitions": (
                self.memory.recent(
                    history_limit
                )
            ),
        }