"""Build compact inputs for Kiba's two LLM personas."""

from __future__ import annotations

from typing import Any


def _duration(seconds: Any) -> dict[str, float]:
    value = max(0.0, float(seconds or 0.0))
    return {
        "seconds": round(value, 1),
        "minutes": round(value / 60.0, 1),
    }


def _event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None

    return {
        "id": event.get("id"),
        "title": event.get("summary", "Untitled event"),
        "location": event.get("location") or None,
        "status": event.get("status"),
        "time_text": event.get("time_text"),
        "start_utc": event.get("start_utc"),
        "end_utc": event.get("end_utc"),
        "all_day": bool(event.get("all_day", False)),
        "time_until_start": _duration(event.get("seconds_until_start")),
        "time_until_end": _duration(event.get("seconds_until_end")),
    }


def _period(period: dict[str, Any] | None) -> dict[str, Any] | None:
    if not period:
        return None

    return {
        "started_at_simulated_utc": period.get("started_at_simulated_utc"),
        "ended_at_simulated_utc": period.get("ended_at_simulated_utc"),
        "duration": _duration(period.get("duration_s")),
    }


def _periods(
    periods: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    selected = [period for period in periods[-limit:] if period]
    first_number = max(1, len(periods) - len(selected) + 1)

    return [
        {
            "session_number": first_number + index,
            **(_period(period) or {}),
        }
        for index, period in enumerate(selected)
    ]


def build_conversation_payload(
    user_message: str,
    source: str,
    state: dict[str, Any],
    metrics: dict[str, Any],
    conversation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build context for a user-initiated turn."""

    history = [
        {
            "role": row["role"],
            "content": row["content"],
        }
        for row in conversation_history
        if row.get("role") in {"user", "assistant"}
        and row.get("content")
    ]

    return {
        "interaction_type": "user_message",
        "input_source": source,
        "user_message": user_message.strip(),
        "user": {
            "name": state.get("user_name"),
            "goal": state.get("goal"),
        },
        "conversation_history": history,
        "current_context": {
            "direct_contact_now": True,
            "person_at_desk": state.get("person_at_desk"),
            "person_is_owner": state.get("owner_at_desk"),
            "emotion_estimate": {
                "label": state.get("expression", "Unknown"),
                "confidence": state.get("expression_confidence", 0.0),
            },
            "current_time_away_from_desk": _duration(0.0),
            "current_computer_input_idle_time": _duration(
                metrics.get("current_input_idle_s")
            ),
            "last_break": _period(metrics.get("last_break")),
            "number_of_times_left_desk": int(
                metrics.get("times_left_desk", 0)
            ),
            "number_of_completed_breaks": int(
                metrics.get("break_count", 0)
            ),
            "total_time_away_from_desk": _duration(
                metrics.get("total_away_s")
            ),
            "event_coming_up": _event(
                state.get("next_calendar_event")
            ),
        },
    }


def build_contextual_payload(
    trigger_name: str,
    state: dict[str, Any],
    metrics: dict[str, Any],
    trigger_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build only the context relevant to one trigger."""

    details = trigger_details or {}

    payload: dict[str, Any] = {
        "interaction_type": "contextual_trigger",
        "trigger_name": trigger_name,
        "user": {
            "name": state.get("user_name"),
            "goal": state.get("goal"),
        },
    }

    if trigger_name == "calendar_event_soon":
        payload["trigger_context"] = {
            "event": _event(
                details.get("event")
                or state.get("next_calendar_event")
            ),
            "owner_at_desk": state.get("owner_at_desk"),
        }

    elif trigger_name == "computer_input_inactive":
        payload["trigger_context"] = {
            "meaning": (
                "No keyboard or mouse input was detected; "
                "this is not evidence of no physical movement."
            ),
            "owner_at_desk": state.get("owner_at_desk"),
            "current_inactivity_period": _period(
                metrics.get("current_inactivity_period")
            ),
            "current_input_idle_time": _duration(
                metrics.get("current_input_idle_s")
            ),
            "number_of_inactivity_sessions": int(
                metrics.get("inactivity_session_count", 0)
            ),
            "total_computer_input_inactive_time": _duration(
                metrics.get("total_qualified_inactive_s")
            ),
            "recent_completed_inactivity_sessions": _periods(
                metrics.get("recent_inactivity_periods", [])
            ),
        }

    elif trigger_name == "active_at_desk_too_long":
        payload["trigger_context"] = {
            "owner_at_desk": state.get("owner_at_desk"),
            "continuous_time_at_desk": _duration(
                metrics.get("continuous_owner_at_desk_s")
            ),
            "number_of_times_left_desk": int(
                metrics.get("times_left_desk", 0)
            ),
            "number_of_completed_breaks": int(
                metrics.get("break_count", 0)
            ),
            "total_time_away_from_desk": _duration(
                metrics.get("total_away_s")
            ),
            "last_break": _period(metrics.get("last_break")),
            "number_of_inactivity_sessions": int(
                metrics.get("inactivity_session_count", 0)
            ),
            "total_computer_input_inactive_time": _duration(
                metrics.get("total_qualified_inactive_s")
            ),
            "recent_completed_inactivity_sessions": _periods(
                metrics.get("recent_inactivity_periods", [])
            ),
        }

    elif trigger_name == "negative_emotion_sustained":
        current_emotion = (
            metrics.get("current_negative_emotion") or {}
        )

        payload["trigger_context"] = {
            "meaning": (
                "This is an uncertain facial-expression "
                "estimate, not a fact about the user's feelings."
            ),
            "owner_at_desk": state.get("owner_at_desk"),
            "expression_estimate": current_emotion.get(
                "expression",
                state.get("expression", "Unknown"),
            ),
            "confidence": current_emotion.get(
                "confidence",
                state.get("expression_confidence", 0.0),
            ),
            "continuous_duration": _duration(
                metrics.get("negative_emotion_duration_s")
            ),
            "current_computer_input_idle_time": _duration(
                metrics.get("current_input_idle_s")
            ),
            "last_break": _period(metrics.get("last_break")),
        }

    else:
        payload["trigger_context"] = dict(details)

    return payload