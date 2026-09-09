"""Database entities for SQLite or Neon PostgreSQL."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UserEntity(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    display_name: Mapped[str] = mapped_column(
        String(120),
        default="",
    )

    automatic_nudges: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class GoalEntity(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    goal_text: Mapped[str] = mapped_column(
        Text
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class AppSessionEntity(Base):
    __tablename__ = "app_sessions"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    initial_clock_speed: Mapped[float] = mapped_column(
        Float,
        default=1.0,
    )


class MessageEntity(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("app_sessions.id"),
        index=True,
    )

    # User and assistant messages in the same interaction
    # will share a UUID here.
    turn_id: Mapped[str | None] = mapped_column(
        String(36),
        index=True,
    )

    # user, assistant, event or system
    role: Mapped[str] = mapped_column(
        String(20),
        index=True,
    )

    # text, speech or contextual_event
    source: Mapped[str] = mapped_column(
        String(30),
        default="text",
    )

    content: Mapped[str] = mapped_column(
        Text
    )

    # Reserved for Petoi commands or operating modes.
    action_data: Mapped[
        dict[str, Any] | None
    ] = mapped_column(JSON)

    # Exact context supplied to the LLM for this turn.
    context_data: Mapped[
        dict[str, Any] | None
    ] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )


class StateSnapshotEntity(Base):
    __tablename__ = "state_snapshots"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("app_sessions.id"),
        index=True,
    )

    sequence: Mapped[int]

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )

    simulated_elapsed_s: Mapped[float] = mapped_column(
        Float
    )

    trigger: Mapped[str] = mapped_column(
        String(60),
        index=True,
    )

    changed_fields: Mapped[list[str]] = mapped_column(
        JSON
    )

    # Important queryable state values.
    person_at_desk: Mapped[bool | None] = mapped_column(
        Boolean
    )

    owner_at_desk: Mapped[bool | None] = mapped_column(
        Boolean
    )

    unknown_person_present: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    face_count: Mapped[int] = mapped_column(
        default=0
    )

    identity: Mapped[str] = mapped_column(
        String(120),
        default="Unknown",
    )

    expression: Mapped[str] = mapped_column(
        String(40),
        default="Unknown",
    )

    away_seconds: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    inactive_seconds: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    is_inactive: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    # Complete extensible state, including future services.
    state_data: Mapped[dict[str, Any]] = mapped_column(
        JSON
    )


class SessionPeriodEntity(Base):
    """One completed period measured using the simulated clock."""

    __tablename__ = "session_periods"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("app_sessions.id"), index=True)
    period_type: Mapped[str] = mapped_column(String(40), index=True)
    started_at_simulated: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at_simulated: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_elapsed_s: Mapped[float] = mapped_column(Float)
    ended_elapsed_s: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float)
    period_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Database:
    def __init__(
        self,
        database_url: str | None = None,
    ) -> None:
        project_root = (
            Path(__file__).resolve().parents[1]
        )

        load_dotenv(
            project_root / ".env"
        )

        url = (
            database_url
            or os.getenv("DATABASE_URL", "").strip()
        )

        if not url:
            database_path = (
                project_root
                / "data"
                / "kiba.db"
            )

            database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            url = (
                "sqlite:///"
                f"{database_path.as_posix()}"
            )

        # Neon commonly supplies postgresql://.
        # Select Psycopg 3 explicitly.
        if url.startswith("postgresql://"):
            url = url.replace(
                "postgresql://",
                "postgresql+psycopg://",
                1,
            )

        connect_args = (
            {"check_same_thread": False}
            if url.startswith("sqlite")
            else {}
        )

        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        self._sessions = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    def create_tables(self) -> None:
        Base.metadata.create_all(
            self.engine
        )

    def load_or_create_profile(
        self,
    ) -> dict[str, Any]:
        with self._sessions.begin() as session:
            # Load the profile used most recently.
            user = session.scalar(
                select(UserEntity)
                .join(
                    AppSessionEntity,
                    AppSessionEntity.user_id
                    == UserEntity.id,
                )
                .order_by(
                    AppSessionEntity.started_at.desc(),
                    AppSessionEntity.id.desc(),
                )
                .limit(1)
            )

            # A user may exist before any session exists.
            if user is None:
                user = session.scalar(
                    select(UserEntity)
                    .order_by(UserEntity.id)
                    .limit(1)
                )

            # Create the first empty user when the
            # database has never been used.
            if user is None:
                user = UserEntity(
                    display_name="",
                    automatic_nudges=True,
                )

                session.add(user)
                session.flush()

            return self._profile_payload(
                session,
                user,
            )

    @staticmethod
    def _profile_payload(
        session,
        user: UserEntity,
    ) -> dict[str, Any]:
        """Return one user and their currently active goal."""

        goal = session.scalar(
            select(GoalEntity)
            .where(
                GoalEntity.user_id == user.id,
                GoalEntity.is_active.is_(True),
            )
            .order_by(
                GoalEntity.created_at.desc(),
                GoalEntity.id.desc(),
            )
            .limit(1)
        )

        return {
            "user_id": user.id,
            "display_name": user.display_name,
            "goal": (
                goal.goal_text
                if goal is not None
                else ""
            ),
            "automatic_nudges": (
                user.automatic_nudges
            ),
        }

    def save_or_switch_profile(
        self,
        current_user_id: int,
        display_name: str,
        goal_text: str,
        automatic_nudges: bool,
    ) -> dict[str, Any]:
        """
        Update the current profile or switch to another one.

        Profile-name matching is case-insensitive. Therefore,
        'Demo' and 'demo' refer to the same database user.

        When switching to an existing user, that user's saved
        goal and nudge preference are loaded rather than being
        overwritten with the previous user's settings.
        """

        cleaned_name = display_name.strip()

        if not cleaned_name:
            raise ValueError(
                "A profile name is required."
            )

        with self._sessions.begin() as session:
            current_user = session.get(
                UserEntity,
                current_user_id,
            )

            if current_user is None:
                raise ValueError(
                    f"User {current_user_id} does not exist."
                )

            target_user = current_user
            created = False
            switching_to_existing = False

            current_name = (
                current_user.display_name.strip()
            )

            # A different name means create or switch profiles.
            if (
                current_name
                and current_name.casefold()
                != cleaned_name.casefold()
            ):
                users = session.scalars(
                    select(UserEntity)
                    .order_by(UserEntity.id)
                ).all()

                target_user = next(
                    (
                        user
                        for user in users
                        if (
                            user.display_name
                            .strip()
                            .casefold()
                            == cleaned_name.casefold()
                        )
                    ),
                    None,
                )

                if target_user is None:
                    # The name does not exist, so create
                    # a separate user.
                    target_user = UserEntity(
                        display_name=cleaned_name,
                        automatic_nudges=(
                            automatic_nudges
                        ),
                    )

                    session.add(target_user)
                    session.flush()

                    created = True
                else:
                    # Load this user's saved settings instead
                    # of copying the previous user's settings.
                    switching_to_existing = True

            if not switching_to_existing:
                target_user.display_name = (
                    cleaned_name
                )

                target_user.automatic_nudges = (
                    automatic_nudges
                )

            active_goal = session.scalar(
                select(GoalEntity)
                .where(
                    GoalEntity.user_id
                    == target_user.id,
                    GoalEntity.is_active.is_(True),
                )
                .order_by(
                    GoalEntity.created_at.desc(),
                    GoalEntity.id.desc(),
                )
                .limit(1)
            )

            cleaned_goal = goal_text.strip()

            # Only update the goal when updating the current
            # profile or creating a completely new one.
            if (
                not switching_to_existing
                and (
                    active_goal is None
                    or active_goal.goal_text
                    != cleaned_goal
                )
            ):
                session.execute(
                    update(GoalEntity)
                    .where(
                        GoalEntity.user_id
                        == target_user.id,
                        GoalEntity.is_active.is_(
                            True
                        ),
                    )
                    .values(
                        is_active=False,
                        ended_at=utc_now(),
                    )
                )

                session.add(
                    GoalEntity(
                        user_id=target_user.id,
                        goal_text=cleaned_goal,
                    )
                )

            session.flush()

            result = self._profile_payload(
                session,
                target_user,
            )

            result.update(
                {
                    "created": created,
                    "switched": (
                        target_user.id
                        != current_user_id
                    ),
                }
            )

            return result

    def save_profile(
        self,
        user_id: int,
        display_name: str,
        goal_text: str,
        automatic_nudges: bool,
    ) -> None:
        with self._sessions.begin() as session:
            user = session.get(
                UserEntity,
                user_id,
            )

            if user is None:
                raise ValueError(
                    f"User {user_id} does not exist."
                )

            user.display_name = (
                display_name.strip()
            )

            user.automatic_nudges = (
                automatic_nudges
            )

            active_goal = session.scalar(
                select(GoalEntity)
                .where(
                    GoalEntity.user_id == user_id,
                    GoalEntity.is_active.is_(True),
                )
                .order_by(
                    GoalEntity.created_at.desc()
                )
                .limit(1)
            )

            cleaned_goal = goal_text.strip()

            if (
                active_goal is None
                or active_goal.goal_text
                != cleaned_goal
            ):
                session.execute(
                    update(GoalEntity)
                    .where(
                        GoalEntity.user_id
                        == user_id,
                        GoalEntity.is_active.is_(
                            True
                        ),
                    )
                    .values(
                        is_active=False,
                        ended_at=utc_now(),
                    )
                )

                session.add(
                    GoalEntity(
                        user_id=user_id,
                        goal_text=cleaned_goal,
                    )
                )

    def start_app_session(
        self,
        user_id: int,
        clock_speed: float,
    ) -> int:
        with self._sessions.begin() as session:
            app_session = AppSessionEntity(
                user_id=user_id,
                initial_clock_speed=clock_speed,
            )

            session.add(app_session)
            session.flush()

            return app_session.id

    def finish_app_session(
        self,
        session_id: int,
    ) -> None:
        with self._sessions.begin() as session:
            app_session = session.get(
                AppSessionEntity,
                session_id,
            )

            if (
                app_session is not None
                and app_session.ended_at is None
            ):
                app_session.ended_at = utc_now()

    def save_message(
        self,
        session_id: int,
        role: str,
        content: str,
        source: str = "text",
        turn_id: str | None = None,
        action_data: (
            dict[str, Any] | None
        ) = None,
        context_data: (
            dict[str, Any] | None
        ) = None,
    ) -> int:
        with self._sessions.begin() as session:
            message = MessageEntity(
                session_id=session_id,
                turn_id=turn_id,
                role=role,
                source=source,
                content=content,
                action_data=action_data,
                context_data=context_data,
            )

            session.add(message)
            session.flush()

            return message.id

    def save_state_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> int:
        state = snapshot["state"]

        observed_at = datetime.fromisoformat(
            state["observed_at_utc"]
        )

        with self._sessions.begin() as session:
            entity = StateSnapshotEntity(
                session_id=int(
                    state["session_id"]
                ),
                sequence=int(
                    snapshot["sequence"]
                ),
                observed_at=observed_at,
                simulated_elapsed_s=float(
                    state["simulated_elapsed_s"]
                ),
                trigger=str(
                    snapshot["trigger"]
                ),
                changed_fields=list(
                    snapshot["changed_fields"]
                ),
                person_at_desk=state[
                    "person_at_desk"
                ],
                owner_at_desk=state[
                    "owner_at_desk"
                ],
                unknown_person_present=bool(
                    state[
                        "unknown_person_present"
                    ]
                ),
                face_count=int(
                    state["face_count"]
                ),
                identity=str(
                    state["identity"]
                ),
                expression=str(
                    state["expression"]
                ),
                away_seconds=float(
                    state["away_seconds"]
                ),
                inactive_seconds=float(
                    state["inactive_seconds"]
                ),
                is_inactive=bool(
                    state["is_inactive"]
                ),
                state_data=state,
            )

            session.add(entity)
            session.flush()

            return entity.id

    def recent_messages(
        self,
        session_id: int,
        limit: int = 20,
        after_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._sessions() as session:
            conditions = [
                MessageEntity.session_id
                == session_id
            ]

            if after_message_id is not None:
                conditions.append(
                    MessageEntity.id
                    > after_message_id
                )

            rows = session.scalars(
                select(MessageEntity)
                .where(*conditions)
                .order_by(
                    MessageEntity.created_at.desc(),
                    MessageEntity.id.desc(),
                )
                .limit(limit)
            ).all()

            return [
                {
                    "id": row.id,
                    "turn_id": row.turn_id,
                    "role": row.role,
                    "source": row.source,
                    "content": row.content,
                    "action_data": row.action_data,
                    "created_at": (
                        row.created_at.isoformat()
                    ),
                }
                for row in reversed(rows)
            ]

    def save_session_period(self, period: dict[str, Any]) -> int:
        """Store one completed simulated-time period."""

        with self._sessions.begin() as session:
            entity = SessionPeriodEntity(
                session_id=int(period["session_id"]),
                period_type=str(period["period_type"]),
                started_at_simulated=datetime.fromisoformat(period["started_at_simulated_utc"]),
                ended_at_simulated=datetime.fromisoformat(period["ended_at_simulated_utc"]),
                started_elapsed_s=float(period["started_elapsed_s"]),
                ended_elapsed_s=float(period["ended_elapsed_s"]),
                duration_s=float(period["duration_s"]),
                period_data=dict(period.get("details", {})),
            )

            session.add(entity)
            session.flush()
            return entity.id

    def latest_message_id(
        self,
        session_id: int,
    ) -> int | None:
        """Return the newest stored message in this session."""

        with self._sessions() as session:
            return session.scalar(
                select(MessageEntity.id)
                .where(
                    MessageEntity.session_id
                    == session_id
                )
                .order_by(
                    MessageEntity.id.desc()
                )
                .limit(1)
            )

    def close(self) -> None:
        self.engine.dispose()
