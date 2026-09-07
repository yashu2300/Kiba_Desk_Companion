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
            user = session.scalar(
                select(UserEntity)
                .order_by(UserEntity.id)
                .limit(1)
            )

            if user is None:
                user = UserEntity(
                    display_name="",
                    automatic_nudges=True,
                )

                session.add(user)
                session.flush()

            goal = session.scalar(
                select(GoalEntity)
                .where(
                    GoalEntity.user_id == user.id,
                    GoalEntity.is_active.is_(True),
                )
                .order_by(
                    GoalEntity.created_at.desc()
                )
                .limit(1)
            )

            return {
                "user_id": user.id,
                "display_name": user.display_name,
                "goal": (
                    goal.goal_text
                    if goal
                    else ""
                ),
                "automatic_nudges": (
                    user.automatic_nudges
                ),
            }

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
    ) -> list[dict[str, Any]]:
        with self._sessions() as session:
            rows = session.scalars(
                select(MessageEntity)
                .where(
                    MessageEntity.session_id
                    == session_id
                )
                .order_by(
                    MessageEntity.created_at.desc()
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

    def close(self) -> None:
        self.engine.dispose()