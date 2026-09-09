"""Qt slot controller connecting frontend signals to backend services."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from PyQt5.QtCore import QObject, pyqtSlot


# Import all services such as demo clock and etc. 
from backend.services.webcam_service import WebcamService
from backend.services.face_recognition_service import FaceRecognitionService
from backend.services.activity_monitor_service import ActivityMonitorService
from backend.services.google_calendar_service import GoogleCalendarService
from backend.services.llm_service import LLMService

from backend.state_manager import CurrentStateManager
from backend.database import Database
from backend.demo_clock import DemoClock

from frontend.main_window import DeskCompanionWindow


class SlotController(QObject):
    def __init__(
        self,
        webcam: WebcamService,
        face_recognition: FaceRecognitionService,
        activity_monitor: ActivityMonitorService,
        calendar: GoogleCalendarService,
        llm: LLMService,
        clock: DemoClock,
        state_manager: CurrentStateManager,
        database: Database,
        user_id: int,
        session_id: int,
        parent: QObject | None = None) -> None:

        super().__init__(parent)
        self.webcam = webcam
        self.face_recognition = face_recognition
        self.activity_monitor = activity_monitor
        self.calendar=calendar
        self.llm = llm
        self.clock = clock
        self.state_manager = state_manager
        self.database = database
        self.user_id = user_id
        self.session_id = session_id
        self.window : DeskCompanionWindow | None = None
        self._context_message_floor_id : int | None = None


    def connect_window_to_slots(self, window:DeskCompanionWindow):
        """ Wire every frontend/pyqtSignals and service result signals in ONE PLACE"""
        self.window = window

        # Connect Main Window Signals
        window.camera_toggled.connect(self.on_camera_toggled)
        window.face_enrollment_requested.connect(self.on_face_enrollment_requested)
        window.demo_clock_changed.connect(self.on_demo_clock_changed)
        window.goal_saved.connect(self.on_goal_saved)
        window.user_activity_detected.connect(self.state_manager.record_user_activity)
        window.calendar_refresh_requested.connect(self.calendar.refresh)
        window.message_send_requested.connect(self.on_message_send_requested)
        window.reset_conversation_requested.connect(self.on_conversation_reset_requested)

        # Connect WebcamService Signals
        self.webcam.frame_ready.connect(window.set_camera_frame)
        self.webcam.camera_opened.connect(window.show_camera_opened)
        self.webcam.error.connect(window.show_camera_error)
        self.webcam.error.connect(self._on_webcam_error)
        self.webcam.stopped.connect(window.show_camera_stopped)
        self.webcam.stopped.connect(self._on_webcam_stopped)

        # Connect Face Recongition Signals
        self.face_recognition.result_ready.connect(window.show_vision_result)
        self.face_recognition.enrollment_busy_changed.connect(window.set_face_enrollment_busy)
        self.face_recognition.enrollment_finished.connect(window.show_face_enrollment_result)
        self.face_recognition.error.connect(window.show_vision_error)
        self.face_recognition.error.connect(self._on_vision_error)
        self.face_recognition.initialized.connect(self._on_vision_initialized)

        # Activity Monitors
        self.activity_monitor.started.connect(self._on_activity_monitor_started)
        self.activity_monitor.stopped.connect(self._on_activity_monitor_stopped)
        self.activity_monitor.error.connect(self._on_activity_monitor_error)

        # State Related
        self.state_manager.state_changed.connect(window.show_current_state)
        self.state_manager.snapshot_created.connect(self._persist_state_snapshot)

        # Calendar Related
        self.calendar.calendar_updated.connect(window.show_calendar)
        self.calendar.busy_changed.connect(window.set_calendar_busy)
        self.calendar.error.connect(window.show_calendar_error)
        self.calendar.error.connect(self._on_calendar_error)

        # LLM Related
        self.llm.response_ready.connect(window.show_llm_response)
        self.llm.response_ready.connect(self._on_llm_response)
        self.llm.busy_changed.connect(window.set_llm_busy)
        self.llm.error.connect(window.show_llm_error)
        self.llm.error.connect(self._on_llm_error)


        # Interconnected Serivce Signals
        self.webcam.analysis_frame_ready.connect(self.face_recognition.submit_frame)
        self.face_recognition.result_ready.connect(self.state_manager.update_vision_result)
        self.webcam.stopped.connect(self.state_manager.mark_vision_unavailable)
        self.activity_monitor.activity_detected.connect(self.state_manager.record_user_activity)
        self.calendar.calendar_updated.connect(self.state_manager.update_calendar_result)

    # Webcam Services
    @pyqtSlot(bool)
    def on_camera_toggled(self, camera_active: bool) -> None:
        print(f"[SLOT] on_camera_toggled(camera_active={camera_active})")
        if camera_active:
            self.webcam.start()
        else:
            self.webcam.stop()

    @pyqtSlot(str)
    def _on_webcam_error(self, message: str) -> None:
        print(f"[WEBCAM ERROR] {message}")

    @pyqtSlot()
    def _on_webcam_stopped(self) -> None:
        print("[WEBCAM] Capture stopped and the camera was released.")

    # Vision Face Recognition Service
    @pyqtSlot()
    def on_face_enrollment_requested(self) -> None:
        print("[SLOT] on_face_enrollment_requested()")
        self.face_recognition.enroll_owner()

    @pyqtSlot(bool)
    def _on_vision_initialized(
        self,
        owner_enrolled: bool,
    ) -> None:
        enrolled = "yes" if owner_enrolled else "no"

        print(
            "[VISION] Models ready. "
            f"Owner enrolled: {enrolled}."
        )

    @pyqtSlot(str)
    def _on_vision_error(
        self,
        message: str,
    ) -> None:
        print(f"[VISION ERROR] {message}")

    # Activity Monitor Services
    @pyqtSlot()
    def _on_activity_monitor_started(self) -> None:
        print("[ACTIVITY] Global mouse and keyboard monitoring started.")

    @pyqtSlot()
    def _on_activity_monitor_stopped(self) -> None:
        print("[ACTIVITY] Global input monitoring stopped.")


    @pyqtSlot(str)
    def _on_activity_monitor_error(self, message: str) -> None:
        print(f"[ACTIVITY ERROR] {message}")

    # DEMO & STATE RELATED
    @pyqtSlot(str, float)
    def on_demo_clock_changed(self, label: str, speed: float) -> None:
        print(f"[CLOCK] {label} ({speed:g}x)")
        self.state_manager.set_clock_speed(label, speed)

    @pyqtSlot(str, str, bool)
    def on_goal_saved(self, display_name: str, goal: str, automatic_nudges: bool) -> None:
        profile = self.database.save_or_switch_profile(current_user_id=self.user_id, display_name=display_name, goal_text=goal, automatic_nudges=automatic_nudges)

        if profile["switched"]:
            # Record and close the previous user's session.
            self.state_manager.capture_snapshot("session_ended")

            self.database.finish_app_session(self.session_id)

            self.user_id = int( profile["user_id"])

            # Switching users starts a separate session.
            self.session_id = (self.database.start_app_session(self.user_id,self.clock.speed))

            # The new session has no chat-history cutoff.
            self._context_message_floor_id = None
            self.state_manager.switch_profile(session_id=self.session_id, 
                                              user_name=str(profile["display_name"]),
                                              goal=str(profile["goal"]),
                                              automatic_nudges=bool(profile["automatic_nudges"])
            )

        else:
            self.state_manager.update_profile(
                str(profile["display_name"]),
                str(profile["goal"]),
                bool(
                    profile["automatic_nudges"]
                ),
            )

        if self.window is not None:
            self.window.load_profile(
                str(profile["display_name"]),
                str(profile["goal"]),
                bool(
                    profile["automatic_nudges"]
                ),
            )

            if profile["switched"]:
                # Do not display messages belonging to
                # the previous user's session.
                self.window.clear_conversation()

        status = (
            "created"
            if profile["created"]
            else (
                "switched"
                if profile["switched"]
                else "updated"
            )
        )

        print(
            "[DATABASE] User profile "
            f"{status}: "
            f"{profile['display_name']} "
            f"(user_id={self.user_id})."
        )


    @pyqtSlot(dict)
    def _persist_state_snapshot(self, snapshot: dict,) -> None:
        try:
            snapshot_id = (self.database.save_state_snapshot(snapshot))

            print(
                f"[STATE] Snapshot {snapshot_id} saved: "
                f"{snapshot['trigger']} -> "
                f"{snapshot['changed_fields']}"
            )

        except Exception as error:
            print(
                "[DATABASE ERROR] State snapshot "
                f"was not saved: {error}"
            )


    # Calendar Slots
    @pyqtSlot(str)
    def _on_calendar_error(self,message: str) -> None:
        print(f"[CALENDAR ERROR] {message}")

    # LLM SLOTS
    @pyqtSlot()
    def on_conversation_reset_requested(self) -> None:
        """
        Set the LLM history boundary without deleting
        anything from the database.
        """
        self._context_message_floor_id = (self.database.latest_message_id(self.session_id))
        print("[LLM CONTEXT] Conversation history reset; database messages retained.")

    @pyqtSlot(str, str)
    def on_message_send_requested(
        self,
        message: str,
        source: str,
    ) -> None:
        turn_id = str(uuid4())

        context = (
            self.state_manager.context_payload(
                history_limit=8
            )
        )

        context["recent_messages"] = (
            self.database.recent_messages(
                self.session_id,
                limit=8,
                after_message_id=(
                    self._context_message_floor_id
                ),
            )
        )

        self.database.save_message(
            session_id=self.session_id,
            role="user",
            content=message,
            source=source,
            turn_id=turn_id,
            context_data=context,
        )

        self.llm.generate(
            turn_id=turn_id,
            user_message=message,
            source=source,
            trigger="user_message",
            context=context,
        )


    def request_contextual_response(
        self,
        trigger: str,
    ) -> None:
        """
        Entry point for a future automatic
        contextual-event evaluator.
        """

        turn_id = str(uuid4())

        context = (
            self.state_manager.context_payload(
                history_limit=8
            )
        )

        context["recent_messages"] = (
            self.database.recent_messages(
                self.session_id,
                limit=8,
                after_message_id=(
                    self._context_message_floor_id
                ),
            )
        )

        self.database.save_message(
            session_id=self.session_id,
            role="event",
            content=trigger,
            source="contextual_event",
            turn_id=turn_id,
            context_data=context,
        )

        self.llm.generate(
            turn_id=turn_id,
            user_message=None,
            source="contextual_event",
            trigger=trigger,
            context=context,
        )


    @pyqtSlot(dict)
    def _on_llm_response(
        self,
        result: dict,
    ) -> None:
        actions = list(
            result.get("actions", [])
        )

        self.database.save_message(
            session_id=self.session_id,
            role="assistant",
            content=str(
                result.get(
                    "text_response",
                    "",
                )
            ),
            source="llm",
            turn_id=str(
                result.get("turn_id", "")
            ),
            action_data={
                "actions": actions,
                "model": result.get("model"),
            },
        )
        print(
            f"[LLM MODEL] "
            f"{result.get('model', 'Unknown')}"
        )

        print(
            f"[LLM TEXT] "
            f"{result.get('text_response', '')}"
        )

        print(
            f"[LLM ACTIONS] {actions}"
        )

        for action in actions:
            if action != "no_action":
                print(
                    f"[ACTION SIMULATED] "
                    f"{action}"
                )


    @pyqtSlot(str)
    def _on_llm_error(
        self,
        message: str,
    ) -> None:
        print(
            f"[LLM ERROR] {message}"
        )

    @pyqtSlot()
    def shutdown(self) -> None:
        self.webcam.stop()
        self.face_recognition.stop()
        self.activity_monitor.stop()

        self.state_manager.capture_snapshot("session_ended")
        self.state_manager.stop()

        self.calendar.stop()
        self.llm.stop()
        
        self.database.finish_app_session(self.session_id)
        self.database.close()


    