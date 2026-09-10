"""Qt slot controller connecting frontend signals to backend services."""

from __future__ import annotations

from uuid import uuid4

from PyQt5.QtCore import QObject, pyqtSlot

from backend.database import Database
from backend.demo_clock import DemoClock
from backend.state_manager import CurrentStateManager
from backend.services.STT_service import SpeechToTextService
from backend.services.TTS_service import TextToSpeechService
from backend.services.activity_monitor_service import ActivityMonitorService
from backend.services.event_evaluator_service import EventEvaluatorService
from backend.services.face_recognition_service import FaceRecognitionService
from backend.services.google_calendar_service import GoogleCalendarService
from backend.services.guard_mode_service import GuardModeService
from backend.services.llm_context_builders import (
    build_contextual_payload,
    build_conversation_payload,
)
from backend.services.llm_service import (
    ContextualLLMService,
    ConversationLLMService,
)
from backend.services.session_metrics_service import SessionMetricsService
from backend.services.webcam_service import WebcamService
from frontend.main_window import DeskCompanionWindow


class SlotController(QObject):
    def __init__(
        self,
        webcam: WebcamService,
        face_recognition: FaceRecognitionService,
        activity_monitor: ActivityMonitorService,
        session_metrics: SessionMetricsService,
        calendar: GoogleCalendarService,
        conversation_llm: ConversationLLMService,
        contextual_llm: ContextualLLMService,
        tts: TextToSpeechService,
        stt: SpeechToTextService,
        guard_mode: GuardModeService,
        event_evaluator: EventEvaluatorService,
        clock: DemoClock,
        state_manager: CurrentStateManager,
        database: Database,
        user_id: int,
        session_id: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.webcam = webcam
        self.face_recognition = face_recognition
        self.activity_monitor = activity_monitor
        self.session_metrics = session_metrics
        self.calendar = calendar
        self.conversation_llm = conversation_llm
        self.contextual_llm = contextual_llm
        self.tts = tts
        self.stt = stt
        self.guard_mode = guard_mode
        self.event_evaluator = event_evaluator
        self.clock = clock
        self.state_manager = state_manager
        self.database = database

        self.user_id = user_id
        self.session_id = session_id

        self.window: DeskCompanionWindow | None = None
        self._context_message_floor_id: int | None = None

        self._shutting_down = False
        self._handling_user_message = False
        self._conversation_llm_busy = False
        self._contextual_llm_busy = False
        self._tts_busy = False
        self._speech_input_busy = False
        self._guard_mode_active = False

    def connect_window_to_slots(
        self,
        window: DeskCompanionWindow,
    ) -> None:
        """Connect frontend and backend signals in one place."""

        self.window = window

        # Main-window signals.
        window.camera_toggled.connect(self.on_camera_toggled)
        window.face_enrollment_requested.connect(self.on_face_enrollment_requested)
        window.demo_clock_changed.connect(self.on_demo_clock_changed)
        window.goal_saved.connect(self.on_goal_saved)
        window.user_activity_detected.connect(self.state_manager.record_user_activity)
        window.calendar_refresh_requested.connect(self.calendar.refresh)
        window.message_send_requested.connect(self.on_message_send_requested)
        window.reset_conversation_requested.connect(
            self.on_conversation_reset_requested
        )

        # Webcam signals.
        self.webcam.frame_ready.connect(window.set_camera_frame)
        self.webcam.camera_opened.connect(window.show_camera_opened)
        self.webcam.error.connect(window.show_camera_error)
        self.webcam.error.connect(self._on_webcam_error)
        self.webcam.stopped.connect(window.show_camera_stopped)
        self.webcam.stopped.connect(self._on_webcam_stopped)

        # Face-recognition signals.
        self.face_recognition.result_ready.connect(window.show_vision_result)
        self.face_recognition.enrollment_busy_changed.connect(
            window.set_face_enrollment_busy
        )
        self.face_recognition.enrollment_finished.connect(
            window.show_face_enrollment_result
        )
        self.face_recognition.error.connect(window.show_vision_error)
        self.face_recognition.error.connect(self._on_vision_error)
        self.face_recognition.initialized.connect(self._on_vision_initialized)

        # Activity-monitor signals.
        self.activity_monitor.started.connect(self._on_activity_monitor_started)
        self.activity_monitor.stopped.connect(self._on_activity_monitor_stopped)
        self.activity_monitor.error.connect(self._on_activity_monitor_error)

        # State and session-metric signals.
        self.state_manager.state_changed.connect(window.show_current_state)
        self.state_manager.snapshot_created.connect(self._persist_state_snapshot)
        self.state_manager.state_changed.connect(self.session_metrics.observe_state)
        self.session_metrics.period_completed.connect(
            self._persist_session_period
        )

        # Calendar signals.
        self.calendar.calendar_updated.connect(window.show_calendar)
        self.calendar.busy_changed.connect(window.set_calendar_busy)
        self.calendar.error.connect(window.show_calendar_error)
        self.calendar.error.connect(self._on_calendar_error)

        # Conversation LLM signals.
        self.conversation_llm.response_ready.connect(window.show_llm_response)
        self.conversation_llm.response_ready.connect(self._on_llm_response)
        self.conversation_llm.busy_changed.connect(
            self._on_conversation_llm_busy
        )
        self.conversation_llm.error.connect(window.show_llm_error)
        self.conversation_llm.error.connect(self._on_llm_error)

        # Contextual LLM signals.
        self.contextual_llm.response_ready.connect(window.show_llm_response)
        self.contextual_llm.response_ready.connect(self._on_llm_response)
        self.contextual_llm.busy_changed.connect(self._on_contextual_llm_busy)
        self.contextual_llm.error.connect(window.show_llm_error)
        self.contextual_llm.error.connect(self._on_llm_error)

        # Contextual evaluator signals.
        self.session_metrics.metrics_changed.connect(
            self._evaluate_context_triggers
        )
        self.event_evaluator.trigger_ready.connect(
            self.request_contextual_response
        )

        # TTS signals.
        self.tts.ready_changed.connect(self._on_tts_ready_changed)
        self.tts.busy_changed.connect(self._on_tts_busy_changed)
        self.tts.audio_ready.connect(self._on_tts_audio_ready)
        self.tts.playback_finished.connect(self._on_tts_playback_finished)
        self.tts.error.connect(self._on_tts_error)

        # Always-listening STT signals.
        self.stt.ready_changed.connect(self._on_stt_ready_changed)
        self.stt.status_changed.connect(window.show_speech_status)
        self.stt.capture_busy_changed.connect(
            self._on_stt_capture_busy_changed
        )
        self.stt.wake_word_detected.connect(self._on_wake_word_detected)
        self.stt.transcript_ready.connect(self.on_speech_transcript_ready)
        self.stt.error.connect(window.show_speech_error)
        self.stt.error.connect(self._on_stt_error)

        # Guard-mode signals.
        self.guard_mode.mode_changed.connect(self._on_operating_mode_changed)
        self.guard_mode.warning_requested.connect(
            self._on_guard_warning_requested
        )
        self.guard_mode.error.connect(window.show_guard_error)
        self.guard_mode.error.connect(self._on_guard_error)

        self.face_recognition.initialized.connect(
            self.guard_mode.set_owner_enrolled
        )
        self.face_recognition.enrollment_finished.connect(
            self.guard_mode.on_enrollment_finished
        )
        self.face_recognition.result_ready.connect(
            self.guard_mode.observe_vision
        )

        # Inter-service signals.
        self.webcam.analysis_frame_ready.connect(
            self.face_recognition.submit_frame
        )
        self.face_recognition.result_ready.connect(
            self.state_manager.update_vision_result
        )
        self.webcam.stopped.connect(
            self.state_manager.mark_vision_unavailable
        )
        self.activity_monitor.activity_detected.connect(
            self.state_manager.record_user_activity
        )
        self.calendar.calendar_updated.connect(
            self.state_manager.update_calendar_result
        )

    # ------------------------------------------------------------------
    # Webcam
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def on_camera_toggled(
        self,
        camera_active: bool,
    ) -> None:
        if self._guard_mode_active and not camera_active:
            return

        print(f"[SLOT] on_camera_toggled(camera_active={camera_active})")

        if camera_active:
            self.webcam.start()
        else:
            self.webcam.stop()

    @pyqtSlot(str)
    def _on_webcam_error(
        self,
        message: str,
    ) -> None:
        print(f"[WEBCAM ERROR] {message}")

    @pyqtSlot()
    def _on_webcam_stopped(self) -> None:
        print("[WEBCAM] Capture stopped and the camera was released.")

    # ------------------------------------------------------------------
    # Face recognition
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_face_enrollment_requested(self) -> None:
        if self._guard_mode_active or self._shutting_down:
            return

        print("[SLOT] on_face_enrollment_requested()")
        self.face_recognition.enroll_owner()

    @pyqtSlot(bool)
    def _on_vision_initialized(
        self,
        owner_enrolled: bool,
    ) -> None:
        enrolled = "yes" if owner_enrolled else "no"
        print(f"[VISION] Models ready. Owner enrolled: {enrolled}.")

    @pyqtSlot(str)
    def _on_vision_error(
        self,
        message: str,
    ) -> None:
        print(f"[VISION ERROR] {message}")

    # ------------------------------------------------------------------
    # Activity monitor
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_activity_monitor_started(self) -> None:
        print("[ACTIVITY] Global mouse and keyboard monitoring started.")

    @pyqtSlot()
    def _on_activity_monitor_stopped(self) -> None:
        print("[ACTIVITY] Global input monitoring stopped.")

    @pyqtSlot(str)
    def _on_activity_monitor_error(
        self,
        message: str,
    ) -> None:
        print(f"[ACTIVITY ERROR] {message}")

    # ------------------------------------------------------------------
    # Clock, state and user profile
    # ------------------------------------------------------------------

    @pyqtSlot(str, float)
    def on_demo_clock_changed(
        self,
        label: str,
        speed: float,
    ) -> None:
        if self._guard_mode_active or self._shutting_down:
            return

        print(f"[CLOCK] {label} ({speed:g}x)")
        self.state_manager.set_clock_speed(label, speed)

    @pyqtSlot(str, str, bool)
    def on_goal_saved(
        self,
        display_name: str,
        goal: str,
        automatic_nudges: bool,
    ) -> None:
        if self._guard_mode_active or self._shutting_down:
            return

        profile = self.database.save_or_switch_profile(
            current_user_id=self.user_id,
            display_name=display_name,
            goal_text=goal,
            automatic_nudges=automatic_nudges,
        )

        if profile["switched"]:
            # Finish the previous user's session.
            self.state_manager.capture_snapshot("session_ended")
            self.session_metrics.finish_session()
            self.database.finish_app_session(self.session_id)

            self.user_id = int(profile["user_id"])

            # Start a completely separate session for the selected user.
            self.session_id = self.database.start_app_session(
                self.user_id,
                self.clock.speed,
            )

            # The selected user's new session starts with no LLM history.
            self._context_message_floor_id = None

            self.session_metrics.reset_session(
                self.session_id
            )
            self.event_evaluator.reset_session()

            self.state_manager.switch_profile(
                session_id=self.session_id,
                user_name=str(profile["display_name"]),
                goal=str(profile["goal"]),
                automatic_nudges=bool(
                    profile["automatic_nudges"]
                ),
            )

        else:
            self.state_manager.update_profile(
                str(profile["display_name"]),
                str(profile["goal"]),
                bool(profile["automatic_nudges"]),
            )

        if self.window is not None:
            self.window.load_profile(
                str(profile["display_name"]),
                str(profile["goal"]),
                bool(profile["automatic_nudges"]),
            )

            if profile["switched"]:
                self.window.clear_conversation()

        if profile["created"]:
            status = "created"
        elif profile["switched"]:
            status = "switched"
        else:
            status = "updated"

        print(
            f"[DATABASE] User profile {status}: "
            f"{profile['display_name']} "
            f"(user_id={self.user_id})."
        )

    @pyqtSlot(dict)
    def _persist_state_snapshot(
        self,
        snapshot: dict,
    ) -> None:
        try:
            snapshot_id = self.database.save_state_snapshot(
                snapshot
            )

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

    @pyqtSlot(dict)
    def _persist_session_period(
        self,
        period: dict,
    ) -> None:
        try:
            period_id = self.database.save_session_period(
                period
            )

            print(
                f"[METRICS] Period {period_id} saved: "
                f"{period['period_type']} for "
                f"{period['duration_s']:.1f} simulated seconds."
            )

        except Exception as error:
            print(
                "[DATABASE ERROR] Session period "
                f"was not saved: {error}"
            )

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_calendar_error(
        self,
        message: str,
    ) -> None:
        print(f"[CALENDAR ERROR] {message}")

    # ------------------------------------------------------------------
    # Conversation reset and contextual evaluation
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_conversation_reset_requested(self) -> None:
        """
        Reset the LLM conversation boundary without deleting
        any messages from the database.
        """

        self._context_message_floor_id = (
            self.database.latest_message_id(
                self.session_id
            )
        )

        print(
            "[LLM CONTEXT] Conversation history reset; "
            "database messages retained."
        )

    @pyqtSlot(dict)
    def _evaluate_context_triggers(
        self,
        metrics: dict,
    ) -> None:
        if (
            self._shutting_down
            or self._handling_user_message
            or self._guard_mode_active
        ):
            return

        state = (
            self.state_manager
            .current_state_payload()
        )

        self.event_evaluator.evaluate(
            state,
            metrics,
        )

    # ------------------------------------------------------------------
    # Combined LLM, STT and TTS busy state
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def _on_conversation_llm_busy(
        self,
        busy: bool,
    ) -> None:
        self._conversation_llm_busy = busy
        self._publish_combined_llm_busy()

    @pyqtSlot(bool)
    def _on_contextual_llm_busy(
        self,
        busy: bool,
    ) -> None:
        self._contextual_llm_busy = busy
        self._publish_combined_llm_busy()

    def _publish_combined_llm_busy(self) -> None:
        llm_busy = (
            self._conversation_llm_busy
            or self._contextual_llm_busy
        )

        evaluator_busy = (
            self._guard_mode_active
            or llm_busy
            or self._speech_input_busy
        )

        speech_busy = (
            self._guard_mode_active
            or llm_busy
            or self._tts_busy
        )

        self.event_evaluator.set_busy(
            evaluator_busy
        )

        self.stt.set_interaction_busy(
            speech_busy
        )

        if self.window is not None:
            self.window.set_llm_busy(
                llm_busy
            )

    # ------------------------------------------------------------------
    # Speech-to-text
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_speech_transcript_ready(
        self,
        message: str,
    ) -> None:
        if (
            self._guard_mode_active
            or self._shutting_down
        ):
            return

        clean_message = message.strip()

        if not clean_message or self.window is None:
            return

        self.window.add_conversation_message(
            "user",
            clean_message,
            "VOICE",
        )

        self.on_message_send_requested(
            clean_message,
            "speech",
        )

    @pyqtSlot(bool)
    def _on_stt_ready_changed(
        self,
        ready: bool,
    ) -> None:
        if ready:
            print(
                "[STT] Sherpa wake word and "
                "Faster-Whisper tiny.en are ready."
            )

    @pyqtSlot(bool)
    def _on_stt_capture_busy_changed(
        self,
        busy: bool,
    ) -> None:
        self._speech_input_busy = busy
        self._publish_combined_llm_busy()

    @pyqtSlot(str)
    def _on_wake_word_detected(
        self,
        keyword: str,
    ) -> None:
        print(
            f"[STT] Wake phrase detected: {keyword}"
        )

    @pyqtSlot(str)
    def _on_stt_error(
        self,
        message: str,
    ) -> None:
        print(f"[STT ERROR] {message}")

    # ------------------------------------------------------------------
    # User-initiated conversation LLM
    # ------------------------------------------------------------------

    @pyqtSlot(str, str)
    def on_message_send_requested(
        self,
        message: str,
        source: str,
    ) -> None:
        if (
            self._guard_mode_active
            or self._shutting_down
            or self._conversation_llm_busy
            or self._contextual_llm_busy
        ):
            return

        clean_message = message.strip()

        if not clean_message:
            return

        self._handling_user_message = True

        try:
            # Direct interaction counts as current computer activity.
            self.state_manager.record_user_activity(
                source
            )

            turn_id = str(uuid4())

            state = (
                self.state_manager
                .current_state_payload()
            )

            metrics = (
                self.session_metrics
                .summary()
            )

            history = (
                self.database
                .conversation_messages(
                    self.session_id,
                    limit=10,
                    after_message_id=(
                        self._context_message_floor_id
                    ),
                )
            )

            payload = build_conversation_payload(
                clean_message,
                source,
                state,
                metrics,
                history,
            )

            self.database.save_message(
                session_id=self.session_id,
                role="user",
                content=clean_message,
                source=source,
                turn_id=turn_id,
                context_data=payload,
            )

            self.conversation_llm.generate(
                turn_id,
                payload,
                source,
            )

        finally:
            self._handling_user_message = False
            self._publish_combined_llm_busy()

    # ------------------------------------------------------------------
    # Contextual LLM
    # ------------------------------------------------------------------

    @pyqtSlot(str, dict)
    def request_contextual_response(
        self,
        trigger_name: str,
        trigger_details: dict,
    ) -> None:
        if (
            self._guard_mode_active
            or self._conversation_llm_busy
            or self._contextual_llm_busy
            or self._shutting_down
        ):
            return

        turn_id = str(uuid4())

        state = (
            self.state_manager
            .current_state_payload()
        )

        metrics = (
            self.session_metrics
            .summary()
        )

        payload = build_contextual_payload(
            trigger_name,
            state,
            metrics,
            trigger_details,
        )

        self.database.save_message(
            session_id=self.session_id,
            role="event",
            content=trigger_name,
            source="contextual_event",
            turn_id=turn_id,
            context_data=payload,
        )

        self.contextual_llm.generate(
            turn_id,
            payload,
        )

    # ------------------------------------------------------------------
    # LLM results
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def _on_llm_response(
        self,
        result: dict,
    ) -> None:
        if (
            self._guard_mode_active
            or self._shutting_down
        ):
            return

        actions = list(
            result.get(
                "actions",
                [],
            )
        )

        persona = str(
            result.get(
                "persona",
                "unknown",
            )
        )

        text_response = str(
            result.get(
                "text_response",
                "",
            )
        ).strip()

        self.database.save_message(
            session_id=self.session_id,
            role="assistant",
            content=text_response,
            source=f"{persona}_llm",
            turn_id=str(
                result.get(
                    "turn_id",
                    "",
                )
            ),
            action_data={
                "actions": actions,
                "model": result.get("model"),
                "persona": persona,
            },
        )

        print(f"[LLM PERSONA] {persona}")
        print(f"[LLM MODEL] {result.get('model', 'Unknown')}")
        print(f"[LLM TEXT] {text_response}")
        print(f"[LLM ACTIONS] {actions}")

        for action in actions:
            if action != "no_action":
                print(
                    f"[ACTION SIMULATED] {action}"
                )

        # Guard mode is entered before TTS, so the LLM response
        # is not spoken and guard audio takes control.
        if (
            "guard_mode" in actions
            and self.guard_mode.enter()
        ):
            return

        if text_response:
            self.tts.speak(
                text=text_response,
                turn_id=str(
                    result.get(
                        "turn_id",
                        "",
                    )
                ),
                persona=persona,
            )

    @pyqtSlot(str)
    def _on_llm_error(
        self,
        message: str,
    ) -> None:
        print(f"[LLM ERROR] {message}")

    # ------------------------------------------------------------------
    # Text-to-speech
    # ------------------------------------------------------------------

    @pyqtSlot(bool)
    def _on_tts_busy_changed(
        self,
        busy: bool,
    ) -> None:
        self._tts_busy = busy
        self._publish_combined_llm_busy()

    @pyqtSlot(bool)
    def _on_tts_ready_changed(
        self,
        ready: bool,
    ) -> None:
        if ready:
            print(
                f"[TTS] Kokoro ready: "
                f"voice={self.tts.voice}, "
                f"language={self.tts.language}, "
                f"speed={self.tts.speed:g}."
            )

    @pyqtSlot(dict)
    def _on_tts_audio_ready(
        self,
        result: dict,
    ) -> None:
        print(
            "[TTS] Latest WAV ready: "
            f"revision={result['revision']}, "
            f"size={result['size_bytes']} bytes."
        )

    @pyqtSlot(dict)
    def _on_tts_playback_finished(
        self,
        result: dict,
    ) -> None:
        if result.get("played_locally"):
            print(
                "[TTS] Playback finished: "
                f"turn_id={result['turn_id']}"
            )

    @pyqtSlot(str)
    def _on_tts_error(
        self,
        message: str,
    ) -> None:
        print(f"[TTS ERROR] {message}")

    # ------------------------------------------------------------------
    # Guard mode
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_operating_mode_changed(
        self,
        mode: str,
    ) -> None:
        guard_active = (
            mode.lower() == "guard"
        )

        if (
            self._guard_mode_active
            == guard_active
        ):
            return

        self._guard_mode_active = (
            guard_active
        )

        if guard_active:
            # Stop any existing spoken response before guard
            # warning audio takes control.
            self.tts.stop_playback()

            # Prevent state snapshots, metrics, contextual
            # evaluation and database writes from guard observations.
            self.state_manager.set_suspended(
                True
            )
            self.session_metrics.set_suspended(
                True
            )

            # This closes the actual microphone input stream.
            self.stt.set_guard_mode(True)

            self.activity_monitor.stop()
            self.calendar.stop()

            # Guard mode requires the camera to remain running.
            if not self.webcam.is_running:
                self.webcam.start()

            self._publish_combined_llm_busy()

            if self.window is not None:
                self.window.set_operating_mode(
                    "guard"
                )

            print(
                "[MODE] Guard mode activated. "
                "Only vision and guard audio are active."
            )
            return

        # The owner has returned.
        self.tts.cancel_guard_audio()

        self.state_manager.set_suspended(
            False
        )
        self.session_metrics.set_suspended(
            False
        )

        self.stt.set_guard_mode(False)
        self.activity_monitor.start()
        self.calendar.start()

        self._publish_combined_llm_busy()

        if self.window is not None:
            self.window.set_operating_mode(
                "normal"
            )

        print(
            "[MODE] Owner detected. "
            "Normal mode restored."
        )

    @pyqtSlot(str)
    def _on_guard_warning_requested(
        self,
        wav_path: str,
    ) -> None:
        if (
            not self._guard_mode_active
            or self.tts.is_busy
            or self._shutting_down
        ):
            return

        started = self.tts.play_wav_file(
            wav_path,
            label="guard_warning",
        )

        if started:
            print(
                "[GUARD] Unknown face detected; "
                "warning WAV published and played."
            )

    @pyqtSlot(str)
    def _on_guard_error(
        self,
        message: str,
    ) -> None:
        print(f"[GUARD ERROR] {message}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    @pyqtSlot()
    def shutdown(self) -> None:
        if self._shutting_down:
            return

        self._shutting_down = True

        # Prevent new contextual requests immediately.
        self.event_evaluator.set_busy(True)

        # Stop live inputs and repeating guard events first.
        self.guard_mode.stop()
        self.stt.stop()
        self.activity_monitor.stop()
        self.calendar.stop()
        self.webcam.stop()

        # Stop processing workers after their input producers.
        self.face_recognition.stop()
        self.conversation_llm.stop()
        self.contextual_llm.stop()
        self.tts.stop()
        self.state_manager.stop()

        # Keep the database open until final session data is saved.
        self.state_manager.capture_snapshot(
            "session_ended"
        )
        self.session_metrics.finish_session()
        self.database.finish_app_session(
            self.session_id
        )

        self.database.close()