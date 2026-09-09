from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QScrollArea, 
    QVBoxLayout, QGridLayout, QHBoxLayout, QFrame,
    QPushButton, QComboBox, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage

from frontend.utils import _duration, _set_margins, make_label, make_panel, _mini_header
from frontend.widgets import StatCell, CameraStage, ComposerTextEdit, GoalDialog, CalendarEventCard, MessageBubble

class DeskCompanionWindow(QMainWindow):

    # These signals are triggering something in the backend -> NOTE THESE SIGNALS ARE NOT UI SIGNALS. UI SIGNALS will need its own slots below
    demo_clock_changed = pyqtSignal(str, float) 
    settings_opened = pyqtSignal()
    camera_toggled = pyqtSignal(bool)
    face_enrollment_requested = pyqtSignal()
    goal_editor_opened = pyqtSignal()
    goal_saved = pyqtSignal(str, str, bool)
    calendar_refresh_requested = pyqtSignal()
    microphone_toggled = pyqtSignal(bool)
    message_send_requested = pyqtSignal(str, str) # Text Input sent
    reset_conversation_requested = pyqtSignal()
    user_activity_detected = pyqtSignal(str)
    window_close_requested = pyqtSignal()
    action_execute_requested = pyqtSignal(str)


    CLOCK_OPTIONS = (
        ("1× real time", 1.0),
        ("10× demo", 10.0),
        ("50× demo", 50.0),
        ("100× demo", 100.0),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kibo - Desk Companion")
        desktop_geometry = QApplication.desktop().availableGeometry()
        self.resize(desktop_geometry.width(), desktop_geometry.height())
        
        self.display_name = ""
        self.goal = "No goal set yet"
        self.automatic_nudges = True

        self.camera_active = False
        self.camera_starting = False
        self.recording = False
        
        self._build_ui()
        self._connect_signals()

    # UI Builder and Components
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        page_scroll = QScrollArea()
        page_scroll.setObjectName("pageScroll")
        page_scroll.setWidgetResizable(True)
        page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root_layout.addWidget(page_scroll)

        page = QWidget()
        page_scroll.setWidget(page)
        page_layout = QVBoxLayout(page)
        _set_margins(page_layout, 26, 18, 26, 26)
        page_layout.setSpacing(16)

        page_layout.addWidget(self._build_header())

        workspace = QGridLayout()
        workspace.setHorizontalSpacing(16)
        workspace.setVerticalSpacing(16)
        workspace.setColumnStretch(0, 7)
        workspace.setColumnStretch(1, 3)

        workspace.addWidget(self._build_camera_panel(), 0, 0)
        workspace.addLayout(self._build_context_column(), 0, 1)
        workspace.addWidget(self._build_conversation_panel(), 1, 0, 1, 2)
        page_layout.addLayout(workspace)
        page_layout.addStretch(1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        brand_mark = QFrame()
        brand_mark.setObjectName("brandMark")
        brand_mark.setFixedSize(48, 48)
        mark_layout = QVBoxLayout(brand_mark)
        mark_layout.setContentsMargins(0, 0, 0, 0)
        robot = make_label("♙", "brandIcon")
        robot.setAlignment(Qt.AlignCenter)
        mark_layout.addWidget(robot)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_box.addWidget(make_label("Dashboard", "eyebrow"))
        title_box.addWidget(make_label("Kibo Desk Companion", "appTitle"))

        layout.addWidget(brand_mark)
        layout.addLayout(title_box)
        layout.addStretch(1)

        clock_box = QFrame()
        clock_box.setObjectName("subtleBox")
        clock_layout = QHBoxLayout(clock_box)
        clock_layout.setContentsMargins(11, 3, 5, 3)
        clock_layout.setSpacing(8)
        clock_layout.addWidget(make_label("◴", "brandIcon"))
        clock_layout.addWidget(make_label("Demo clock", "muted"))
        self.clock_combo = QComboBox()
        self.clock_combo.setMinimumWidth(145)
        for label, speed in self.CLOCK_OPTIONS:
            self.clock_combo.addItem(label, speed)
        clock_layout.addWidget(self.clock_combo)
        layout.addWidget(clock_box)

        self.settings_button = QPushButton("⚙")
        self.settings_button.setObjectName("iconButton")
        self.settings_button.setToolTip("Open settings")
        layout.addWidget(self.settings_button)
        return header

    def _build_camera_panel(self) -> QFrame:
        panel = make_panel()
        layout = QVBoxLayout(panel)
        _set_margins(layout, 18, 17, 18, 16)
        layout.setSpacing(13)

        heading = QHBoxLayout()
        heading.addWidget(make_label("Visual Input", "panelTitle"))
        self.camera_badge = make_label("OFFLINE", "chip")
        heading.addStretch(1)
        heading.addWidget(self.camera_badge, 0, Qt.AlignTop)
        layout.addLayout(heading)

        self.camera_stage = CameraStage()
        layout.addWidget(self.camera_stage, 1)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.camera_button = QPushButton("▣  Start camera")
        self.camera_button.setObjectName("primaryButton")
        self.enroll_button = QPushButton("◎  Enroll my face")
        self.enroll_button.hide()

        helper = make_label("Identity enrollment is local and optional; Computer Vision labels are non-medical estimates.", "helper")
        helper.setWordWrap(True)
        helper.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        controls.addWidget(self.camera_button)
        controls.addWidget(self.enroll_button)
        controls.addStretch(1)
        controls.addWidget(helper, 2)
        layout.addLayout(controls)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.presence_stat = StatCell("Presence", "Unknown")
        self.identity_stat = StatCell("Identity", "Unknown")
        self.posture_stat = StatCell("Posture", "Unknown")
        self.expression_stat = StatCell("Expression", "Unknown")
        for stat in (
            self.presence_stat,
            self.identity_stat,
            self.posture_stat,
            self.expression_stat,
        ):
            stats.addWidget(stat, 1)
        layout.addLayout(stats)
        return panel

    def _build_context_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(16)
        column.addWidget(self._build_goal_panel())
        column.addWidget(self._build_telemetry_panel())
        column.addWidget(self._build_calendar_panel())
        column.addWidget(self._build_action_panel())
        column.addStretch(1)
        return column

    def _build_goal_panel(self) -> QFrame:
        panel = make_panel()
        layout = QVBoxLayout(panel)
        _set_margins(layout)
        layout.setSpacing(10)
        self.goal_edit_button = QPushButton("›")
        self.goal_edit_button.setObjectName("linkButton")
        self.goal_edit_button.setToolTip("Edit goal")
        layout.addLayout(_mini_header("◎", "Current goal", "#ff8d6b", self.goal_edit_button))
        self.goal_label = make_label(self.goal, "bodyStrong")
        self.goal_label.setWordWrap(True)
        layout.addWidget(self.goal_label)
        return panel

    def _build_telemetry_panel(self) -> QFrame:
        panel = make_panel()
        layout = QVBoxLayout(panel)
        _set_margins(layout)
        layout.setSpacing(12)
        layout.addLayout(_mini_header("⌁", "Live context", "#8499ff", make_label("NORMAL", "chipNormal")))

        row = QHBoxLayout()
        row.setSpacing(8)
        self.inactive_stat = StatCell("Inactive", "0s")
        self.away_stat = StatCell("Away", "0s")
        self.clock_stat = StatCell("Clock", "1×")
        row.addWidget(self.inactive_stat, 1)
        row.addWidget(self.away_stat, 1)
        row.addWidget(self.clock_stat, 1)
        layout.addLayout(row)
        return panel

    def _build_calendar_panel(self) -> QFrame:
        panel = make_panel()

        layout = QVBoxLayout(panel)
        _set_margins(layout)
        layout.setSpacing(10)

        refresh = QPushButton("↻")
        refresh.setObjectName("linkButton")
        refresh.setToolTip("Refresh Google Calendar")

        self.calendar_refresh_button = refresh

        layout.addLayout(_mini_header("▦", "Today's calendar", "#f2c66e", refresh))

        self.calendar_status_label = make_label("CONNECTING","chip")
        layout.addWidget(self.calendar_status_label)

        self.calendar_summary_label = make_label("Loading today's events…", "helper")
        self.calendar_summary_label.setWordWrap(True)
        layout.addWidget(self.calendar_summary_label)
        self.calendar_events_widget = QWidget()

        self.calendar_events_layout = QVBoxLayout(self.calendar_events_widget)
        self.calendar_events_layout.setContentsMargins(0,0,0,0)
        self.calendar_events_layout.setSpacing(7)

        layout.addWidget(self.calendar_events_widget)
        return panel

    def _build_action_panel(self) -> QFrame:
        panel = make_panel()
        layout = QVBoxLayout(panel)
        _set_margins(layout)
        layout.setSpacing(10)
        layout.addLayout(
            _mini_header("♙", "Recent Petoi Output", "#b697ff", make_label("SIMULATED", "chipSimulated"))
        )

        action_box = QFrame()
        action_box.setObjectName("subtleBox")
        action_layout = QHBoxLayout(action_box)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.addWidget(make_label("↶", "muted"))
        self.action_label = make_label("Commands will appear here", "helper")
        action_layout.addWidget(self.action_label, 1)
        layout.addWidget(action_box)
        return panel

    def _build_conversation_panel(self) -> QFrame:
        panel = make_panel()
        panel.setMinimumHeight(390)
        
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        heading_widget = QWidget()
        heading_layout = QHBoxLayout(heading_widget)
        _set_margins(heading_layout, 18, 15, 18, 14)
        title_box = QVBoxLayout()
        title_box.setSpacing(3)
        title_box.addWidget(make_label("Chat with Kibo", "panelTitle"))
        heading_layout.addLayout(title_box)
        heading_layout.addStretch(1)

        self.reset_button = QPushButton("↻")
        self.reset_button.setObjectName("linkButton")
        self.reset_button.setToolTip("Clear visible chat and reset LLM conversation context")
        heading_layout.addWidget(self.reset_button)
        main_layout.addWidget(heading_widget)

        self.conversation_scroll = QScrollArea()
        self.conversation_scroll.setWidgetResizable(True)
        #self.conversation_scroll.setMinimumHeight(180)
        self.conversation_container = QWidget()
        self.conversation_layout = QVBoxLayout(self.conversation_container)
        _set_margins(self.conversation_layout, 18, 15, 18, 15)
        self.conversation_layout.setSpacing(10)
        self.empty_conversation = QWidget()
        empty_layout = QVBoxLayout(self.empty_conversation)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_layout.setSpacing(6)
        empty_icon = make_label("▤", "muted")
        empty_icon.setAlignment(Qt.AlignCenter)
        empty_icon.setStyleSheet("font-size:32px; color:#7180a0;")
        empty_title = make_label("Start with a message or your voice", "helper")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_icon)
        empty_layout.addWidget(empty_title)
        self.conversation_layout.addWidget(self.empty_conversation, 1)
        self.conversation_layout.addStretch(1)
        self.conversation_scroll.setWidget(self.conversation_container)

        self.notice_label = QLabel("")
        self.notice_label.setWordWrap(True)
        self.notice_label.setStyleSheet(
            "background:#32151d; color:#ffadb4; border:1px solid #70313d; "
            "border-radius:9px; padding:9px 11px; margin:0 16px 8px 16px;"
        )
        self.notice_label.hide()

        scroll_lay = QVBoxLayout()
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(0)

        scroll_lay.addWidget(self.conversation_scroll, 1)
        scroll_lay.addWidget(self.notice_label)
        layout.addLayout(scroll_lay)

        composer = QWidget()
        composer_layout = QVBoxLayout(composer)
        _set_margins(composer_layout, 18, 12, 18, 15)
        composer_layout.setSpacing(8)
        self.message_input = ComposerTextEdit()
        self.message_input.setObjectName("composerInput")
        self.message_input.setPlaceholderText("Message Kibo…")
        composer_layout.addWidget(self.message_input)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.microphone_button = QPushButton("Record voice")
        self.send_button = QPushButton("➤  Send")
        self.send_button.setObjectName("primaryButton")
        helper = make_label("Enter to send · Shift + Enter for a new line", "helper")
        actions.addWidget(self.microphone_button)
        actions.addWidget(helper)
        actions.addStretch(1)
        actions.addWidget(self.send_button)
        composer_layout.addLayout(actions)

        layout.addWidget(composer)
        main_layout.addLayout(layout)
        return panel


    # Connect UI Components to backend slots
    def _connect_signals(self):
        self.camera_button.clicked.connect(self._toggle_camera)
        self.enroll_button.clicked.connect(lambda _checked=False: self.face_enrollment_requested.emit())
        self.clock_combo.currentIndexChanged.connect(self._on_clock_changed)
        self.goal_edit_button.clicked.connect(self._open_goal_editor)
        self.settings_button.clicked.connect(self._open_goal_editor)
        self.calendar_refresh_button.clicked.connect(lambda _checked=False: self.calendar_refresh_requested.emit())
        self.send_button.clicked.connect(self._submit_text_message)
        self.message_input.submit_requested.connect(self._submit_text_message)
        self.reset_button.clicked.connect(self._reset_conversation)
    

    def _show_notice(
        self,
        message: str,
    ) -> None:
        self.notice_label.setText(message)
        self.notice_label.show()

    # Web Cam Related
    def _toggle_camera(self):
        if self.camera_active or self.camera_starting: # If camera already active or is starting -> PAUSE/STOP
            self.camera_toggled.emit(False)
            self._set_camera_ui_stopped()
            return

        # Otherwise need to start the camera
        self.notice_label.hide()
        self.camera_starting = True
        self.camera_button.setEnabled(False)
        self.camera_button.setText("◌  Starting camera…")
        self.camera_badge.setText("STARTING")
        self.camera_badge.setObjectName("chip")
        self._repolish_camera_controls()
        self.camera_toggled.emit(True)

    def _set_camera_ui_stopped(self) -> None:
        self.presence_stat.set_value("Unknown")
        self.identity_stat.set_value("Unknown")
        self.expression_stat.set_value("Unknown")
        self.posture_stat.set_value("Unknown")

        self.camera_starting = False
        self.camera_active = False
        self.camera_stage.set_camera_active(False)
        self.camera_button.setEnabled(True)
        self.camera_button.setText("▣  Start camera")
        self.camera_button.setObjectName("primaryButton")
        self.camera_badge.setText("OFFLINE")
        self.camera_badge.setObjectName("chip")
        self.enroll_button.hide()
        self.camera_button.setToolTip("Open the laptop webcam")
        self._repolish_camera_controls()

    def _repolish_camera_controls(self) -> None:
        # Re-polish because objectName changes at runtime.
        self.camera_button.style().unpolish(self.camera_button)
        self.camera_button.style().polish(self.camera_button)
        self.camera_badge.style().unpolish(self.camera_badge)
        self.camera_badge.style().polish(self.camera_badge)

    @pyqtSlot(QImage)
    def set_camera_frame(self, frame: QImage) -> None:
        self.camera_stage.set_frame(frame)

    @pyqtSlot(int, int, int, str)
    def show_camera_opened(self, camera_index: int, width: int, height: int, backend_name: str) -> None:
        self.camera_starting = False
        self.camera_active = True
        self.camera_stage.set_camera_active(True)
        self.camera_button.setEnabled(True)
        self.camera_button.setText("Ⅱ  Pause camera")
        self.camera_button.setObjectName("")
        self.camera_badge.setText("LIVE")
        self.camera_badge.setObjectName("chipLive")
        self.enroll_button.show()
        self.enroll_button.setEnabled(False)
        self.enroll_button.setToolTip("Waiting for the first vision frame.")
        self.presence_stat.set_value("Vision pending")
        self.camera_button.setToolTip(
            f"Camera {camera_index}: {width}×{height} using {backend_name}"
        )
        self._repolish_camera_controls()

    @pyqtSlot(str)
    def show_camera_error(self, message: str) -> None:
        self._show_notice(message)
        self._set_camera_ui_stopped()

    @pyqtSlot()
    def show_camera_stopped(self) -> None:
        self._set_camera_ui_stopped()


    # Computer Vision
    @pyqtSlot(dict)
    def show_vision_result(self, result: dict) -> None:
        """Update the visual-context cards."""
        # At least one frame has now passed through the CV service.
        if "Enrolling" not in self.enroll_button.text():
            self.enroll_button.setEnabled(True)
            self.enroll_button.setToolTip("Store the currently visible face as the owner.")

        self.camera_stage.set_vision_result(result)
        self.presence_stat.set_value(result.get("presence", "Unknown"))
        self.identity_stat.set_value(result.get("identity", "Unknown"))
        self.expression_stat.set_value(result.get("expression", "Unknown"))
        self.posture_stat.set_value(result.get("posture", "Unknown"))

    @pyqtSlot(bool)
    def set_face_enrollment_busy(self,busy: bool) -> None:
        self.enroll_button.setEnabled(not busy)
        self.enroll_button.setText("◌  Enrolling…" if busy else "◎  Enroll my face")

    @pyqtSlot(bool, str)
    def show_face_enrollment_result(self, success: bool,message: str) -> None:
        self.enroll_button.setEnabled(True)

        self.enroll_button.setText("◎  Re-enroll my face" if success else "◎  Enroll my face")

        self.enroll_button.setToolTip(message)

        if success:
            self.identity_stat.set_value("Owner enrolled")
        else:
            self._show_notice(message)

    @pyqtSlot(str)
    def show_vision_error(
        self,
        message: str,
    ) -> None:
        self.presence_stat.set_value(
            "Vision unavailable"
        )

        self._show_notice(message)

    # Calendar
    @pyqtSlot(bool)
    def set_calendar_busy(
        self,
        busy: bool,
    ) -> None:
        self.calendar_refresh_button.setEnabled(
            not busy
        )

        self.calendar_refresh_button.setText(
            "…" if busy else "↻"
        )

    @pyqtSlot(str)
    def show_calendar_error(
        self,
        message: str,
    ) -> None:
        self.calendar_status_label.setText(
            "UNAVAILABLE"
        )

        self.calendar_summary_label.setText(
            "Calendar could not be loaded. "
            + message
        )

    @pyqtSlot(dict)
    def show_calendar(
        self,
        result: dict,
    ) -> None:
        while self.calendar_events_layout.count():
            item = (
                self.calendar_events_layout
                .takeAt(0)
            )

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        if not result.get("connected", False):
            self.calendar_status_label.setText(
                "UNAVAILABLE"
            )
            return

        events = list(
            result.get("events", [])
        )

        past_count = int(
            result.get("past_count", 0)
        )

        current_count = int(
            result.get("current_count", 0)
        )

        upcoming_count = int(
            result.get("upcoming_count", 0)
        )

        if current_count:
            self.calendar_status_label.setText(
                "EVENT NOW"
            )
        elif upcoming_count:
            self.calendar_status_label.setText(
                "UPCOMING"
            )
        else:
            self.calendar_status_label.setText(
                "CLEAR"
            )

        self.calendar_summary_label.setText(
            f"{past_count} passed · "
            f"{current_count} now · "
            f"{upcoming_count} upcoming"
        )

        if not events:
            empty = make_label(
                "No events scheduled for today.",
                "helper",
            )

            self.calendar_events_layout.addWidget(
                empty
            )
            return

        for event in events:
            self.calendar_events_layout.addWidget(
                CalendarEventCard(event)
            )

    # Demo Clock + Settings
    def _on_clock_changed(self, _index: int) -> None:
        label = self.clock_combo.currentText()
        speed = float(
            self.clock_combo.currentData()
        )

        self.clock_stat.set_value(
            f"{speed:g}×"
        )

        self.demo_clock_changed.emit(
            label,
            speed,
        )

    def _open_goal_editor(self) -> None:
        self.goal_editor_opened.emit()
        dialog = GoalDialog(self.display_name, (self.goal if self.goal != "No goal set yet" else ""), self.automatic_nudges, self)

        if dialog.exec_() != dialog.Accepted:
            return

        (display_name, goal, automatic_nudges) = dialog.values()

        self.goal_saved.emit(display_name, goal, automatic_nudges)

    @pyqtSlot(str, str, bool)
    def load_profile(self, display_name: str, goal: str, automatic_nudges: bool) -> None:
        self.display_name = display_name
        self.goal = goal or "No goal set yet"
        self.automatic_nudges = automatic_nudges

        self.goal_label.setText(self.goal)

    @pyqtSlot(dict)
    def show_current_state(self,state: dict) -> None:
        self.inactive_stat.set_value(
            _duration(
                state.get(
                    "inactive_seconds",
                    0,
                )
            )
        )

        self.away_stat.set_value(
            _duration(
                state.get(
                    "away_seconds",
                    0,
                )
            )
        )

        speed = state.get(
            "clock_speed",
            1,
        )

        self.clock_stat.set_value(
            f"{speed:g}×"
        )

    # LLM + Text Trigger
    def _submit_text_message(self) -> None:
        message = (self.message_input.toPlainText().strip())
        if not message:
            return

        self.message_input.clear()

        self.add_conversation_message("user",message,"TEXT")

        self.message_send_requested.emit(message,"text")

    def add_conversation_message(self,role: str, text: str, source: str) -> None:
        self.empty_conversation.hide()

        bubble = MessageBubble(role=role, text=text, meta=source)

        insert_at = max(0, self.conversation_layout.count() - 1)

        self.conversation_layout.insertWidget(insert_at, bubble)

        def scroll_to_bottom() -> None:
            scroll_bar = (self.conversation_scroll.verticalScrollBar())

            scroll_bar.setValue(scroll_bar.maximum())

        QTimer.singleShot(0, scroll_to_bottom)

    @pyqtSlot(bool)
    def set_llm_busy(self,busy: bool) -> None:
        self.reset_button.setEnabled(not busy)
        self.goal_edit_button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.send_button.setText("◌  Thinking…" if busy else "➤  Send")


    @pyqtSlot(dict)
    def show_llm_response(self, result: dict) -> None:
        self.add_conversation_message("assistant", str(result.get("text_response", "")), "LLM",)

        actions = list(result.get("actions", []))

        visible_actions = [action for action in actions if action != "no_action"]

        if visible_actions:
            self.action_label.setText(" → ".join(visible_actions))
        else:
            self.action_label.setText("No movement requested")

    @pyqtSlot(str)
    def show_llm_error(
        self,
        message: str,
    ) -> None:
        self._show_notice(
            "Kibo could not generate a "
            f"response: {message}"
        )

    @pyqtSlot()
    def clear_conversation(self) -> None:
        """
        Remove message bubbles from the interface.

        This method does not communicate with the database.
        """

        for index in reversed(
            range(
                self.conversation_layout.count()
            )
        ):
            item = (
                self.conversation_layout.itemAt(
                    index
                )
            )

            widget = item.widget()

            if isinstance(
                widget,
                MessageBubble,
            ):
                self.conversation_layout.removeWidget(
                    widget
                )

                widget.deleteLater()

        self.empty_conversation.show()


    def _reset_conversation(self) -> None:
        """
        Clear the interface and notify the controller
        that earlier messages must not be sent to the LLM.
        """

        self.clear_conversation()

        self.reset_conversation_requested.emit()




    