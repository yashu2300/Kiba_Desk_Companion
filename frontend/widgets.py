from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QTextEdit, QDialog, QDialogButtonBox, QCheckBox, QLineEdit
from PyQt5.QtCore import QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QKeyEvent, QPen
from frontend.utils import make_label, _set_margins


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str, meta: str = "TEXT", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("messageBubble")
        if role == "user":
            self.setStyleSheet(
                "QFrame#messageBubble { background:#24151a; border:1px solid #5b302f; "
                "border-radius:12px; }"
            )
            speaker = "YOU"
        elif role == "event":
            self.setStyleSheet(
                "QFrame#messageBubble { background:#111827; border:1px dashed #53617e; "
                "border-radius:12px; }"
            )
            speaker = "EVENT"
        else:
            self.setStyleSheet(
                "QFrame#messageBubble { background:#12192a; border:1px solid #28334c; "
                "border-radius:12px; }"
            )
            speaker = "KIBO"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(5)
        header = QHBoxLayout()
        name = make_label(speaker, "miniHeading")
        source = make_label(meta.upper(), "helper")
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(source)
        message = QLabel(text)
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addLayout(header)
        layout.addWidget(message)

class StatCell(QFrame):
    def __init__(self, heading: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("subtleBox")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(3)
        heading_label = make_label(heading.upper(), "kicker")
        self.value_label = make_label(value, "bodyStrong")
        self.value_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(heading_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)

class CameraStage(QWidget):
    """Painted camera placeholder; no webcam dependency is used."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(500)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.camera_active = False
        self._frame: QImage | None = None

    def set_camera_active(self, active: bool) -> None:
        self.camera_active = active
        if not active:
            self._frame = None
        self.update()

    def set_frame(self, frame: QImage) -> None:
        """Receive an already detached frame from WebcamService."""

        self._frame = frame
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt method name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#05070d"))

        if self._frame is not None and not self._frame.isNull():
            scaled = self._frame.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            source_x = max(0, (scaled.width() - self.width()) // 2)
            source_y = max(0, (scaled.height() - self.height()) // 2)
            painter.drawImage(
                self.rect(),
                scaled,
                QRect(source_x, source_y, self.width(), self.height()),
            )
        else:
            grid_pen = QPen(QColor(39, 50, 75, 55), 1)
            painter.setPen(grid_pen)
            grid_size = 42
            for x in range(0, self.width(), grid_size):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), grid_size):
                painter.drawLine(0, y, self.width(), y)

            painter.setPen(QColor("#637091"))
            painter.setFont(QFont("Segoe UI Symbol", 28))
            painter.drawText(
                QRectF(0, self.height() * 0.30, self.width(), 50),
                Qt.AlignCenter,
                "▣" if self.camera_active else "⊘",
            )

            painter.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
            painter.setPen(QColor("#d7dded"))
            title = "Waiting for the first frame" if self.camera_active else "Camera is paused"
            painter.drawText(
                QRectF(20, self.height() * 0.48, self.width() - 40, 26),
                Qt.AlignCenter,
                title,
            )

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor("#808aa2"))
            helper = (
                "The local webcam service is starting."
                if self.camera_active
                else "Press Start camera to open the laptop webcam."
            )
            painter.drawText(
                QRectF(35, self.height() * 0.56, self.width() - 70, 38),
                Qt.AlignHCenter | Qt.TextWordWrap,
                helper,
            )

        overlay_font = QFont("Consolas", 8)
        painter.setFont(overlay_font)
        painter.setPen(QColor("#dce2f5"))
        bottom = self.height() - 15
        left = "LIVE LAPTOP WEBCAM" if self._frame is not None else "NO FACE DETECTED"
        painter.drawText(14, bottom, left)
        right = "VISION NOT CONNECTED" if self._frame is not None else "UNKNOWN"
        width = painter.fontMetrics().horizontalAdvance(right)
        painter.drawText(self.width() - width - 14, bottom, right)

class ComposerTextEdit(QTextEdit):
    """Text editor where Enter submits and Shift+Enter inserts a newline."""

    submit_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.submit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

class GoalDialog(QDialog):
    """Local-only goal/settings editor."""

    def __init__(
        self,
        display_name: str,
        goal: str,
        automatic_nudges: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Desk Companion settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        _set_margins(layout, 22, 20, 22, 20)
        layout.setSpacing(12)

        icon = make_label("◎", "brandIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(42, 42)
        icon.setStyleSheet(
            "background:#202a43; color:#95a6ff; border-radius:11px; font-size:21px;"
        )
        title = make_label("Set the companion's direction", "panelTitle")
        description = make_label(
            "These values are local UI state for now. The save slot is where you can "
            "later call FastAPI or a custom Python function.",
            "muted",
        )
        description.setWordWrap(True)

        self.name_input = QLineEdit(display_name)
        self.name_input.setPlaceholderText("Display name (optional)")
        self.goal_input = QTextEdit(goal)
        self.goal_input.setPlaceholderText("What goal should Kibo help with?")
        self.goal_input.setFixedHeight(95)
        self.nudges_checkbox = QCheckBox("Enable automatic contextual nudges")
        self.nudges_checkbox.setChecked(automatic_nudges)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        save_button = buttons.button(QDialogButtonBox.Save)
        save_button.setObjectName("primaryButton")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(icon, 0, Qt.AlignLeft)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(3)
        layout.addWidget(make_label("DISPLAY NAME", "kicker"))
        layout.addWidget(self.name_input)
        layout.addWidget(make_label("INITIAL GOAL", "kicker"))
        layout.addWidget(self.goal_input)
        layout.addWidget(self.nudges_checkbox)
        layout.addSpacing(4)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        if len(self.goal_input.toPlainText().strip()) < 3:
            self.goal_input.setFocus()
            self.goal_input.setStyleSheet("border-color:#ed5d69;")
            return
        self.accept()

    def values(self) -> tuple[str, str, bool]:
        return (
            self.name_input.text().strip(),
            self.goal_input.toPlainText().strip(),
            self.nudges_checkbox.isChecked(),
        )
