from PyQt5.QtWidgets import QLayout, QFrame, QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt


def make_label(text: str, object_name: str = "") -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


def make_panel() -> QFrame:
    frame = QFrame()
    frame.setObjectName("panel")
    return frame

def _mini_header(
        symbol: str,
        title: str,
        accent: str,
        trailing: QWidget | None = None,
    ) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(9)
        icon = QLabel(symbol)
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(32, 32)
        icon.setStyleSheet(
            f"background:{accent}22; color:{accent}; border-radius:9px; "
            "font-size:16px; font-weight:700;"
        )
        layout.addWidget(icon)
        layout.addWidget(make_label(title.upper(), "miniHeading"))
        layout.addStretch(1)
        if trailing is not None:
            layout.addWidget(trailing)
        return layout


def _set_margins(layout:QLayout, left=16, top=16, right=16, bottom=16) -> None:
    layout.setContentsMargins(left, top, right, bottom)


def _duration(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
