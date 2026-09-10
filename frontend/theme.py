"""Central colour palette and Qt stylesheet for the Kibo interface."""

BACKGROUND = "#070a12"
CARD = "#0e1322"
BORDER = "#25304a"
PRIMARY = "#6e85ff"
TEXT = "#f2f5ff"
MUTED = "#8791aa"
SUCCESS = "#49d3a1"
CORAL = "#ff8d6b"


APP_STYLESHEET = r"""
* {
    font-family: "Segoe UI", "Aptos", Arial, sans-serif;
    color: #f2f5ff;
    font-size: 13px;
}
QMainWindow, QWidget#appRoot, QScrollArea#pageScroll,
QScrollArea#pageScroll > QWidget > QWidget {
    background: #070a12;
}
QScrollArea { border: none; }
QFrame#panel {
    background: #0e1423;
    border: 1px solid #25304a;
    border-radius: 15px;
}
QFrame#subtleBox {
    background: #090e19;
    border: 1px solid #242f47;
    border-radius: 10px;
}
QFrame#brandMark {
    background: #6379ee;
    border: 1px solid #8193ff;
    border-radius: 13px;
}
QLabel#brandIcon { color: white; font-size: 21px; font-weight: 700; }
QLabel#eyebrow, QLabel#kicker {
    color: #8791aa;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
QLabel#appTitle { font-size: 20px; font-weight: 600; }
QLabel#panelTitle { font-size: 20px; font-weight: 600; }
QLabel#miniHeading {
    color: #aeb7cd;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#muted { color: #8791aa; }
QLabel#helper { color: #7f899f; font-size: 11px; }
QLabel#bodyStrong { color: #f1f3f9; font-size: 15px; font-weight: 600; }
QLabel#sensorOnline { color: #c2cfca; font-size: 11px; }
QLabel#sensorOffline { color: #929cb4; font-size: 11px; }
QLabel#greenDot { color: #49d3a1; font-size: 16px; }
QLabel#greyDot { color: #667085; font-size: 16px; }
QLabel#chip, QLabel#chipLive, QLabel#chipNormal, QLabel#chipGuard, QLabel#chipSimulated {
    border: 1px solid #303b55;
    border-radius: 9px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#chipGuard { color: #ffb4b4; border-color: #a83c4a; background: #3a1119; }
QLabel#chip { color: #7f8aa4; }
QLabel#chipLive { color: #72e2b8; border-color: #30785f; background: #0c2c24; }
QLabel#chipNormal { color: #c7cfdf; background: #18223a; }
QLabel#chipSimulated { color: #d5c7ff; border-color: #514274; }
QPushButton {
    min-height: 36px;
    padding: 0 14px;
    border: 1px solid #303b55;
    border-radius: 10px;
    background: #151d30;
    color: #e7ebff;
    font-weight: 600;
}
QPushButton:hover { background: #202a43; border-color: #53617e; }
QPushButton:pressed { background: #293651; }
QPushButton:disabled { color: #626d84; background: #111726; border-color: #222b40; }
QPushButton#primaryButton {
    background: #6e85ff;
    border-color: #8295ff;
    color: white;
}
QPushButton#primaryButton:hover { background: #7b90ff; }
QPushButton#iconButton {
    min-width: 38px;
    max-width: 38px;
    padding: 0;
    font-size: 17px;
}
QPushButton#linkButton {
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    padding: 0;
    border: none;
    background: transparent;
    color: #aeb7cd;
    font-size: 18px;
}
QPushButton#linkButton:hover { color: white; background: #1b2439; }
QComboBox {
    min-height: 38px;
    padding: 0 30px 0 12px;
    border: 1px solid #303b55;
    border-radius: 10px;
    background: #101726;
    color: #dce2f5;
}
QComboBox:hover { border-color: #53617e; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView {
    background: #12182a;
    border: 1px solid #303b55;
    selection-background-color: #263352;
    outline: none;
}
QLineEdit, QTextEdit {
    background: #080d18;
    border: 1px solid #303b55;
    border-radius: 10px;
    padding: 10px;
    selection-background-color: #5269df;
}
QLineEdit:focus, QTextEdit:focus { border-color: #8295ff; }
QTextEdit#composerInput { min-height: 58px; }
QCheckBox { color: #d7ddec; spacing: 9px; }
QCheckBox::indicator {
    width: 18px; height: 18px;
    border: 1px solid #3b4865;
    border-radius: 5px;
    background: #090e19;
}
QCheckBox::indicator:checked { background: #6e85ff; border-color: #8295ff; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: #34405c; min-height: 28px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QDialog { background: #0e1423; }
QToolTip {
    color: #f2f5ff;
    background: #1a2237;
    border: 1px solid #3b4865;
    padding: 5px;
}
"""

