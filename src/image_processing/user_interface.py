import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QFrame, QSizePolicy, QGridLayout, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from object_detection import Detector, COLOR_PROFILES
from tracker import Tracker, TrackedObject

# ── QSS stylesheet ────────────────────────────────────────────────────────────
QSS = """
QMainWindow, QWidget#root {
    background-color: #0a0c0f;
}

QLabel#title {
    color: #c8f542;
    font-family: "Courier New", monospace;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 6px;
    padding: 8px 0px;
}

QLabel#subtitle {
    color: #4a5568;
    font-family: "Courier New", monospace;
    font-size: 10px;
    letter-spacing: 3px;
}

QLabel#feed_label {
    background-color: #050608;
    border: 1px solid #1e2530;
    border-radius: 2px;
}

QLabel#mask_label {
    background-color: #050608;
    border: 1px solid #1e2530;
    border-radius: 2px;
}

QGroupBox {
    color: #4a5568;
    font-family: "Courier New", monospace;
    font-size: 10px;
    letter-spacing: 2px;
    border: 1px solid #1a2030;
    border-radius: 2px;
    margin-top: 14px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #3a4a60;
}

QLabel#stat_key {
    color: #3a4a60;
    font-family: "Courier New", monospace;
    font-size: 10px;
    letter-spacing: 1px;
}
QLabel#stat_val {
    color: #c8f542;
    font-family: "Courier New", monospace;
    font-size: 11px;
    font-weight: bold;
}

QLabel#status_searching {
    color: #f5a623;
    font-family: "Courier New", monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 6px 12px;
    background-color: #1a1200;
    border: 1px solid #f5a62340;
    border-radius: 2px;
}
QLabel#status_tracking {
    color: #c8f542;
    font-family: "Courier New", monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 6px 12px;
    background-color: #0a1400;
    border: 1px solid #c8f54240;
    border-radius: 2px;
}
QLabel#status_stopped {
    color: #ff4040;
    font-family: "Courier New", monospace;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 6px 12px;
    background-color: #140000;
    border: 1px solid #ff404040;
    border-radius: 2px;
}

QPushButton#btn_start {
    background-color: #0d1a00;
    color: #c8f542;
    border: 1px solid #c8f54260;
    border-radius: 2px;
    font-family: "Courier New", monospace;
    font-size: 11px;
    letter-spacing: 3px;
    padding: 9px 18px;
    font-weight: bold;
}
QPushButton#btn_start:hover {
    background-color: #1a3300;
    border-color: #c8f542;
}
QPushButton#btn_start:pressed {
    background-color: #c8f542;
    color: #0a0c0f;
}
QPushButton#btn_start:disabled {
    color: #2a3020;
    border-color: #1a2010;
    background-color: #0a0c0a;
}

QPushButton#btn_stop {
    background-color: #1a0000;
    color: #ff4040;
    border: 1px solid #ff404060;
    border-radius: 2px;
    font-family: "Courier New", monospace;
    font-size: 11px;
    letter-spacing: 3px;
    padding: 9px 18px;
    font-weight: bold;
}
QPushButton#btn_stop:hover {
    background-color: #330000;
    border-color: #ff4040;
}
QPushButton#btn_stop:pressed {
    background-color: #ff4040;
    color: #0a0c0f;
}
QPushButton#btn_stop:disabled {
    color: #302020;
    border-color: #201010;
    background-color: #0c0a0a;
}

QPushButton#btn_reset {
    background-color: #0a0c10;
    color: #5a6a80;
    border: 1px solid #2a3a50;
    border-radius: 2px;
    font-family: "Courier New", monospace;
    font-size: 10px;
    letter-spacing: 2px;
    padding: 7px 14px;
}
QPushButton#btn_reset:hover {
    color: #8aaad0;
    border-color: #4a6a90;
    background-color: #0f1520;
}

QComboBox {
    background-color: #0d1018;
    color: #8aaad0;
    border: 1px solid #1e2d40;
    border-radius: 2px;
    font-family: "Courier New", monospace;
    font-size: 10px;
    letter-spacing: 2px;
    padding: 5px 10px;
    min-width: 100px;
}
QComboBox:hover {
    border-color: #3a5a80;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #0d1018;
    color: #8aaad0;
    border: 1px solid #1e2d40;
    selection-background-color: #1a2d40;
    font-family: "Courier New", monospace;
    font-size: 10px;
}

QFrame#divider {
    background-color: #1a2030;
    max-height: 1px;
    min-height: 1px;
}

QLabel#section_label {
    color: #2a3a50;
    font-family: "Courier New", monospace;
    font-size: 9px;
    letter-spacing: 3px;
}
"""

# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SARUT — Vision System")
        self.setMinimumSize(1100, 680)

        self.cap = None
        self.tracker = Tracker(profile="Red")

        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)

        self._build_ui()
        self.setStyleSheet(QSS)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("SARUT")
        title.setObjectName("title")
        sub = QLabel("AUTONOMOUS VISION TRACKING SYSTEM")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignBottom)
        header.addWidget(title)
        header.addSpacing(12)
        header.addWidget(sub)
        header.addStretch()

        self.status_label = QLabel("● OFFLINE")
        self.status_label.setObjectName("status_stopped")
        header.addWidget(self.status_label)
        outer.addLayout(header)

        # ── Thin divider ──────────────────────────────────────────────────────
        div = QFrame()
        div.setObjectName("divider")
        outer.addWidget(div)

        # ── Main content row ──────────────────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(16)

        # ── Left: video feeds (camera on top, mask below) ─────────────────────
        feeds = QVBoxLayout()
        feeds.setSpacing(8)

        # primary feed
        cam_lbl = QLabel("CAMERA FEED")
        cam_lbl.setObjectName("section_label")
        self.feed_label = QLabel()
        self.feed_label.setObjectName("feed_label")
        self.feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed_label.setFixedSize(700, 400)
        feeds.addWidget(cam_lbl)
        feeds.addWidget(self.feed_label)

        # mask feed below
        mask_lbl = QLabel("DETECTION MASK")
        mask_lbl.setObjectName("section_label")
        self.mask_label = QLabel()
        self.mask_label.setObjectName("mask_label")
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label.setFixedSize(700, 160)
        feeds.addWidget(mask_lbl)
        feeds.addWidget(self.mask_label)

        content.addLayout(feeds)

        # ── Right: control panel ──────────────────────────────────────────────
        panel = QVBoxLayout()
        panel.setSpacing(12)
        panel.setContentsMargins(0, 0, 0, 0)

        # STATUS
        status_group = QGroupBox("SYSTEM STATUS")
        sg_layout = QGridLayout(status_group)
        sg_layout.setHorizontalSpacing(16)
        sg_layout.setVerticalSpacing(8)

        self._add_stat(sg_layout, 0, "MODE", "—")
        self._add_stat(sg_layout, 1, "TARGET X", "—")
        self._add_stat(sg_layout, 2, "TARGET Y", "—")
        self._add_stat(sg_layout, 3, "BBOX W", "—")
        self._add_stat(sg_layout, 4, "BBOX H", "—")
        self._add_stat(sg_layout, 5, "PROFILE", "Red")
        panel.addWidget(status_group)

        # CONTROLS
        ctrl_group = QGroupBox("CONTROLS")
        cl_layout = QVBoxLayout(ctrl_group)
        cl_layout.setSpacing(8)

        # profile selector
        prof_row = QHBoxLayout()
        prof_key = QLabel("COLOR PROFILE")
        prof_key.setObjectName("stat_key")
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(COLOR_PROFILES.keys()))
        self.profile_combo.currentTextChanged.connect(self._on_profile_change)
        prof_row.addWidget(prof_key)
        prof_row.addStretch()
        prof_row.addWidget(self.profile_combo)
        cl_layout.addLayout(prof_row)

        div2 = QFrame()
        div2.setObjectName("divider")
        cl_layout.addWidget(div2)

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start)

        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)

        self.btn_reset = QPushButton("↺  RESET TRACKER")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self._reset_tracker)

        cl_layout.addWidget(self.btn_start)
        cl_layout.addWidget(self.btn_stop)
        cl_layout.addWidget(self.btn_reset)
        panel.addWidget(ctrl_group)

        panel.addStretch()
        content.addLayout(panel)

        outer.addLayout(content)

        # save stat value labels for update
        self._stat_vals = {}
        for i, key in enumerate(["MODE", "TARGET X", "TARGET Y", "BBOX W", "BBOX H", "PROFILE"]):
            val_lbl = status_group.findChildren(QLabel)[i * 2 + 1]
            self._stat_vals[key] = val_lbl

    def _add_stat(self, layout, row, key, value):
        k = QLabel(key)
        k.setObjectName("stat_key")
        v = QLabel(value)
        v.setObjectName("stat_val")
        layout.addWidget(k, row, 0)
        layout.addWidget(v, row, 1, Qt.AlignmentFlag.AlignRight)

    # ── Slot handlers ─────────────────────────────────────────────────────────
    def _start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("● CAM ERROR")
            self.status_label.setObjectName("status_stopped")
            self.status_label.setStyleSheet(QSS)
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("searching")
        self.timer.start(30)

    def _stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.feed_label.clear()
        self.mask_label.clear()
        self._set_status("offline")
        for key in ["MODE", "TARGET X", "TARGET Y", "BBOX W", "BBOX H"]:
            self._update_stat(key, "—")

    def _reset_tracker(self):
        self.tracker = Tracker(profile=self.profile_combo.currentText())
        self._update_stat("MODE", "SEARCHING")

    def _on_profile_change(self, profile):
        self.tracker.set_profile(profile)
        self._update_stat("PROFILE", profile)

    # ── Frame processing ──────────────────────────────────────────────────────
    def _process_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        mode, bbox = self.tracker.update(frame)
        self._set_status(mode)
        self._update_stat("MODE", mode.upper())

        display = frame.copy()

        if mode == "tracking" and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = x + w // 2, y + h // 2

            # outer bracket corners
            blen = 14
            col = (100, 230, 20)
            thick = 2
            pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            for (px, py), (dx, dy) in zip(pts, dirs):
                cv2.line(display, (px, py), (px + dx * blen, py), col, thick)
                cv2.line(display, (px, py), (px, py + dy * blen), col, thick)

            # crosshair
            cv2.line(display, (cx - 10, cy), (cx + 10, cy), (180, 255, 60), 2)
            cv2.line(display, (cx, cy - 10), (cx, cy + 10), (180, 255, 60), 2)
            cv2.circle(display, (cx, cy), 3, (180, 255, 60), -1)

            self._update_stat("TARGET X", str(cx))
            self._update_stat("TARGET Y", str(cy))
            self._update_stat("BBOX W", str(w))
            self._update_stat("BBOX H", str(h))
        else:
            self._update_stat("TARGET X", "—")
            self._update_stat("TARGET Y", "—")
            self._update_stat("BBOX W", "—")
            self._update_stat("BBOX H", "—")

        # render frames to labels
        self._render_frame(display, self.feed_label)

        # get mask from detector for display
        _, mask = self.tracker.detector.detect(frame)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._render_frame(mask_bgr, self.mask_label)

    def _render_frame(self, frame, label):
        h, w, ch = frame.shape
        target_w = label.width()
        target_h = label.height()
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.strides[0], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, mode):
        mapping = {
            "searching": ("● SEARCHING", "status_searching"),
            "tracking":  ("● TRACKING",  "status_tracking"),
            "offline":   ("● OFFLINE",   "status_stopped"),
        }
        text, obj = mapping.get(mode, ("● OFFLINE", "status_stopped"))
        self.status_label.setText(text)
        self.status_label.setObjectName(obj)
        # force QSS re-apply after objectName change
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _update_stat(self, key, value):
        lbl = self._stat_vals.get(key)
        if lbl:
            lbl.setText(value)

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())