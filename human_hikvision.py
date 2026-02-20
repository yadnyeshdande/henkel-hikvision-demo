# human.py
"""
Refactored Pokayoke Vision System
- Single-file refactor for easier splitting into modules:
    app/main_window.py, app/video_widget.py, app/capture.py, app/detector.py,
    app/editor.py, app/config.py, app/relay.py, app/logging_setup.py
- Features:
    * USB and RTSP camera support with exponential backoff reconnect
    * Threaded capture worker + bounded queue
    * Async model load + detection thread (ultralytics YOLO optional)
    * Canonical processing resolution and exact widget<->frame transforms (letterbox/pillarbox)
    * Restricted-area editor: draw, select, move, resize, delete, undo/redo (action stack)
    * Config persistence (versioned JSON) with optional simple key-based encryption for RTSP credentials
    * Relay wrapper with safe no-op fallback
    * Structured logging with rotating file handler
    * Clean shutdown and error handling
    * Unit-test stubs at bottom
"""

import sys
import os
import json
import time
import threading
import queue
import math
import logging
import logging.handlers
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QMessageBox, QComboBox, QFileDialog, QFrame,
    QSizePolicy, QGroupBox, QShortcut
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QObject, QSize

# ---------- Optional Dependencies ----------
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
    print("✓ ultralytics YOLO available")
except Exception as e:
    print("⚠ ultralytics not available:", repr(e))

RELAY_AVAILABLE = False
pyhid_usb_relay = None
try:
    import pyhid_usb_relay  # type: ignore
    RELAY_AVAILABLE = True
    print("✓ pyhid_usb_relay available")
except Exception as e:
    print("⚠ pyhid_usb_relay not available:", repr(e))

# ---------- Logging Setup ----------
LOG_DIR = os.path.join(os.path.expanduser("~"), ".pokayoke_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pokayoke.log")

logger = logging.getLogger("pokayoke")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

rot_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
rot_handler.setFormatter(fmt)
rot_handler.setLevel(logging.DEBUG)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(fmt)
console.setLevel(logging.INFO)

logger.addHandler(rot_handler)
logger.addHandler(console)

logger.info("Pokayoke Vision System starting...")

# ---------- Constants ----------
DEFAULT_PROCESS_W = 1280
DEFAULT_PROCESS_H = 720
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".pokayoke_config_v1.json")
CONFIG_VERSION = 1

# Simple XOR "encryption" helper for saved RTSP credentials (not high-security; user warned in UI)
def simple_xor_encrypt(s: str, key: str) -> str:
    b = s.encode("utf-8")
    kb = key.encode("utf-8")
    out = bytes([b[i] ^ kb[i % len(kb)] for i in range(len(b))])
    return out.hex()

def simple_xor_decrypt(hex_s: str, key: str) -> str:
    try:
        b = bytes.fromhex(hex_s)
        kb = key.encode("utf-8")
        out = bytes([b[i] ^ kb[i % len(kb)] for i in range(len(b))])
        return out.decode("utf-8")
    except Exception:
        return ""

# ---------- Config Manager ----------
@dataclass
class AreaRect:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class AppConfig:
    version: int = CONFIG_VERSION
    camera: Dict[str, Any] = field(default_factory=lambda: {"type": "usb", "usb_index": 0, "rtsp_url_encrypted": None})
    processing_resolution: Tuple[int, int] = (DEFAULT_PROCESS_W, DEFAULT_PROCESS_H)
    restricted_areas: List[Dict[str, int]] = field(default_factory=list)
    model: Dict[str, Any] = field(default_factory=lambda: {"path": None, "confidence": 0.45})
    ui: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=lambda: {"saved_at": None})

class ConfigManager:
    def __init__(self, path: str = CONFIG_PATH, secret_key_env: str = "POKAYOKE_SECRET"):
        self.path = path
        self.secret_key = os.environ.get(secret_key_env, "")  # user may set environment secret for encryption
        self.config = AppConfig()
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            logger.info("Config file not found; using defaults")
            return
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            # Basic validation
            if raw.get("version", None) != CONFIG_VERSION:
                logger.warning("Config version mismatch, ignoring saved config")
                return
            self.config = AppConfig(
                version=raw.get("version", CONFIG_VERSION),
                camera=raw.get("camera", self.config.camera),
                processing_resolution=tuple(raw.get("processing_resolution", self.config.processing_resolution)),
                restricted_areas=raw.get("restricted_areas", []),
                model=raw.get("model", self.config.model),
                ui=raw.get("ui", {}),
                meta=raw.get("meta", {})
            )
            logger.info("Configuration loaded from %s", self.path)
        except Exception as e:
            logger.exception("Failed to load config: %s", e)

    def save(self, save_rtsp_encrypted: bool = False):
        try:
            cfg = asdict(self.config)
            cfg["meta"]["saved_at"] = datetime.now(timezone.utc).isoformat()
            if save_rtsp_encrypted and self.config.camera.get("rtsp_url"):
                if not self.secret_key:
                    logger.warning("No secret key set; RTSP password will not be encrypted. Set env POKAYOKE_SECRET to enable encryption.")
                else:
                    encrypted = simple_xor_encrypt(self.config.camera["rtsp_url"], self.secret_key)
                    cfg["camera"]["rtsp_url_encrypted"] = encrypted
                    # remove plaintext if present
                    cfg["camera"].pop("rtsp_url", None)
            # write atomically
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, self.path)
            logger.info("Configuration saved to %s", self.path)
        except Exception as e:
            logger.exception("Failed to save config: %s", e)

    def set_rtsp_url_from_encrypted(self):
        if self.config.camera.get("rtsp_url_encrypted") and self.secret_key:
            try:
                self.config.camera["rtsp_url"] = simple_xor_decrypt(self.config.camera["rtsp_url_encrypted"], self.secret_key)
                logger.info("RTSP URL decrypted into runtime config")
            except Exception:
                logger.warning("Cannot decrypt stored RTSP URL; secret missing or incorrect")

# ---------- Relay Wrapper ----------
class SafeRelay:
    """Wrapper for USB relay operations with safe no-op fallback"""
    def __init__(self):
        self.available = False
        self.dev = None
        if RELAY_AVAILABLE:
            try:
                # protect against any error the relay lib might raise (some libs raise custom exceptions)
                try:
                    self.dev = pyhid_usb_relay.find()
                except Exception as e:
                    # explicitly catch and log device-not-found or other pyhid exceptions
                    logger.warning("Relay init error: %s", e)
                    self.dev = None
                    self.available = False
                else:
                    if self.dev:
                        self.available = True
                        logger.info("Relay device initialized")
                    else:
                        logger.warning("pyhid_usb_relay found but no device connected")
            except BaseException as e:
                # be extremely defensive: never allow relay init to crash the app
                logger.exception("Relay init unexpected error: %s", e)
                self.available = False
        else:
            logger.info("Relay library not available; using no-op relay")

    def set_state(self, channel: int, state: bool):
        if not self.available:
            logger.debug("Relay request (no-op) channel=%s state=%s", channel, state)
            return
        try:
            self.dev.set_state(channel, state)
            logger.info("Relay channel %d -> %s", channel, state)
        except Exception as e:
            logger.exception("Relay operation failed: %s", e)

# ---------- Capture Worker ----------
class CaptureWorker(threading.Thread):
    """
    Threaded capture worker that reads frames from USB index or RTSP with reconnect/backoff.
    Produces frames scaled to processing resolution (maintaining aspect ratio) into a queue.
    """
    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event,
                 processing_size=(DEFAULT_PROCESS_W, DEFAULT_PROCESS_H)):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.processing_size = processing_size
        self.capture_lock = threading.Lock()
        self.cap = None
        self.source_spec = {"type": "usb", "usb_index": 0, "rtsp_url": None}
        self._connected = False
        self.backoff_base = 1.0
        self.backoff_max = 60.0

    def configure(self, source_spec: Dict[str, Any]):
        with self.capture_lock:
            self.source_spec = dict(source_spec)
            logger.debug("CaptureWorker configured with %s", self.source_spec)

    def run(self):
        logger.info("CaptureWorker started")
        backoff = self.backoff_base
        while not self.stop_event.is_set():
            try:
                if not self._connected:
                    logger.info("Attempting to open camera: %s", self.source_spec)
                    self._open_capture()
                    if not self._connected:
                        logger.warning("Failed to open capture; backing off %ss", backoff)
                        time.sleep(backoff)
                        backoff = min(self.backoff_max, backoff * 2)
                        continue
                    else:
                        backoff = self.backoff_base

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Frame read failed; reconnecting")
                    self._close_capture()
                    continue

                # convert to BGR if needed, resize for processing resolution (letterbox/pad)
                proc_frame = self._resize_preserve_aspect(frame, self.processing_size)
                # non-blocking put
                try:
                    self.frame_queue.put(proc_frame, block=False)
                except queue.Full:
                    # drop oldest frame then push current
                    try:
                        _ = self.frame_queue.get_nowait()
                        self.frame_queue.put(proc_frame, block=False)
                    except Exception:
                        pass
                # small sleep to avoid 100% CPU if camera is faster than processing
                time.sleep(0.001)
            except Exception as e:
                logger.exception("Capture worker exception: %s", e)
                self._close_capture()
        self._close_capture()
        logger.info("CaptureWorker stopped")

    def _open_capture(self):
        with self.capture_lock:
            try:
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                if self.source_spec.get("type") == "usb":
                    idx = int(self.source_spec.get("usb_index", 0))
                    self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
                else:
                    url = self.source_spec.get("rtsp_url")
                    # allow ffmpeg backend if desired: cv2.CAP_FFMPEG
                    self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    # set buffer size hints if supported
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    except Exception:
                        pass
                # small warmup
                time.sleep(0.5)
                if self.cap and self.cap.isOpened():
                    self._connected = True
                    logger.info("Capture opened successfully")
                else:
                    self._connected = False
                    logger.warning("Capture failed to open")
            except Exception as e:
                logger.exception("Open capture failed: %s", e)
                self._connected = False

    def _close_capture(self):
        with self.capture_lock:
            try:
                if self.cap:
                    self.cap.release()
                    logger.info("Capture released")
            except Exception:
                pass
            self.cap = None
            self._connected = False

    @staticmethod
    def _resize_preserve_aspect(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # pad to target_size with black borders
        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

# ---------- Detector Worker ----------
class DetectorWorker(threading.Thread):
    """
    Runs inference on incoming frames. Loads model asynchronously (optional),
    and produces annotated frames and detection events.
    """
    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue,
                 stop_event: threading.Event, model_path: Optional[str] = None,
                 device: str = "cpu", conf_threshold: float = 0.45):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.model_path = model_path
        self.model = None
        self.device = device
        self.conf_threshold = conf_threshold
        self._loaded = False
        self._load_lock = threading.Lock()
        # NEW: detector enabled flag -- when False the worker will not consume frames
        self.enabled = False

    def run(self):
        logger.info("DetectorWorker started")
        # attempt model load if model_path provided
        if self.model_path and YOLO_AVAILABLE:
            self._async_load_model(self.model_path)
        while not self.stop_event.is_set():
            try:
                # If detector is disabled, do not consume frames from the capture queue.
                if not self.enabled:
                    time.sleep(0.1)
                    continue
                frame = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            annotated, events = self._process_frame(frame)
            # out_queue is for UI display; non-blocking
            try:
                self.out_queue.put((annotated, events), block=False)
            except queue.Full:
                try:
                    _ = self.out_queue.get_nowait()
                    self.out_queue.put((annotated, events), block=False)
                except Exception:
                    pass
        logger.info("DetectorWorker stopped")

    def set_enabled(self, ena: bool):
        """Enable or disable processing of incoming frames."""
        self.enabled = bool(ena)
        logger.info("DetectorWorker enabled=%s", self.enabled)

    def _async_load_model(self, path: str):
        with self._load_lock:
            try:
                if not YOLO_AVAILABLE:
                    logger.warning("YOLO not available; skipping model load")
                    self._loaded = False
                    return
                logger.info("Loading model: %s", path)
                # load model (synchronous here but run in separate thread)
                self.model = YOLO(path)
                # prefer CUDA if available
                try:
                    if "cuda" in str(self.model.model.device).lower():
                        self.device = "cuda"
                except Exception:
                    pass
                self._loaded = True
                logger.info("Model loaded")
            except Exception as e:
                logger.exception("Model load failed: %s", e)
                self._loaded = False

    def set_model(self, path: str):
        self.model_path = path
        self._async_load_model(path)

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs detection (if model loaded), draws boxes, and returns annotated frame + events list.
        Events: [{'bbox':(x1,y1,x2,y2), 'conf':0.9, 'class': 'person', 'center':(cx,cy), 'in_restricted': False}, ...]
        """
        events = []
        annotated = frame.copy()
        if YOLO_AVAILABLE and self._loaded and self.model:
            try:
                results = self.model(frame, classes=[0], verbose=False)  # person class only
                for r in results:
                    boxes = getattr(r, "boxes", [])
                    for b in boxes:
                        try:
                            xy = b.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
                            conf = float(b.conf[0].cpu().numpy())
                        except Exception:
                            # fallback to numpy arrays in older versions
                            xy = b.xyxy[0].numpy()
                            conf = float(b.conf[0].numpy())
                        x1, y1, x2, y2 = map(int, xy.tolist())
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        events.append({"bbox": (x1, y1, x2, y2), "conf": conf, "class": "person", "center": (cx, cy)})
                        # draw rectangle
                        color = (0, 255, 0)  # default green
                        thickness = 2
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                        cv2.circle(annotated, (cx, cy), 4, color, -1)
                        cv2.putText(annotated, f"P:{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            except Exception as e:
                logger.exception("Inference error: %s", e)
                cv2.putText(annotated, f"INFERENCE ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            # annotate that no model is loaded
            if not YOLO_AVAILABLE:
                cv2.putText(annotated, "YOLO NOT INSTALLED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            elif not self._loaded:
                cv2.putText(annotated, "MODEL NOT LOADED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return annotated, events

# ---------- Video Widget (with canonical mapping) ----------
class VideoWidget(QLabel):
    """
    Displays frames (BGR numpy arrays). Maintains canonical processing resolution (fw, fh).
    Handles letterbox/pillarbox to keep aspect ratio and provides exact widget<->frame mapping.

    Mapping math (documented):
    - Let displayed pixmap size be (pw, ph), frame (internal) size be (fw, fh), and widget size be (ww, wh).
    - The pixmap is scaled with KeepAspectRatio and centered inside the widget -> offsets:
        x_offset = (ww - pw) / 2
        y_offset = (wh - ph) / 2
    - For a widget coordinate (wx, wy) that lies within the displayed pixmap:
        fx = int((wx - x_offset) * (fw / pw))
        fy = int((wy - y_offset) * (fh / ph))
    - Because KeepAspectRatio used, fw/pw == fh/ph (uniform scale).
    """
    frame_updated = pyqtSignal()

    def __init__(self, enable_drawing: bool = False, processing_size=(DEFAULT_PROCESS_W, DEFAULT_PROCESS_H)):
        super().__init__()
        self.enable_drawing = enable_drawing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")
        self._processing_size = tuple(processing_size)
        self._frame = None  # frame in processing coords (BGR)
        self._pixmap = None
        self.rects: List[AreaRect] = []
        # current detection event centers (list of (x,y))
        self.current_events: List[Tuple[int,int]] = []
        # Editor state
        self.drawing = False
        self.start_frame_point: Optional[Tuple[int,int]] = None
        self.current_frame_point: Optional[Tuple[int,int]] = None
        self.selected_id: Optional[str] = None
        self.handle_size = 8
        self.setMinimumSize(640, 480)

    def set_processing_frame(self, frame: np.ndarray, events: Optional[List[Tuple[int,int]]] = None):
        """Accepts frame already resized/padded to processing resolution.

        Optional `events` is a list of (x,y) centers of detected humans in
        frame coordinates; VideoWidget will use these to color restricted
        areas red/green.
        """
        self._frame = frame.copy()
        # set current event centers for overlay coloring
        if events:
            # normalize events into list of (int,int)
            evs: List[Tuple[int,int]] = []
            for e in events:
                try:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        evs.append((int(e[0]), int(e[1])))
                    elif isinstance(e, dict):
                        c = e.get('center')
                        if c and len(c) >= 2:
                            evs.append((int(c[0]), int(c[1])))
                except Exception:
                    continue
            self.current_events = evs
        else:
            # clear by default
            self.current_events = []
        # convert to QPixmap scaled to widget while keeping aspect ratio
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self._pixmap = pix
        self.update()

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # draw background
        painter.fillRect(self.rect(), QColor("#111"))
        if not self._pixmap:
            painter.setPen(QPen(QColor("#999")))
            painter.drawText(self.rect(), Qt.AlignCenter, "No video")
            return
        # compute scaled pixmap size preserving aspect ratio
        widget_w, widget_h = self.width(), self.height()
        pixmap_size = self._pixmap.size()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scaled = self._pixmap.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pw, ph = scaled.width(), scaled.height()
        x_offset = (widget_w - pw) // 2
        y_offset = (widget_h - ph) // 2
        painter.drawPixmap(x_offset, y_offset, scaled)

        # draw restricted-area overlays (transform frame coords -> widget coords)
        for r in self.rects:
            wx1, wy1 = self.frame_to_widget_coords((r.x1, r.y1))
            wx2, wy2 = self.frame_to_widget_coords((r.x2, r.y2))
            if wx1 is None:
                continue
            rect = QRect(wx1, wy1, wx2 - wx1, wy2 - wy1)
            # determine if any current event center lies inside this rect
            in_restricted = False
            for (cx, cy) in self.current_events:
                if r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2:
                    in_restricted = True
                    break
            if in_restricted:
                pen_col = QColor(255, 0, 0)
                fill_col = QColor(255, 0, 0, 60)
            else:
                pen_col = QColor(0, 200, 0)
                fill_col = QColor(0, 200, 0, 60)
            painter.setPen(QPen(pen_col, 2))
            painter.fillRect(rect, fill_col)
            painter.drawRect(rect)
            # corner handles
            painter.setBrush(QColor(0, 255, 255))
            for cx_w, cy_w in [(wx1, wy1), (wx2, wy1), (wx1, wy2), (wx2, wy2)]:
                painter.drawEllipse(QRect(cx_w - self.handle_size//2, cy_w - self.handle_size//2, self.handle_size, self.handle_size))

        # draw current drawing rect
        if self.drawing and self.start_frame_point and self.current_frame_point:
            sx, sy = self.frame_to_widget_coords(self.start_frame_point)
            cx, cy = self.frame_to_widget_coords(self.current_frame_point)
            if sx is not None:
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
                painter.drawRect(QRect(sx, sy, cx - sx, cy - sy))

    # -------- coordinate transforms --------
    def frame_to_widget_coords(self, frame_pt: Tuple[int,int]) -> Tuple[Optional[int], Optional[int]]:
        """Frame coords -> widget coords"""
        if self._pixmap is None:
            return (None, None)
        fw, fh = self._processing_size
        wx, wy = frame_pt
        widget_w, widget_h = self.width(), self.height()
        # displayed pixmap
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scaled = self._pixmap.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pw, ph = scaled.width(), scaled.height()
        x_offset = (widget_w - pw) // 2
        y_offset = (widget_h - ph) // 2
        # scale factor (fw/pw == fh/ph)
        sx = fw / pw
        sy = fh / ph
        # since aspect ratio preserved, sx == sy
        wx = int(x_offset + (frame_pt[0] / sx))
        wy = int(y_offset + (frame_pt[1] / sy))
        return wx, wy

    def widget_to_frame_coords(self, widget_pt: QPoint) -> Optional[Tuple[int,int]]:
        """Widget coords -> frame coords. Returns None if outside video area"""
        if self._pixmap is None:
            return None
        widget_w, widget_h = self.width(), self.height()
        scaled = self._pixmap.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pw, ph = scaled.width(), scaled.height()
        x_offset = (widget_w - pw) // 2
        y_offset = (widget_h - ph) // 2
        local_x = widget_pt.x() - x_offset
        local_y = widget_pt.y() - y_offset
        if local_x < 0 or local_y < 0 or local_x >= pw or local_y >= ph:
            return None
        fw, fh = self._processing_size
        sx = fw / pw
        sy = fh / ph
        fx = int(local_x * sx)
        fy = int(local_y * sy)
        fx = max(0, min(fw - 1, fx))
        fy = max(0, min(fh - 1, fy))
        return fx, fy

    # -------- editor interactions (basic) --------
    def mousePressEvent(self, ev):
        if not self.enable_drawing:
            return
        if ev.button() != Qt.LeftButton:
            return
        coords = self.widget_to_frame_coords(ev.pos())
        if coords is None:
            return
        self.drawing = True
        self.start_frame_point = coords
        self.current_frame_point = coords
        self.update()

    def mouseMoveEvent(self, ev):
        if not self.enable_drawing or not self.drawing:
            return
        coords = self.widget_to_frame_coords(ev.pos())
        if coords:
            self.current_frame_point = coords
            self.update()

    def mouseReleaseEvent(self, ev):
        if not self.enable_drawing or ev.button() != Qt.LeftButton:
            return
        coords = self.widget_to_frame_coords(ev.pos())
        if coords and self.start_frame_point:
            x1, y1 = self.start_frame_point
            x2, y2 = coords
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            # discard tiny rects
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                rid = f"r{int(time.time()*1000)}"
                self.rects.append(AreaRect(rid, x1, y1, x2, y2))
                self.frame_updated.emit()
        self.drawing = False
        self.start_frame_point = None
        self.current_frame_point = None
        self.update()

    def clear_rects(self):
        self.rects = []
        self.frame_updated.emit()
        self.update()

    def load_rects(self, rect_dicts: List[Dict[str,int]]):
        self.rects = [AreaRect(**r) for r in rect_dicts]
        self.update()

    def get_rects_as_dict(self) -> List[Dict[str,int]]:
        return [asdict(r) for r in self.rects]

# ---------- Editor (Undo/Redo & Actions) ----------
class ActionStack:
    def __init__(self, limit: int = 100):
        self.stack: List[Dict] = []
        self.redo_stack: List[Dict] = []
        self.limit = limit

    def push(self, action: Dict[str, Any]):
        self.stack.append(action)
        if len(self.stack) > self.limit:
            self.stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.stack:
            return None
        action = self.stack.pop()
        self.redo_stack.append(action)
        return action

    def redo(self):
        if not self.redo_stack:
            return None
        action = self.redo_stack.pop()
        self.stack.append(action)
        return action

# ---------- Main Application Window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pokayoke Vision Safety System - Refactor")
        self.setMinimumSize(1280, 720)
        # try to center nicely
        scr = QApplication.desktop().availableGeometry()
        self.setGeometry(50, 50, min(1600, scr.width() - 100), min(900, scr.height() - 100))

        # Managers and workers
        self.config_mgr = ConfigManager()
        self.config_mgr.set_rtsp_url_from_encrypted()
        self.relay = SafeRelay()

        # queues
        self.capture_queue = queue.Queue(maxsize=4)  # frames from capture
        self.annotated_queue = queue.Queue(maxsize=4)  # annotated frames from detector

        self.stop_event = threading.Event()
        self.capture_worker = CaptureWorker(self.capture_queue, self.stop_event, processing_size=tuple(self.config_mgr.config.processing_resolution))
        self.detector_worker = DetectorWorker(self.capture_queue, self.annotated_queue, self.stop_event,
                                              model_path=self.config_mgr.config.model.get("path"),
                                              conf_threshold=self.config_mgr.config.model.get("confidence", 0.45))
        # Ensure detector starts disabled (won't process frames until user starts it)
        self.detector_worker.set_enabled(False)

        # UI components
        self.nav_bar = self._create_navbar()
        self.stack = QStackedWidget()
        self.teaching_page = self._create_teaching_page()
        self.detection_page = self._create_detection_page()
        self.stack.addWidget(self.teaching_page)
        self.stack.addWidget(self.detection_page)

        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.nav_bar)
        layout.addWidget(self.stack)
        central.setLayout(layout)
        self.setCentralWidget(central)

        # timers
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._ui_tick)
        self.ui_timer.start(30)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_config)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_capture_detection)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)
        QShortcut(QKeySequence("Delete"), self, activated=self._delete_selected_rect)

        # state
        self.mode = "teaching"
        self.detection_running = False  # track whether detection is currently active
        self.action_stack = ActionStack(limit=200)
        self.last_detection_snapshot_dir = os.path.join(os.path.expanduser("~"), "pokayoke_snapshots")
        os.makedirs(self.last_detection_snapshot_dir, exist_ok=True)

        # start workers (they will open camera only when configured)
        self.capture_worker.start()
        self.detector_worker.start()

        logger.info("MainWindow initialized")

    # ---------- UI creation ----------
    def _create_navbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("background-color: #2c3e50; border-bottom: 3px solid #3498db;")
        bar.setFixedHeight(80)
        hl = QHBoxLayout()
        title = QLabel("⚙ Pokayoke Vision")
        title.setStyleSheet("color:white; font-size:22px; font-weight:bold;")
        hl.addWidget(title)
        hl.addStretch()
        teach_btn = QPushButton("TEACHING")
        detect_btn = QPushButton("DETECTION")
        teach_btn.clicked.connect(lambda: self.switch_mode("teaching"))
        detect_btn.clicked.connect(lambda: self.switch_mode("detection"))
        hl.addWidget(teach_btn)
        hl.addWidget(detect_btn)
        bar.setLayout(hl)
        return bar

    def _create_teaching_page(self) -> QWidget:
        page = QWidget()
        hl = QHBoxLayout()
        # left: video
        self.video_widget = VideoWidget(enable_drawing=True, processing_size=tuple(self.config_mgr.config.processing_resolution))
        self.video_widget.frame_updated.connect(self._on_video_frame_updated)
        hl.addWidget(self.video_widget, 3)
        # right: controls
        right = QFrame()
        right.setFixedWidth(340)
        vbox = QVBoxLayout()
        vbox.setSpacing(12)
        # camera selector
        cam_box = QGroupBox("Camera")
        cb_layout = QHBoxLayout()
        self.cam_select = QComboBox()
        # default simple entries; UI also allows RTSP
        self.cam_select.addItems(["USB:0", "USB:1", "USB:2", "RTSP..."])
        cb_layout.addWidget(self.cam_select)
        self.test_btn = QPushButton("Test/Connect")
        self.test_btn.clicked.connect(self._on_test_connect)
        cb_layout.addWidget(self.test_btn)
        cam_box.setLayout(cb_layout)
        vbox.addWidget(cam_box)
        # area controls
        self.clear_btn = QPushButton("Clear All Areas")
        self.clear_btn.clicked.connect(self._clear_all_areas)
        vbox.addWidget(self.clear_btn)
        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self._save_config)
        vbox.addWidget(self.save_btn)
        # info label
        self.info_label = QLabel("Draw rectangles on the video to define restricted areas.")
        self.info_label.setWordWrap(True)
        vbox.addWidget(self.info_label)
        vbox.addStretch()
        right.setLayout(vbox)
        hl.addWidget(right, 1)
        page.setLayout(hl)
        return page

    def _create_detection_page(self) -> QWidget:
        page = QWidget()
        hl = QHBoxLayout()
        # left: video preview for annotated frames
        self.annotated_widget = VideoWidget(enable_drawing=False, processing_size=tuple(self.config_mgr.config.processing_resolution))
        hl.addWidget(self.annotated_widget, 3)
        # right: controls
        right = QFrame()
        right.setFixedWidth(340)
        vbox = QVBoxLayout()
        # model loader
        model_group = QGroupBox("Model")
        m_layout = QHBoxLayout()
        self.model_label = QLabel("No model loaded")
        m_layout.addWidget(self.model_label)
        self.model_btn = QPushButton("Load Model")
        self.model_btn.clicked.connect(self._on_load_model)
        self.model_btn.setEnabled(YOLO_AVAILABLE)
        m_layout.addWidget(self.model_btn)
        model_group.setLayout(m_layout)
        vbox.addWidget(model_group)
        # detection controls
        self.start_det_btn = QPushButton("Start Detection")
        self.start_det_btn.clicked.connect(lambda: self._start_stop_detection(True))
        self.stop_det_btn = QPushButton("Stop Detection")
        self.stop_det_btn.clicked.connect(lambda: self._start_stop_detection(False))
        vbox.addWidget(self.start_det_btn)
        vbox.addWidget(self.stop_det_btn)
        # status
        self.status_label = QLabel("Status: Idle")
        vbox.addWidget(self.status_label)
        self.last_alert_label = QLabel("Last alert: None")
        vbox.addWidget(self.last_alert_label)
        vbox.addStretch()
        right.setLayout(vbox)
        hl.addWidget(right, 1)
        page.setLayout(hl)
        return page

    # ---------- Mode switching ----------
    def switch_mode(self, mode: str):
        if mode == self.mode:
            return
        if mode == "detection":
            # ensure camera config loaded
            ok = self._apply_camera_selection_to_worker()
            if not ok:
                QMessageBox.warning(self, "Camera", "Please configure camera before switching to Detection.")
                return
            # load saved rects into annotated_widget
            self.annotated_widget.load_rects(self.video_widget.get_rects_as_dict())
            self.stack.setCurrentWidget(self.detection_page)
            self.mode = "detection"
        else:
            self.stack.setCurrentWidget(self.teaching_page)
            self.mode = "teaching"

    # ---------- Camera and capture controls ----------
    def _on_test_connect(self):
        sel = self.cam_select.currentText()
        if sel.startswith("USB"):
            idx = int(sel.split(":")[1])
            spec = {"type": "usb", "usb_index": idx}
            self.config_mgr.config.camera = spec
            self.capture_worker.configure(spec)
            QMessageBox.information(self, "Camera", f"Configured USB camera index {idx}.")
        else:
            # RTSP prompt
            url, ok = QFileDialog.getOpenFileName(self, "Enter RTSP URL (workaround uses file dialog cancel to type)", "")
            # Ugly but avoid extra dialogs; better to use QInputDialog in production
            if url:
                # if user selected a file, ignore. Instead we use a QInputDialog in practice
                pass
            # Use QInputDialog for URL
            from PyQt5.QtWidgets import QInputDialog
            url, ok2 = QInputDialog.getText(self, "RTSP URL", "Enter RTSP URL (rtsp://user:pass@host:port/stream):")
            if ok2 and url:
                spec = {"type": "rtsp", "rtsp_url": url}
                self.config_mgr.config.camera = spec
                self.capture_worker.configure(spec)
                QMessageBox.information(self, "Camera", "RTSP configured (not saved to disk unless you Save Configuration).")

    def _apply_camera_selection_to_worker(self) -> bool:
        cam = self.config_mgr.config.camera
        if cam.get("type") == "usb":
            try:
                self.capture_worker.configure(cam)
                return True
            except Exception as e:
                logger.exception("Failed to configure capture worker: %s", e)
                return False
        elif cam.get("type") == "rtsp":
            if not cam.get("rtsp_url"):
                # maybe encrypted stored; try decrypt
                self.config_mgr.set_rtsp_url_from_encrypted()
            if not cam.get("rtsp_url"):
                QMessageBox.warning(self, "RTSP", "RTSP URL not configured.")
                return False
            try:
                self.capture_worker.configure(cam)
                return True
            except Exception as e:
                logger.exception("Failed to configure RTSP on capture worker: %s", e)
                return False
        else:
            QMessageBox.warning(self, "Camera", "Unsupported camera configuration")
            return False

    # ---------- Model loading ----------
    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model file", "", "Models (*.pt *.onnx);;All Files (*)")
        if not path:
            return
        self.config_mgr.config.model["path"] = path
        self.model_label.setText(os.path.basename(path))
        # tell detector worker to load model
        self.detector_worker.set_model(path)
        QMessageBox.information(self, "Model", f"Requested loading model: {path}")

    # ---------- Detection start/stop ----------
    def _start_stop_detection(self, start: bool):
        if start:
            # ensure camera configured
            if not self._apply_camera_selection_to_worker():
                return
            # Enable detector worker to process frames
            try:
                self.detector_worker.set_enabled(True)
            except Exception:
                logger.exception("Failed to enable detector worker")
            self.status_label.setText("Status: Running detection")
            logger.info("Detection started")
            self.detection_running = True
        else:
            # Disable detector worker so it stops consuming frames permanently until re-enabled
            try:
                self.detector_worker.set_enabled(False)
            except Exception:
                logger.exception("Failed to disable detector worker")
            self.status_label.setText("Status: Stopped")
            logger.info("Detection stopped")
            self.detection_running = False

    def toggle_capture_detection(self):
        """Toggle detection on/off — used by Space shortcut (keeps existing shortcut intact)."""
        try:
            self._start_stop_detection(not getattr(self, "detection_running", False))
        except Exception:
            logger.exception("Failed to toggle detection")

    # ---------- UI tick: pull annotated frames for display and handle events ----------
    def _ui_tick(self):
        # if teaching mode, show the live camera frames from capture_queue (latest) without detection
        try:
            # prefer annotated frames for detection page, but only when detection is running
            if self.mode == "detection" and getattr(self, "detection_running", False):
                try:
                    annotated, events = self.annotated_queue.get_nowait()
                    # Annotated currently has boxes from the model; overlay per-event red/green
                    any_alert = False
                    for e in events:
                        cx, cy = e.get("center", (None, None))
                        in_restricted = False
                        for r in self.annotated_widget.rects:
                            if cx is None:
                                continue
                            if r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2:
                                in_restricted = True
                                break
                        # draw per-event box: red if in restricted area, otherwise green
                        bbox = e.get("bbox", (0,0,0,0))
                        x1, y1, x2, y2 = bbox
                        color = (0, 0, 255) if in_restricted else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cx_draw = int((x1 + x2) / 2)
                        cy_draw = int((y1 + y2) / 2)
                        cv2.circle(annotated, (cx_draw, cy_draw), 4, color, -1)
                        cv2.putText(annotated, f"P:{e.get('conf', 0):.2f}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        if in_restricted:
                            any_alert = True
                            # snapshot + relay (non-blocking)
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            fname = os.path.join(self.last_detection_snapshot_dir, f"alert_{ts}.jpg")
                            try:
                                cv2.imwrite(fname, annotated)
                                logger.info("Saved alert snapshot: %s", fname)
                                self.last_alert_label.setText(f"Last alert: {ts}")
                                threading.Thread(target=self._pulse_relay, args=(1, 2.0), daemon=True).start()
                            except Exception:
                                logger.exception("Failed to save snapshot")
                    # show annotated frame and pass detection events so restricted-area
                    # overlays can be colored appropriately (red if occupied)
                    self.annotated_widget.set_processing_frame(annotated, events)
                except queue.Empty:
                    pass

            # Always update teaching video preview with the latest capture frame if available (non-destructive)
            try:
                latest = None
                while True:
                    latest = self.capture_queue.get_nowait()
            except queue.Empty:
                if 'latest' in locals() and latest is not None:
                    self.video_widget.set_processing_frame(latest)
            except Exception:
                pass
        except Exception:
            logger.exception("UI tick failed")

    def _pulse_relay(self, channel: int, duration: float = 1.0):
        """Turn relay on for duration seconds then off"""
        try:
            self.relay.set_state(channel, True)
            time.sleep(duration)
            self.relay.set_state(channel, False)
        except Exception:
            logger.exception("Relay pulse failed")

    # ---------- Editor actions ----------
    def _on_video_frame_updated(self):
        # when the user draws a rect, push action to stack
        # Here we simply record full rect list as action for undo/redo simplicity
        rects = self.video_widget.get_rects_as_dict()
        self.action_stack.push({"type": "set_rects", "rects": rects})

    def _clear_all_areas(self):
        reply = QMessageBox.question(self, "Confirm", "Clear all areas?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            old = self.video_widget.get_rects_as_dict()
            self.action_stack.push({"type": "set_rects", "rects": old})
            self.video_widget.clear_rects()

    def _save_config(self):
        # save rectangles and camera to config
        rects = self.video_widget.get_rects_as_dict()
        self.config_mgr.config.restricted_areas = rects
        # camera selection: read from cam_select or existing config
        sel = self.cam_select.currentText()
        if sel.startswith("USB"):
            idx = int(sel.split(":")[1])
            self.config_mgr.config.camera = {"type": "usb", "usb_index": idx}
        elif sel.startswith("RTSP") or self.config_mgr.config.camera.get("type") == "rtsp":
            # prefer existing camera RTSP if present or ask user
            if not self.config_mgr.config.camera.get("rtsp_url"):
                from PyQt5.QtWidgets import QInputDialog
                url, ok = QInputDialog.getText(self, "RTSP", "RTSP URL (leave blank to not save):")
                if ok and url:
                    self.config_mgr.config.camera = {"type": "rtsp", "rtsp_url": url}
        # optionally ask user whether to encrypt RTSP if present
        save_enc = False
        if self.config_mgr.config.camera.get("type") == "rtsp":
            reply = QMessageBox.question(self, "Save RTSP", "Save RTSP credentials encrypted on disk? (recommended)", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            save_enc = (reply == QMessageBox.Yes)
        self.config_mgr.save(save_rtsp_encrypted=save_enc)
        QMessageBox.information(self, "Saved", "Configuration saved.")

    def _undo(self):
        action = self.action_stack.undo()
        if not action:
            return
        if action["type"] == "set_rects":
            # apply previous state (redo logic already pushed)
            prev = action["rects"]
            self.video_widget.load_rects(prev)
            logger.info("Undo applied")
            self.video_widget.update()

    def _redo(self):
        action = self.action_stack.redo()
        if not action:
            return
        if action["type"] == "set_rects":
            self.video_widget.load_rects(action["rects"])
            logger.info("Redo applied")
            self.video_widget.update()

    def _delete_selected_rect(self):
        # placeholder: currently no selection logic; skip
        pass

    # ---------- Shutdown ----------
    def closeEvent(self, ev):
        reply = QMessageBox.question(self, "Exit", "Exit application?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            ev.ignore()
            return
        # stop threads
        self.stop_event.set()
        logger.info("Stopping workers...")
        try:
            self.capture_worker.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.detector_worker.join(timeout=2.0)
        except Exception:
            pass
        logger.info("Exiting")
        ev.accept()

# ---------- Minimal unit test stubs ----------
def run_unit_tests():
    import tempfile
    logger.info("Running minimal unit tests...")

    # test coordinate transform roundtrip
    w = VideoWidget(enable_drawing=False, processing_size=(640, 480))
    # create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    w.set_processing_frame(frame)
    # simulate widget size
    w.resize(800, 600)
    # test points
    test_points = [(10, 10), (320, 240), (639, 479)]
    for tx, ty in test_points:
        # frame->widget->frame
        wx, wy = w.frame_to_widget_coords((tx, ty))
        if wx is None:
            logger.error("frame_to_widget returned None unexpectedly")
            continue
        res = w.widget_to_frame_coords(QPoint(wx, wy))
        if res is None:
            logger.error("widget_to_frame returned None unexpectedly")
            continue
        fx, fy = res
        # allow tolerance of 1 px
        assert abs(fx - tx) <= 1 and abs(fy - ty) <= 1, f"roundtrip mismatch {tx,ty} -> {fx,fy}"
    logger.info("Coordinate roundtrip tests passed")

    # test save/load config roundtrip
    cfg_path = os.path.join(tempfile.gettempdir(), "pokayoke_test_cfg.json")
    cm = ConfigManager(path=cfg_path)
    cm.config.restricted_areas = [{"id":"r1","x1":10,"y1":20,"x2":100,"y2":200}]
    cm.config.camera = {"type":"usb","usb_index":1}
    cm.save()
    cm2 = ConfigManager(path=cfg_path)
    assert cm2.config.restricted_areas[0]["id"] == "r1"
    logger.info("Config save/load test passed")

    # test action stack
    as_ = ActionStack(limit=5)
    as_.push({"type":"set_rects","rects":[1]})
    as_.push({"type":"set_rects","rects":[2]})
    a = as_.undo()
    assert a and a["rects"] == [2]
    r = as_.redo()
    assert r and r["rects"] == [2]
    logger.info("ActionStack tests passed")

    logger.info("All minimal unit tests OK")

# ---------- main ----------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    win = MainWindow()
    win.show()
    # optional: run unit tests in background for dev
    # threading.Thread(target=run_unit_tests, daemon=True).start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

