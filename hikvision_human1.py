"""
Pokayoke Vision Safety System - Refactored
Architecture follows hikvision_3_v7.py patterns with improvements:
- QThread-based camera and detection workers
- Clean separation of teaching and detection modes
- Signal-based UI updates (no busy loops)
- Robust reconnection logic
- Performance optimizations
"""

import sys
import os
import json
import time
import threading
import queue
import logging
import logging.handlers
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

# Early YOLO import
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[WARN] YOLO not available: {e}", file=sys.stderr)

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QMessageBox, QComboBox, QFileDialog, QFrame,
    QGroupBox, QLineEdit, QTabWidget, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPolygon
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QThread, QSize

# Optional relay support
RELAY_AVAILABLE = False
try:
    import pyhid_usb_relay
    RELAY_AVAILABLE = True
    print("✓ Relay support available")
except Exception:
    print("⚠ Relay support not available")

# ==================== LOGGING SETUP ====================
LOG_DIR = os.path.join(os.path.expanduser("~"), ".pokayoke_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pokayoke.log")

logger = logging.getLogger("pokayoke")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

rot_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
rot_handler.setFormatter(fmt)
rot_handler.setLevel(logging.DEBUG)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(fmt)
console.setLevel(logging.INFO)

logger.addHandler(rot_handler)
logger.addHandler(console)

logger.info("Pokayoke Vision System starting...")

# ==================== CONFIGURATION ====================
DEFAULT_PROCESS_W = 1280
DEFAULT_PROCESS_H = 720
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".pokayoke_config_v1.json")
CONFIG_VERSION = 1
DEFAULT_RTSP_URL = "rtsp://admin:Pass_123@192.168.1.64:554/stream"

CONFIG = {
    "camera": {
        "rtsp_timeout_ms": 5000,
        "buffer_size": 1,
        "max_reconnect_attempts": 10,
        "reconnect_backoff_max": 60
    },
    "detection": {
        "confidence_threshold": 0.45,
        "processing_resolution": (DEFAULT_PROCESS_W, DEFAULT_PROCESS_H)
    }
}

# ==================== HELPER CLASSES ====================
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
    camera: Dict[str, Any] = field(default_factory=lambda: {"type": "usb", "usb_index": 0, "rtsp_url": None})
    processing_resolution: Tuple[int, int] = (DEFAULT_PROCESS_W, DEFAULT_PROCESS_H)
    restricted_areas: List[Dict[str, int]] = field(default_factory=list)
    model: Dict[str, Any] = field(default_factory=lambda: {"path": None, "confidence": 0.45})
    ui: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=lambda: {"saved_at": None})

class ConfigManager:
    def __init__(self, path: str = CONFIG_PATH):
        self.path = path
        self.config = AppConfig()
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            logger.info("Config file not found; using defaults")
            return
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            if raw.get("version", None) != CONFIG_VERSION:
                logger.warning("Config version mismatch")
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
            logger.info("Configuration loaded")
        except Exception as e:
            logger.exception("Failed to load config: %s", e)

    def save(self):
        try:
            cfg = asdict(self.config)
            cfg["meta"]["saved_at"] = datetime.now(timezone.utc).isoformat()
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, self.path)
            logger.info("Configuration saved")
        except Exception as e:
            logger.exception("Failed to save config: %s", e)

class FrameBuffer:
    """Thread-safe frame buffer with automatic overflow handling"""
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        
    def put(self, frame, block=False):
        try:
            self.queue.put(frame, block=block, timeout=0.001)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put(frame, block=False)
            except:
                pass
    
    def get(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear(self):
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

class SafeRelay:
    """Wrapper for USB relay with safe fallback"""
    def __init__(self):
        self.available = False
        self.dev = None
        if RELAY_AVAILABLE:
            try:
                self.dev = pyhid_usb_relay.find()
                if self.dev:
                    self.available = True
                    logger.info("Relay device initialized")
            except Exception as e:
                logger.warning("Relay init error: %s", e)

    def set_state(self, channel: int, state: bool):
        if not self.available:
            logger.debug("Relay request (no-op) channel=%s state=%s", channel, state)
            return
        try:
            self.dev.set_state(channel, state)
            logger.info("Relay channel %d -> %s", channel, state)
        except Exception as e:
            logger.exception("Relay operation failed: %s", e)

# ==================== CAMERA THREAD ====================
class CameraThread(QThread):
    """QThread-based camera worker following hikvision_3_v7 architecture"""
    frame_ready = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, camera_source, processing_size=(DEFAULT_PROCESS_W, DEFAULT_PROCESS_H)):
        super().__init__()
        self.camera_source = camera_source
        self.processing_size = processing_size
        self.running = False
        self.camera = None
        self.frame_buffer = FrameBuffer(maxsize=5)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = CONFIG["camera"]["max_reconnect_attempts"]
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.is_ip_camera = isinstance(camera_source, str)
        
    def run(self):
        self.running = True
        camera_type = "IP Camera (RTSP)" if self.is_ip_camera else f"USB Camera {self.camera_source}"
        logger.info(f"Camera thread started for {camera_type}")
        
        backoff = 1.0
        
        while self.running:
            try:
                if self.camera is None or not self.camera.isOpened():
                    if not self.connect_camera():
                        logger.warning(f"Failed to connect; backing off {backoff}s")
                        time.sleep(backoff)
                        backoff = min(CONFIG["camera"]["reconnect_backoff_max"], backoff * 2)
                        continue
                    else:
                        backoff = 1.0
                
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    logger.warning("Frame read failed; reconnecting")
                    self.reconnect_camera()
                    continue
                
                # Resize to processing resolution
                proc_frame = self._resize_preserve_aspect(frame, self.processing_size)
                
                # Buffered storage
                self.frame_buffer.put(proc_frame.copy(), block=False)
                
                # Emit to UI
                self.frame_ready.emit(proc_frame)
                
                self.reconnect_attempts = 0
                self.frame_count += 1
                
                # FPS logging
                current_time = time.time()
                if current_time - self.last_frame_time >= 5.0:
                    fps = self.frame_count / (current_time - self.last_frame_time)
                    logger.debug(f"Camera FPS: {fps:.1f}")
                    self.frame_count = 0
                    self.last_frame_time = current_time
                
                time.sleep(0.001)  # Small yield
                    
            except Exception as e:
                logger.error(f"Camera thread error: {e}")
                self.error_signal.emit(str(e))
                self.reconnect_camera()
                time.sleep(1)
        
        self.cleanup()
        logger.info("Camera thread stopped")
    
    def connect_camera(self):
        """Connect to camera with proper timeout settings"""
        try:
            if self.camera is not None:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
            
            camera_desc = f"IP camera (RTSP)" if self.is_ip_camera else f"USB camera {self.camera_source}"
            logger.info(f"Connecting to {camera_desc}")
            
            if self.is_ip_camera:
                self.camera = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG["camera"]["buffer_size"])
                self.camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CONFIG["camera"]["rtsp_timeout_ms"])
                self.camera.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, CONFIG["camera"]["rtsp_timeout_ms"])
            else:
                self.camera = cv2.VideoCapture(self.camera_source, cv2.CAP_ANY)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG["camera"]["buffer_size"])
            
            time.sleep(0.5)
            
            if self.camera and self.camera.isOpened():
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    logger.info(f"{camera_desc} connected - Frame size: {test_frame.shape}")
                    self.status_signal.emit(f"Camera Connected ({camera_desc})")
                    return True
                else:
                    logger.warning(f"{camera_desc} opened but cannot read frames")
                    if self.camera:
                        self.camera.release()
                        self.camera = None
            else:
                logger.warning(f"{camera_desc} failed to open")
                    
        except Exception as e:
            logger.error(f"Camera connection error: {e}")
        
        return False
    
    def reconnect_camera(self):
        """Handle camera reconnection with backoff"""
        self.reconnect_attempts += 1
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        
        delay = min(2 ** self.reconnect_attempts, CONFIG["camera"]["reconnect_backoff_max"])
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.status_signal.emit(f"Reconnecting in {delay}s... (attempt {self.reconnect_attempts})")
            time.sleep(delay)
        else:
            logger.warning("Max reconnect attempts reached, resetting counter")
            self.reconnect_attempts = 0
            time.sleep(60)
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        logger.info("Camera resources cleaned up")
    
    def stop(self):
        """Stop the camera thread gracefully"""
        self.running = False
        self.wait(3000)
    
    @staticmethod
    def _resize_preserve_aspect(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame preserving aspect ratio with letterbox/pillarbox padding"""
        h, w = frame.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

# ==================== DETECTION THREAD ====================
class DetectionThread(QThread):
    """QThread-based detection worker following hikvision_3_v7 architecture"""
    detection_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, model_path, boundaries, conf_threshold=0.45):
        super().__init__()
        self.model_path = model_path
        self.boundaries = boundaries
        self.conf_threshold = conf_threshold
        self.running = False
        self.model = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.fps_deque = deque(maxlen=30)
        self.detection_count = 0
        
        # Fallback HOG detector
        self.hog = None
        if not YOLO_AVAILABLE:
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                logger.info("Using OpenCV HOG fallback detector")
            except Exception as e:
                logger.error(f"Failed to init HOG: {e}")
        
    def run(self):
        self.running = True
        logger.info("Detection thread starting")
        
        # Load model
        if self.model_path and YOLO_AVAILABLE:
            try:
                logger.info(f"Loading YOLO model: {self.model_path}")
                self.model = YOLO(self.model_path)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Model load failed: {e}")
                self.error_signal.emit(f"Model load failed: {str(e)}")
        
        loop_count = 0
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                if frame is not None:
                    loop_count += 1
                    if loop_count == 1:
                        logger.info("First frame received in detection thread")
                    
                    start_time = time.time()
                    result = self.process_frame(frame)
                    inference_time = time.time() - start_time
                    
                    self.fps_deque.append(1.0 / inference_time if inference_time > 0 else 0)
                    avg_fps = sum(self.fps_deque) / len(self.fps_deque)
                    
                    self.detection_ready.emit(result)
                    self.fps_signal.emit(avg_fps)
                    self.detection_count += 1
                    
                    if self.detection_count == 1:
                        logger.info("First detection completed")
                    
                    if self.detection_count % 100 == 0:
                        logger.debug(f"Detection count: {self.detection_count}, FPS: {avg_fps:.1f}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection error: {e}")
                self.error_signal.emit(str(e))
                time.sleep(0.1)
        
        logger.info("Detection thread stopped")
    
    def process_frame(self, frame):
        """Process frame and return annotated result with detection events"""
        events = []
        annotated = frame.copy()
        
        # YOLO detection if available
        if YOLO_AVAILABLE and self.model:
            try:
                results = self.model(frame, classes=[0], verbose=False, conf=self.conf_threshold)
                for r in results:
                    boxes = getattr(r, "boxes", [])
                    for b in boxes:
                        try:
                            xy = b.xyxy[0].cpu().numpy()
                            conf = float(b.conf[0].cpu().numpy())
                        except:
                            xy = b.xyxy[0].numpy()
                            conf = float(b.conf[0].numpy())
                        
                        x1, y1, x2, y2 = map(int, xy.tolist())
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        events.append({"bbox": (x1, y1, x2, y2), "conf": conf, "class": "person", "center": (cx, cy)})
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")
        
        # HOG fallback
        elif not YOLO_AVAILABLE and self.hog:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, weights = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
                for i, (x, y, w, h) in enumerate(rects):
                    conf = float(weights[i]) if (weights is not None and len(weights) > i) else 0.5
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    events.append({"bbox": (x1, y1, x2, y2), "conf": conf, "class": "person", "center": (cx, cy)})
            except Exception as e:
                logger.error(f"HOG detection error: {e}")
        
        # Draw restricted areas with color based on occupancy
        for r_dict in self.boundaries:
            r = AreaRect(**r_dict)
            in_restricted = False
            for e in events:
                cx, cy = e.get('center', (None, None))
                if cx and r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2:
                    in_restricted = True
                    break
            
            color = (0, 0, 255) if in_restricted else (0, 255, 0)
            cv2.rectangle(annotated, (r.x1, r.y1), (r.x2, r.y2), color, 2)
            cv2.putText(annotated, "RESTRICTED", (r.x1, r.y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw detections
        for e in events:
            x1, y1, x2, y2 = e["bbox"]
            cx, cy = e["center"]
            
            # Check if in any restricted area
            in_any_restricted = False
            for r_dict in self.boundaries:
                r = AreaRect(**r_dict)
                if r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2:
                    in_any_restricted = True
                    break
            
            color = (0, 0, 255) if in_any_restricted else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 4, color, -1)
            cv2.putText(annotated, f"P:{e['conf']:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            'frame': annotated,
            'events': events,
            'timestamp': time.time()
        }
    
    def add_frame(self, frame):
        """Add frame to detection queue (non-blocking)"""
        try:
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            # Drop oldest frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(frame, block=False)
            except:
                pass
    
    def stop(self):
        """Stop detection thread gracefully"""
        self.running = False
        self.wait(3000)

# ==================== DRAWING WIDGET ====================
class DrawingWidget(QLabel):
    """Widget for drawing restricted areas on a captured frame"""
    boundaries_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 2px solid #333; background-color: white;")
        self.setAlignment(Qt.AlignCenter)
        self.cv_image = None
        self.original_pixmap = None
        self.image = None
        self.drawing_enabled = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boundaries = []
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        
        self.boundary_color = QColor(0, 200, 0)
        self.preview_color = QColor(128, 128, 128, 100)
        self.line_width = 3
        
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_image(self, cv_image):
        """Set the image to draw on"""
        self.cv_image = cv_image
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.original_pixmap = pixmap
        self.image = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.scale_x = pixmap.width() / self.image.width()
        self.scale_y = pixmap.height() / self.image.height()
        
        self.offset_x = (self.width() - self.image.width()) // 2
        self.offset_y = (self.height() - self.image.height()) // 2
        
        self.clear_boundaries()
        self.update_display()

    def set_drawing_enabled(self, enabled: bool):
        """Enable or disable drawing"""
        self.drawing_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def resizeEvent(self, event):
        if self.cv_image is not None:
            self.set_image(self.cv_image)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if not self.drawing_enabled or event.button() != Qt.LeftButton or not self.image:
            return
        
        img_pos = self.widget_to_image_coords(event.pos())
        if not img_pos:
            return
        
        self.drawing = True
        self.start_point = img_pos
        self.end_point = img_pos
        self.update_display()

    def mouseMoveEvent(self, event):
        if self.drawing and self.image:
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.end_point = img_pos
                self.update_display()

    def mouseReleaseEvent(self, event):
        if not self.drawing_enabled or event.button() != Qt.LeftButton or not self.drawing:
            return
        
        img_pos = self.widget_to_image_coords(event.pos())
        if img_pos and self.start_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = img_pos.x(), img_pos.y()
            
            # Normalize coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Discard tiny rectangles
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                # Scale to original image coordinates
                orig_x1 = int(x1 * self.scale_x)
                orig_y1 = int(y1 * self.scale_y)
                orig_x2 = int(x2 * self.scale_x)
                orig_y2 = int(y2 * self.scale_y)
                
                rid = f"r{int(time.time()*1000)}"
                self.boundaries.append({
                    'id': rid,
                    'x1': orig_x1,
                    'y1': orig_y1,
                    'x2': orig_x2,
                    'y2': orig_y2
                })
                self.boundaries_changed.emit()
                logger.info(f"Added restricted area: {rid}")
        
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.update_display()

    def widget_to_image_coords(self, widget_point):
        """Convert widget coordinates to image coordinates"""
        if not self.image:
            return None
        
        img_rect = QRect(self.offset_x, self.offset_y, self.image.width(), self.image.height())
        if not img_rect.contains(widget_point):
            return None
        
        x = widget_point.x() - self.offset_x
        y = widget_point.y() - self.offset_y
        return QPoint(x, y)

    def update_display(self):
        """Redraw the widget with boundaries"""
        if not self.image:
            return
        
        display_pixmap = self.image.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw existing boundaries
        for i, boundary in enumerate(self.boundaries):
            x1 = int(boundary['x1'] / self.scale_x)
            y1 = int(boundary['y1'] / self.scale_y)
            x2 = int(boundary['x2'] / self.scale_x)
            y2 = int(boundary['y2'] / self.scale_y)
            
            pen = QPen(self.boundary_color, self.line_width)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 200, 0, 50))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(x1 + 5, y1 + 15, f"Area {i+1}")
        
        # Draw current rectangle being drawn
        if self.drawing and self.start_point and self.end_point:
            pen = QPen(self.preview_color, self.line_width, Qt.DashLine)
            painter.setPen(pen)
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        
        painter.end()
        
        # Draw on full widget
        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(Qt.white)
        final_painter = QPainter(final_pixmap)
        final_painter.drawPixmap(self.offset_x, self.offset_y, display_pixmap)
        final_painter.end()
        
        self.setPixmap(final_pixmap)

    def clear_boundaries(self):
        """Clear all boundaries"""
        self.boundaries.clear()
        self.drawing = False
        self.start_point = None
        self.end_point = None
        if self.image:
            self.update_display()
        logger.info("All boundaries cleared")

    def undo_last(self):
        """Remove the last drawn boundary"""
        if self.boundaries:
            removed = self.boundaries.pop()
            self.boundaries_changed.emit()
            self.update_display()
            logger.info(f"Removed boundary: {removed['id']}")
            return True
        return False

    def get_boundaries(self):
        """Get all boundaries"""
        return self.boundaries

    def load_boundaries(self, boundaries):
        """Load boundaries from list"""
        self.boundaries = boundaries
        self.update_display()

# ==================== TRAINING PAGE ====================
class TrainingPage(QWidget):
    """Teaching/Training mode page"""
    
    def __init__(self, config_mgr: ConfigManager):
        super().__init__()
        self.config_mgr = config_mgr
        self.camera_thread = None
        self.captured_frame = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training - Define Restricted Areas")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Instructions:\n"
            "1. Select camera and start preview\n"
            "2. Click 'Capture Frame' to freeze the image\n"
            "3. Draw rectangles on the frozen frame to define restricted areas\n"
            "4. Save boundaries when complete"
        )
        instructions.setStyleSheet(
            "background-color: #E1F5FE; padding: 10px; border-radius: 5px; "
            "font-size: 11px; color: #0277BD; border: 1px solid #0288D1;"
        )
        layout.addWidget(instructions)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        # Camera type selection
        camera_type_layout = QVBoxLayout()
        self.usb_radio = QPushButton("USB Camera")
        self.usb_radio.setCheckable(True)
        self.usb_radio.setChecked(True)
        self.ip_radio = QPushButton("IP Camera (RTSP)")
        self.ip_radio.setCheckable(True)
        camera_type_layout.addWidget(self.usb_radio)
        camera_type_layout.addWidget(self.ip_radio)
        camera_controls.addLayout(camera_type_layout)
        
        self.usb_radio.clicked.connect(lambda: self._set_camera_type("usb"))
        self.ip_radio.clicked.connect(lambda: self._set_camera_type("ip"))
        
        # USB camera selector
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        camera_controls.addWidget(QLabel("USB:"))
        camera_controls.addWidget(self.camera_combo)
        
        # RTSP URL input
        camera_controls.addWidget(QLabel("RTSP:"))
        self.ip_url_input = QLineEdit()
        self.ip_url_input.setPlaceholderText("rtsp://user:pass@host:port/stream")
        self.ip_url_input.setText(DEFAULT_RTSP_URL)
        self.ip_url_input.setMinimumWidth(300)
        self.ip_url_input.setEnabled(False)
        camera_controls.addWidget(self.ip_url_input)
        
        # Camera action buttons
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        camera_controls.addWidget(self.start_camera_btn)
        
        self.capture_btn = QPushButton("Capture Frame")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        camera_controls.addWidget(self.capture_btn)
        
        layout.addLayout(camera_controls)
        
        # Stacked widget for preview/drawing
        self.view_stack = QStackedWidget()
        
        # Live preview label
        self.preview_label = QLabel("Start camera to see preview")
        self.preview_label.setMinimumSize(900, 600)
        self.preview_label.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.view_stack.addWidget(self.preview_label)
        
        # Drawing widget
        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setMinimumSize(900, 600)
        self.drawing_widget.boundaries_changed.connect(self.on_boundaries_changed)
        self.view_stack.addWidget(self.drawing_widget)
        
        layout.addWidget(self.view_stack)
        
        # Instruction label
        self.instruction_label = QLabel("Click 'Start Camera' to begin")
        self.instruction_label.setStyleSheet(
            "background-color: #E3F2FD; padding: 10px; border-radius: 5px; "
            "font-size: 12px; color: #1976D2;"
        )
        layout.addWidget(self.instruction_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_boundaries)
        self.clear_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
        button_layout.addWidget(self.clear_btn)
        
        self.undo_btn = QPushButton("Undo Last")
        self.undo_btn.clicked.connect(self.undo_last_boundary)
        self.undo_btn.setStyleSheet("background-color: #FFB300; color: white; padding: 10px;")
        button_layout.addWidget(self.undo_btn)
        
        self.save_btn = QPushButton("Save Boundaries")
        self.save_btn.clicked.connect(self.save_boundaries)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
            "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
        )
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def _set_camera_type(self, cam_type):
        """Handle camera type selection"""
        if cam_type == "usb":
            self.usb_radio.setChecked(True)
            self.ip_radio.setChecked(False)
            self.camera_combo.setEnabled(True)
            self.ip_url_input.setEnabled(False)
        else:
            self.usb_radio.setChecked(False)
            self.ip_radio.setChecked(True)
            self.camera_combo.setEnabled(False)
            self.ip_url_input.setEnabled(True)
    
    def refresh_cameras(self):
        """Scan for available USB cameras"""
        self.camera_combo.clear()
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()
    
    def toggle_camera(self):
        """Start or stop camera"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera thread for live preview"""
        if self.usb_radio.isChecked():
            camera_source = self.camera_combo.currentIndex()
        else:
            camera_source = self.ip_url_input.text().strip()
            if not camera_source or not camera_source.startswith("rtsp://"):
                QMessageBox.warning(self, "Error", "Please enter a valid RTSP URL!")
                return
        
        try:
            self.camera_thread = CameraThread(
                camera_source,
                processing_size=tuple(self.config_mgr.config.processing_resolution)
            )
            self.camera_thread.frame_ready.connect(self.update_preview)
            self.camera_thread.error_signal.connect(self.handle_camera_error)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.start()
            
            self.start_camera_btn.setText("Stop Camera")
            self.start_camera_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px;")
            self.capture_btn.setEnabled(True)
            self.view_stack.setCurrentWidget(self.preview_label)
            self.instruction_label.setText("Camera running - Click 'Capture Frame' to freeze and draw areas")
            
            logger.info("Training camera started")
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start camera:\n{str(e)}")
    
    def stop_camera(self):
        """Stop camera thread"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.preview_label.clear()
        self.preview_label.setText("Start camera to see preview")
        self.start_camera_btn.setText("Start Camera")
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.capture_btn.setEnabled(False)
        self.view_stack.setCurrentWidget(self.preview_label)
        
        logger.info("Training camera stopped")
    
    def update_preview(self, frame):
        """Update live preview (signal handler)"""
        self.captured_frame = frame.copy()
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
    
    def capture_frame(self):
        """Capture current frame for drawing"""
        if self.captured_frame is None:
            QMessageBox.warning(self, "Error", "No frame available to capture!")
            return
        
        # Stop camera
        self.stop_camera()
        
        # Switch to drawing widget
        self.drawing_widget.set_image(self.captured_frame)
        self.drawing_widget.set_drawing_enabled(True)
        self.view_stack.setCurrentWidget(self.drawing_widget)
        
        self.instruction_label.setText(
            "Draw rectangles to define restricted areas. Click and drag to create each area."
        )
        
        logger.info("Frame captured for drawing")
    
    def on_boundaries_changed(self):
        """Handle boundary changes"""
        count = len(self.drawing_widget.get_boundaries())
        self.instruction_label.setText(
            f"Restricted areas defined: {count}. Click 'Save Boundaries' when complete."
        )
    
    def clear_boundaries(self):
        """Clear all boundaries"""
        self.drawing_widget.clear_boundaries()
        self.instruction_label.setText("All boundaries cleared. Draw new areas.")
    
    def undo_last_boundary(self):
        """Undo last boundary"""
        if self.drawing_widget.undo_last():
            count = len(self.drawing_widget.get_boundaries())
            self.instruction_label.setText(f"Undo successful. Restricted areas: {count}")
        else:
            QMessageBox.information(self, "Undo", "No boundaries to undo!")
    
    def save_boundaries(self):
        """Save boundaries to config"""
        boundaries = self.drawing_widget.get_boundaries()
        
        if not boundaries:
            QMessageBox.warning(self, "Error", "No boundaries defined! Draw at least one restricted area.")
            return
        
        try:
            self.config_mgr.config.restricted_areas = boundaries
            
            # Save camera config
            if self.usb_radio.isChecked():
                self.config_mgr.config.camera = {
                    "type": "usb",
                    "usb_index": self.camera_combo.currentIndex()
                }
            else:
                self.config_mgr.config.camera = {
                    "type": "rtsp",
                    "rtsp_url": self.ip_url_input.text().strip()
                }
            
            self.config_mgr.save()
            
            logger.info(f"Boundaries saved: {len(boundaries)} areas")
            
            QMessageBox.information(
                self, "Success",
                f"Boundaries saved successfully!\n\n"
                f"Restricted areas: {len(boundaries)}\n\n"
                f"You can now switch to Detection mode."
            )
            
            self.status_label.setText(f"Saved: {len(boundaries)} restricted areas")
            self.status_label.setStyleSheet(
                "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
                "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
            )
        except Exception as e:
            logger.error(f"Failed to save boundaries: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def handle_camera_error(self, error):
        """Handle camera errors"""
        logger.error(f"Camera error: {error}")
        self.status_label.setText(f"Camera error: {error[:50]}...")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #FFEBEE; color: #C62828; "
            "border: 1px solid #F44336; border-radius: 5px; font-weight: bold;"
        )
    
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

# ==================== DETECTION PAGE ====================
class DetectionPage(QWidget):
    """Detection mode page"""
    
    def __init__(self, config_mgr: ConfigManager, relay: SafeRelay):
        super().__init__()
        self.config_mgr = config_mgr
        self.relay = relay
        self.camera_thread = None
        self.detection_thread = None
        self.running = False
        self.detection_count = 0
        self.alert_count = 0
        self.snapshot_dir = os.path.join(os.path.expanduser("~"), "pokayoke_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Detection - Safety Monitoring System")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Model selection
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setEnabled(YOLO_AVAILABLE)
        controls_layout.addWidget(self.load_model_btn)
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("padding: 5px; background-color: #F5F5F5;")
        controls_layout.addWidget(self.model_label)
        
        controls_layout.addStretch()
        
        # Start/Stop button
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        controls_layout.addWidget(self.start_btn)
        
        layout.addLayout(controls_layout)
        
        # Detection view
        self.detection_label = QLabel("Detection View")
        self.detection_label.setMinimumSize(900, 600)
        self.detection_label.setStyleSheet("border: 2px solid #333; background-color: #111;")
        self.detection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detection_label)
        
        # Metrics
        metrics_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("padding: 5px; font-weight: bold; background-color: #E3F2FD;")
        metrics_layout.addWidget(self.fps_label)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setStyleSheet("padding: 5px; font-weight: bold; background-color: #E3F2FD;")
        metrics_layout.addWidget(self.detection_count_label)
        
        self.alert_count_label = QLabel("Alerts: 0")
        self.alert_count_label.setStyleSheet("padding: 5px; font-weight: bold; background-color: #FFEBEE;")
        metrics_layout.addWidget(self.alert_count_label)
        
        layout.addLayout(metrics_layout)
        
        # Status
        self.status_label = QLabel("System Status: Idle")
        self.status_label.setStyleSheet(
            "padding: 15px; font-size: 16px; font-weight: bold; "
            "background-color: #f0f0f0; border-radius: 5px;"
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def load_model(self):
        """Load YOLO model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "YOLO Model (*.pt *.onnx);;All Files (*)"
        )
        
        if file_path:
            self.config_mgr.config.model["path"] = file_path
            self.model_label.setText(os.path.basename(file_path))
            self.start_btn.setEnabled(True)
            logger.info(f"Model selected: {file_path}")
            QMessageBox.information(self, "Success", f"Model loaded:\n{os.path.basename(file_path)}")
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        if self.running:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """Start detection system"""
        # Validate configuration
        if not self.config_mgr.config.restricted_areas:
            QMessageBox.warning(self, "Error", "No restricted areas defined! Please complete training first.")
            return
        
        model_path = self.config_mgr.config.model.get("path")
        if not model_path and YOLO_AVAILABLE:
            QMessageBox.warning(self, "Error", "Please load a model first!")
            return
        
        try:
            logger.info("="*60)
            logger.info("STARTING DETECTION SYSTEM")
            logger.info("="*60)
            
            # Reset counters
            self.detection_count = 0
            self.alert_count = 0
            
            # Start camera
            camera_config = self.config_mgr.config.camera
            if camera_config.get("type") == "usb":
                camera_source = camera_config.get("usb_index", 0)
            else:
                camera_source = camera_config.get("rtsp_url", DEFAULT_RTSP_URL)
            
            logger.info(f"Starting camera: {camera_source}")
            self.camera_thread = CameraThread(
                camera_source,
                processing_size=tuple(self.config_mgr.config.processing_resolution)
            )
            self.camera_thread.frame_ready.connect(self.on_frame_ready)
            self.camera_thread.error_signal.connect(self.handle_error)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.start()
            
            # Wait for camera to initialize
            for _ in range(30):  # 3 second timeout
                if self.camera_thread.camera and self.camera_thread.camera.isOpened():
                    logger.info("Camera initialized successfully")
                    break
                QThread.msleep(100)
            else:
                raise Exception("Camera failed to initialize within timeout")
            
            # Start detection
            logger.info("Starting detection thread")
            self.detection_thread = DetectionThread(
                model_path,
                self.config_mgr.config.restricted_areas,
                conf_threshold=self.config_mgr.config.model.get("confidence", 0.45)
            )
            self.detection_thread.detection_ready.connect(self.on_detection_ready)
            self.detection_thread.error_signal.connect(self.handle_error)
            self.detection_thread.fps_signal.connect(self.update_fps)
            self.detection_thread.start()
            
            # Wait for model to load
            for _ in range(60):  # 6 second timeout
                if self.detection_thread.model or self.detection_thread.hog:
                    logger.info("Detection model ready")
                    break
                QThread.msleep(100)
            
            self.running = True
            self.start_btn.setText("Stop Detection")
            self.start_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
            self.status_label.setText("System Status: Running")
            self.status_label.setStyleSheet(
                "padding: 15px; font-size: 16px; font-weight: bold; "
                "background-color: #4CAF50; color: white; border-radius: 5px;"
            )
            
            logger.info("Detection system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to start detection:\n{str(e)}")
            self.stop_detection()
    
    def stop_detection(self):
        """Stop detection system"""
        logger.info("Stopping detection system")
        
        self.running = False
        
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        
        self.start_btn.setText("Start Detection")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.detection_label.clear()
        self.detection_label.setText("Detection View")
        self.status_label.setText("System Status: Stopped")
        self.status_label.setStyleSheet(
            "padding: 15px; font-size: 16px; font-weight: bold; "
            "background-color: #f0f0f0; border-radius: 5px;"
        )
        
        logger.info("Detection system stopped")
    
    def on_frame_ready(self, frame):
        """Handle new frame from camera (forward to detection)"""
        if self.detection_thread and self.running:
            self.detection_thread.add_frame(frame)
    
    def on_detection_ready(self, result):
        """Handle detection results"""
        try:
            self.detection_count += 1
            self.detection_count_label.setText(f"Detections: {self.detection_count}")
            
            annotated = result['frame']
            events = result['events']
            
            # Check for alerts
            any_alert = False
            for e in events:
                cx, cy = e.get('center', (None, None))
                if cx is None:
                    continue
                for r_dict in self.config_mgr.config.restricted_areas:
                    r = AreaRect(**r_dict)
                    if r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2:
                        any_alert = True
                        break
                if any_alert:
                    break
            
            if any_alert:
                self.alert_count += 1
                self.alert_count_label.setText(f"Alerts: {self.alert_count}")
                
                # Save snapshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(self.snapshot_dir, f"alert_{ts}.jpg")
                try:
                    cv2.imwrite(fname, annotated)
                    logger.info(f"Alert snapshot saved: {fname}")
                except Exception as e:
                    logger.error(f"Failed to save snapshot: {e}")
                
                # Trigger relay
                threading.Thread(target=self._pulse_relay, args=(1, 2.0), daemon=True).start()
                
                self.status_label.setText("System Status: ALERT DETECTED")
                self.status_label.setStyleSheet(
                    "padding: 15px; font-size: 16px; font-weight: bold; "
                    "background-color: #F44336; color: white; border-radius: 5px;"
                )
            else:
                self.status_label.setText("System Status: Running - All Clear")
                self.status_label.setStyleSheet(
                    "padding: 15px; font-size: 16px; font-weight: bold; "
                    "background-color: #4CAF50; color: white; border-radius: 5px;"
                )
            
            # Display annotated frame
            self.display_frame(annotated)
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
    
    def display_frame(self, frame):
        """Display frame in UI"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.detection_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.detection_label.setPixmap(scaled_pixmap)
    
    def _pulse_relay(self, channel: int, duration: float):
        """Pulse relay for duration"""
        try:
            self.relay.set_state(channel, True)
            time.sleep(duration)
            self.relay.set_state(channel, False)
        except Exception as e:
            logger.error(f"Relay pulse failed: {e}")
    
    def update_fps(self, fps):
        """Update FPS label"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def handle_error(self, error):
        """Handle errors"""
        logger.error(f"Detection error: {error}")
    
    def update_status(self, status):
        """Update status"""
        logger.info(f"Status: {status}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_detection()

# ==================== MAIN WINDOW ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pokayoke Vision Safety System")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Initialize managers
        self.config_mgr = ConfigManager()
        self.relay = SafeRelay()
        
        # Create UI
        self.init_ui()
        
        logger.info("Application initialized")
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Pages
        self.training_page = TrainingPage(self.config_mgr)
        self.detection_page = DetectionPage(self.config_mgr, self.relay)
        
        self.tab_widget.addTab(self.training_page, "Training")
        self.tab_widget.addTab(self.detection_page, "Detection")
        
        # Status bar
        self.statusBar().showMessage("System Ready")
    
    def closeEvent(self, event):
        """Handle application close"""
        logger.info("Application close requested")
        
        if self.detection_page.running:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Detection is running. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # Clean up all resources
        try:
            self.training_page.cleanup()
            self.detection_page.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("="*60)
        logger.info("APPLICATION SHUTDOWN")
        logger.info("="*60)
        event.accept()

# ==================== MAIN ====================
def main():
    # Exception handler
    def exception_handler(exc_type, exc_value, exc_tb):
        logger.critical(
            f"Unhandled exception: {exc_type.__name__}: {exc_value}\n"
            f"{''.join(traceback.format_tb(exc_tb))}"
        )
    
    sys.excepthook = exception_handler
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info("="*60)
    logger.info("POKAYOKE VISION SAFETY SYSTEM STARTED")
    logger.info("="*60)
    logger.info(f"System: {sys.platform}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"YOLO Available: {YOLO_AVAILABLE}")
    logger.info(f"Relay Available: {RELAY_AVAILABLE}")
    logger.info("="*60)
    
    return_code = app.exec_()
    
    logger.info(f"Application exited with code {return_code}")
    sys.exit(return_code)

if __name__ == "__main__":
    main()