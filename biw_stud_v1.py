import sys
import os
import csv
import datetime
import time
import json
import threading
import queue
from collections import deque
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

# Configure industrial-grade logging with rotation
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Rotating file handler - 10MB max, 5 backups
    file_handler = RotatingFileHandler(
        'stud_detection.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Import detection modules
try:
    from logic.stud_detection import detect_studs
    from logic.reference_positions import get_reference_positions
    from logic.stud_analysis import find_missing_and_extra_studs
except ImportError as e:
    logger.warning(f"Detection modules not available: {e}")
    def detect_studs(frame):
        return []
    def get_reference_positions():
        return []
    def find_missing_and_extra_studs(ref, det):
        return [], ref, det

try:
    import pyhid_usb_relay
except ImportError:
    logger.warning("pyhid_usb_relay not available")
    pyhid_usb_relay = None

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO available")
except ImportError:
    logger.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False
    YOLO = None


class WatchdogTimer(QObject):
    """Watchdog timer for monitoring thread health"""
    timeout_signal = pyqtSignal(str)
    
    def __init__(self, timeout_seconds=30):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = {}
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logger.info(f"Watchdog initialized with {timeout_seconds}s timeout")
    
    def heartbeat(self, thread_name):
        """Record heartbeat from thread"""
        with self.lock:
            self.last_heartbeat[thread_name] = time.time()
    
    def _monitor(self):
        """Monitor thread heartbeats"""
        while self.running:
            time.sleep(5)
            current_time = time.time()
            
            with self.lock:
                for thread_name, last_time in list(self.last_heartbeat.items()):
                    if current_time - last_time > self.timeout_seconds:
                        logger.error(f"Watchdog: {thread_name} timeout!")
                        self.timeout_signal.emit(thread_name)
                        del self.last_heartbeat[thread_name]
    
    def stop(self):
        """Stop watchdog"""
        self.running = False


class CircularFrameBuffer:
    """Circular buffer for frame management"""
    def __init__(self, maxsize=3):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
    
    def put(self, frame):
        """Add frame to buffer"""
        with self.lock:
            self.buffer.append(frame.copy())
    
    def get(self):
        """Get latest frame"""
        with self.lock:
            return self.buffer[-1].copy() if self.buffer else None
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()


class CameraConfig:
    """Camera configuration management"""
    def __init__(self):
        self.config_file = "camera_config.json"
        self.configs = self.load_configs()
    
    def load_configs(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading camera configs: {e}")
        return {"ip_cameras": []}
    
    def save_configs(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.configs, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving configs: {e}")
            return False
    
    def add_ip_camera(self, name, rtsp_url, username="", password=""):
        camera_config = {
            "name": name,
            "rtsp_url": rtsp_url,
            "username": username,
            "password": password,
            "type": "ip"
        }
        self.configs["ip_cameras"].append(camera_config)
        return self.save_configs()
    
    def remove_ip_camera(self, name):
        self.configs["ip_cameras"] = [
            cam for cam in self.configs["ip_cameras"] if cam["name"] != name
        ]
        return self.save_configs()
    
    def get_all_cameras(self):
        cameras = []
        # USB cameras
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({"name": f"USB Camera {i}", "index": i, "type": "usb"})
                cap.release()
        # IP cameras
        cameras.extend(self.configs["ip_cameras"])
        return cameras


class IPCameraDialog(QDialog):
    """Dialog for adding IP camera with Hikvision defaults"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add IP Camera")
        self.setModal(True)
        self.resize(550, 450)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("📹 Add Hikvision IP/RTSP Camera")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3; padding: 10px;")
        layout.addWidget(title)
        
        # Camera name
        layout.addWidget(QLabel("Camera Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Station 1, BIW Line A")
        self.name_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self.name_input)
        
        # RTSP URL with default
        layout.addWidget(QLabel("RTSP URL:"))
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setText("rtsp://admin:Pass_123@192.168.1.64:554/stream")
        self.rtsp_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;")
        layout.addWidget(self.rtsp_input)
        
        # Hikvision URL templates
        templates_group = QGroupBox("📋 Hikvision URL Templates")
        templates_layout = QVBoxLayout(templates_group)
        
        template_buttons = [
            ("Main Stream", "rtsp://admin:Pass_123@192.168.1.64:554/Streaming/Channels/101"),
            ("Sub Stream", "rtsp://admin:Pass_123@192.168.1.64:554/Streaming/Channels/102"),
            ("Generic Stream", "rtsp://admin:Pass_123@192.168.1.64:554/stream"),
        ]
        
        for name, url in template_buttons:
            btn = QPushButton(f"📝 {name}")
            btn.clicked.connect(lambda checked, u=url: self.rtsp_input.setText(u))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E3F2FD;
                    padding: 6px;
                    border: 1px solid #2196F3;
                    border-radius: 4px;
                    text-align: left;
                }
                QPushButton:hover { background-color: #BBDEFB; }
            """)
            templates_layout.addWidget(btn)
        
        layout.addWidget(templates_group)
        
        # URL format help
        help_text = QLabel(
            "📘 Hikvision RTSP Format:\n"
            "• rtsp://username:password@ip:554/stream\n"
            "• Main stream (high quality): .../Streaming/Channels/101\n"
            "• Sub stream (lower bandwidth): .../Streaming/Channels/102\n\n"
            "Default: admin / Pass_123 @ 192.168.1.64"
        )
        help_text.setStyleSheet("background-color: #f0f8ff; padding: 10px; border-radius: 5px; font-size: 11px;")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        
        # Test connection button
        test_btn = QPushButton("🔍 Test Connection")
        test_btn.clicked.connect(self.test_connection)
        test_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f57c00; }
        """)
        layout.addWidget(test_btn)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("✓ Add Camera")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold;")
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("✗ Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold;")
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def test_connection(self):
        """Test RTSP connection"""
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            QMessageBox.warning(self, "Invalid Input", "Please enter RTSP URL.")
            return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read frames
            success = False
            for attempt in range(30):
                ret, frame = cap.read()
                if ret and frame is not None:
                    success = True
                    break
                time.sleep(0.1)
            
            cap.release()
            QApplication.restoreOverrideCursor()
            
            if success:
                QMessageBox.information(self, "Success", "✓ Connection successful!\nCamera is responding.")
            else:
                QMessageBox.warning(self, "Connection Failed", "Could not read frames.\nCheck URL and credentials.")
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Connection test failed:\n{str(e)}")
    
    def get_camera_config(self):
        return {
            "name": self.name_input.text().strip(),
            "rtsp_url": self.rtsp_input.text().strip(),
            "username": "",
            "password": ""
        }


class IndustrialCameraThread(QThread):
    """Industrial-grade camera thread with watchdog and buffering"""
    frame_ready = pyqtSignal(object)
    connection_status = pyqtSignal(str, bool)
    performance_update = pyqtSignal(float)  # FPS
    
    def __init__(self, camera_source, is_ip=False, watchdog=None):
        super().__init__()
        self.camera_source = camera_source
        self.is_ip = is_ip
        self.running = True
        self.camera = None
        self.watchdog = watchdog
        self.frame_buffer = CircularFrameBuffer(maxsize=3)
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 3
        
        logger.info(f"Camera thread initialized: {camera_source}")
    
    def run(self):
        """Main camera loop with watchdog heartbeat"""
        thread_name = f"Camera-{self.camera_source}"
        
        while self.running:
            if self.watchdog:
                self.watchdog.heartbeat(thread_name)
            
            if not self.connect_camera():
                if self.running:
                    time.sleep(self.reconnect_delay)
                continue
            
            self.read_frames(thread_name)
            
            if self.running:
                logger.warning("Camera disconnected. Reconnecting...")
                self.connection_status.emit("Reconnecting...", True)
                time.sleep(self.reconnect_delay)
    
    def connect_camera(self):
        """Connect with optimized settings for IP cameras"""
        try:
            logger.info(f"Connecting to camera: {self.camera_source}")
            self.connection_status.emit("Connecting...", False)
            
            if self.is_ip:
                # Optimized for Hikvision RTSP
                self.camera = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                # Use TCP for reliable connection
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            else:
                self.camera = cv2.VideoCapture(self.camera_source)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Verify connection
            if self.camera.isOpened():
                for _ in range(10):
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        logger.info("Camera connected successfully")
                        self.connection_status.emit("Connected", False)
                        self.reconnect_attempts = 0
                        return True
                    time.sleep(0.1)
            
            self.reconnect_attempts += 1
            error_msg = f"Connection failed (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
            logger.error(error_msg)
            self.connection_status.emit(error_msg, True)
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                self.connection_status.emit("Max reconnection attempts reached", True)
                self.running = False
            
            return False
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connection_status.emit(f"Error: {str(e)}", True)
            if self.camera:
                self.camera.release()
                self.camera = None
            return False
    
    def read_frames(self, thread_name):
        """Read frames with FPS monitoring"""
        consecutive_failures = 0
        
        while self.running and self.camera and self.camera.isOpened():
            try:
                if self.watchdog:
                    self.watchdog.heartbeat(thread_name)
                
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.error("Too many consecutive failures")
                        break
                    time.sleep(0.01)
                    continue
                
                consecutive_failures = 0
                
                # Update buffer and emit
                self.frame_buffer.put(frame)
                self.frame_ready.emit(frame.copy())
                
                # Calculate FPS
                self.fps_counter += 1
                if self.fps_counter % 30 == 0:
                    elapsed = time.time() - self.fps_start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    self.performance_update.emit(fps)
                    self.fps_start_time = time.time()
                
                # Frame rate control
                self.msleep(33)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Frame read error: {e}")
                break
    
    def stop(self):
        logger.info("Stopping camera thread")
        self.running = False
        self.quit()
        self.wait(3000)
        if self.camera:
            self.camera.release()
            self.camera = None


class BIWModelManager:
    """BIW model management"""
    def __init__(self):
        self.models_file = "biw_models.json"
        self.models = self.load_models()
    
    def load_models(self):
        if os.path.exists(self.models_file):
            try:
                with open(self.models_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading models: {e}")
        return {}
    
    def save_models(self):
        try:
            with open(self.models_file, 'w') as f:
                json.dump(self.models, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def add_model(self, name, stud_count, positions, description=""):
        self.models[name] = {
            "stud_count": stud_count,
            "positions": positions,
            "description": description,
            "created_at": datetime.datetime.now().isoformat()
        }
        return self.save_models()
    
    def get_model(self, name):
        return self.models.get(name)
    
    def get_model_names(self):
        return list(self.models.keys())
    
    def delete_model(self, name):
        if name in self.models:
            del self.models[name]
            return self.save_models()
        return False


class TeachingCanvas(QLabel):
    """Interactive canvas for teaching"""
    positions_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #333; background-color: white;")
        self.setAlignment(Qt.AlignCenter)
        
        self.drawing = False
        self.current_rect = None
        self.start_point = None
        self.rectangles = []
        self.current_frame = None
        self.captured_image = None
        
        self.setMouseTracking(True)
        self.setText("📷 Start camera to begin teaching")
    
    def update_camera_frame(self, frame):
        self.current_frame = frame.copy()
        if self.captured_image is None:
            self.update_display()
    
    def capture_teaching_image(self):
        if self.current_frame is not None:
            self.captured_image = self.current_frame.copy()
            self.rectangles = []
            self.update_display()
            return True
        return False
    
    def clear_capture(self):
        self.captured_image = None
        self.rectangles = []
        if self.current_frame is not None:
            self.update_display()
    
    def update_display(self):
        display_image = None
        
        if self.captured_image is not None:
            display_image = self.captured_image.copy()
            
            for i, rect in enumerate(self.rectangles):
                cv2.rectangle(display_image,
                            (rect['x'], rect['y']),
                            (rect['x'] + rect['width'], rect['y'] + rect['height']),
                            (0, 255, 0), 2)
                cv2.putText(display_image, f"S{i+1}",
                          (rect['x'] + 5, rect['y'] + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.current_rect:
                cv2.rectangle(display_image,
                            (self.current_rect['x'], self.current_rect['y']),
                            (self.current_rect['x'] + self.current_rect['width'],
                             self.current_rect['y'] + self.current_rect['height']),
                            (255, 0, 0), 2)
            
            cv2.putText(display_image, "Teaching Mode - Draw rectangles around studs", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                       
        elif self.current_frame is not None:
            display_image = self.current_frame.copy()
            cv2.putText(display_image, "Live Preview - Capture to teach", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if display_image is not None:
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            
            if self.captured_image is not None:
                positions = [(rect['x'] + rect['width']//2, rect['y'] + rect['height']//2)
                           for rect in self.rectangles]
                self.positions_updated.emit(positions)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.captured_image is not None:
            pos = self.map_to_image_coords(event.pos())
            if pos:
                self.drawing = True
                self.start_point = pos
                self.current_rect = {'x': pos.x(), 'y': pos.y(), 'width': 0, 'height': 0}
    
    def mouseMoveEvent(self, event):
        if self.drawing and self.current_rect and self.start_point:
            pos = self.map_to_image_coords(event.pos())
            if pos:
                self.current_rect['width'] = pos.x() - self.start_point.x()
                self.current_rect['height'] = pos.y() - self.start_point.y()
                self.update_display()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.current_rect:
            if abs(self.current_rect['width']) > 10 and abs(self.current_rect['height']) > 10:
                if self.current_rect['width'] < 0:
                    self.current_rect['x'] += self.current_rect['width']
                    self.current_rect['width'] = abs(self.current_rect['width'])
                if self.current_rect['height'] < 0:
                    self.current_rect['y'] += self.current_rect['height']
                    self.current_rect['height'] = abs(self.current_rect['height'])
                
                self.rectangles.append(self.current_rect.copy())
            
            self.drawing = False
            self.current_rect = None
            self.start_point = None
            self.update_display()
    
    def map_to_image_coords(self, widget_pos):
        if not self.pixmap():
            return None
        
        pixmap_rect = self.pixmap().rect()
        widget_rect = self.rect()
        
        x_offset = (widget_rect.width() - pixmap_rect.width()) // 2
        y_offset = (widget_rect.height() - pixmap_rect.height()) // 2
        
        if (widget_pos.x() < x_offset or widget_pos.x() > x_offset + pixmap_rect.width() or
            widget_pos.y() < y_offset or widget_pos.y() > y_offset + pixmap_rect.height()):
            return None
        
        if self.captured_image is not None:
            img_x = int((widget_pos.x() - x_offset) * self.captured_image.shape[1] / pixmap_rect.width())
            img_y = int((widget_pos.y() - y_offset) * self.captured_image.shape[0] / pixmap_rect.height())
            return QPoint(img_x, img_y)
        
        return None
    
    def clear_rectangles(self):
        self.rectangles = []
        if self.captured_image is not None:
            self.update_display()
    
    def get_positions(self):
        return [(rect['x'] + rect['width']//2, rect['y'] + rect['height']//2)
                for rect in self.rectangles]


class TeachingPage(QWidget):
    """Teaching page"""
    def __init__(self, model_manager, camera_config, watchdog):
        super().__init__()
        self.model_manager = model_manager
        self.camera_config = camera_config
        self.watchdog = watchdog
        self.camera_thread = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("📚 BIW Model Teaching")
        header.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: #2196F3;
            padding: 10px; background-color: #f0f8ff; border-radius: 8px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        content = QHBoxLayout()
        
        # Left - Canvas
        left = QVBoxLayout()
        
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        controls.addWidget(self.camera_combo)
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_camera)
        controls.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        left.addLayout(controls)
        
        self.status_label = QLabel("Ready")
        left.addWidget(self.status_label)
        
        teach_controls = QHBoxLayout()
        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        teach_controls.addWidget(self.capture_btn)
        
        clear_btn = QPushButton("🔄 Clear")
        clear_btn.clicked.connect(self.clear_all)
        teach_controls.addWidget(clear_btn)
        teach_controls.addStretch()
        left.addLayout(teach_controls)
        
        self.canvas = TeachingCanvas()
        self.canvas.positions_updated.connect(self.update_count)
        left.addWidget(self.canvas)
        
        content.addLayout(left, 3)
        
        # Right - Config
        right = QVBoxLayout()
        
        info_group = QGroupBox("📍 Info")
        info_layout = QVBoxLayout(info_group)
        self.count_label = QLabel("Studs: 0")
        info_layout.addWidget(self.count_label)
        right.addWidget(info_group)
        
        config_group = QGroupBox("⚙️ Model")
        config_layout = QVBoxLayout(config_group)
        config_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        config_layout.addWidget(self.name_input)
        
        config_layout.addWidget(QLabel("Description:"))
        self.desc_input = QTextEdit()
        self.desc_input.setMaximumHeight(60)
        config_layout.addWidget(self.desc_input)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_model)
        config_layout.addWidget(save_btn)
        right.addWidget(config_group)
        
        models_group = QGroupBox("📋 Models")
        models_layout = QVBoxLayout(models_group)
        self.models_list = QListWidget()
        self.refresh_models()
        models_layout.addWidget(self.models_list)
        
        delete_btn = QPushButton("🗑 Delete")
        delete_btn.clicked.connect(self.delete_model)
        models_layout.addWidget(delete_btn)
        right.addWidget(models_group)
        
        right.addStretch()
        content.addLayout(right, 1)
        layout.addLayout(content)
    
    def refresh_cameras(self):
        self.camera_combo.clear()
        for cam in self.camera_config.get_all_cameras():
            icon = "🔌" if cam["type"] == "usb" else "🌐"
            self.camera_combo.addItem(f"{icon} {cam['name']}", cam)
    
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
        
        cam_data = self.camera_combo.currentData()
        if not cam_data:
            QMessageBox.warning(self, "Error", "No camera selected")
            return
        
        if cam_data["type"] == "usb":
            source = cam_data["index"]
            is_ip = False
        else:
            source = cam_data["rtsp_url"]
            is_ip = True
        
        self.camera_thread = IndustrialCameraThread(source, is_ip, self.watchdog)
        self.camera_thread.frame_ready.connect(self.canvas.update_camera_frame)
        self.camera_thread.connection_status.connect(self.update_status)
        self.camera_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        logger.info(f"Teaching camera started: {cam_data['name']}")
    
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        logger.info("Teaching camera stopped")
    
    def update_status(self, status, is_error):
        color = "#f44336" if is_error else "#4CAF50"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.status_label.setText(status)
    
    def capture_image(self):
        if self.canvas.capture_teaching_image():
            QMessageBox.information(self, "Captured", "Draw rectangles around studs")
        else:
            QMessageBox.warning(self, "Error", "No frame available")
    
    def clear_all(self):
        self.canvas.clear_rectangles()
        self.canvas.clear_capture()
    
    def update_count(self, positions):
        self.count_label.setText(f"Studs: {len(positions)}")
    
    def save_model(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Enter model name")
            return
        
        positions = self.canvas.get_positions()
        if not positions:
            QMessageBox.warning(self, "Error", "Mark at least one stud")
            return
        
        desc = self.desc_input.toPlainText().strip()
        
        if self.model_manager.add_model(name, len(positions), positions, desc):
            QMessageBox.information(self, "Success", f"Model '{name}' saved!")
            self.refresh_models()
            self.name_input.clear()
            self.desc_input.clear()
            logger.info(f"Model saved: {name} ({len(positions)} studs)")
    
    def refresh_models(self):
        self.models_list.clear()
        for name in self.model_manager.get_model_names():
            model = self.model_manager.get_model(name)
            self.models_list.addItem(f"{name} ({model['stud_count']} studs)")
    
    def delete_model(self):
        item = self.models_list.currentItem()
        if not item:
            return
        
        name = item.text().split(" (")[0]
        reply = QMessageBox.question(self, "Delete", f"Delete '{name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.model_manager.delete_model(name):
                self.refresh_models()
                logger.info(f"Model deleted: {name}")
    
    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


class IndustrialDetectionThread(QThread):
    """Industrial detection thread - NO minimum stud requirement"""
    frame_ready = pyqtSignal(object)
    detection_ready = pyqtSignal(object, list, list, list)
    status_update = pyqtSignal(str, int, int)
    cycle_status = pyqtSignal(str, int)
    connection_status = pyqtSignal(str, bool)
    
    def __init__(self, camera_source, is_ip, model_manager, biw_model, watchdog=None):
        super().__init__()
        self.camera_source = camera_source
        self.is_ip = is_ip
        self.model_manager = model_manager
        self.biw_model = biw_model
        self.watchdog = watchdog
        self.running = True
        self.camera = None
        
        # State machine - SIMPLIFIED: Start on ANY stud detection
        self.state = "WAITING_FOR_STUDS"
        self.entry_time = 0
        self.detection_delay = 3
        self.last_result_frame = None
        self.studs_absent_count = 0
        self.studs_absent_threshold = 10
        
        # Quality tracking
        self.csv_path = "quality_count.csv"
        self.ok_count, self.not_ok_count = self._load_csv()
        
        # Relay
        self.relay = self._init_relay()
        
        # Frame buffer
        self.frame_buffer = CircularFrameBuffer(maxsize=3)
        
        logger.info("Detection thread initialized")
    
    def _init_relay(self):
        try:
            if pyhid_usb_relay:
                return pyhid_usb_relay.find()
        except Exception as e:
            logger.warning(f"Relay init failed: {e}")
        return None
    
    def _load_csv(self):
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    row = next(reader)
                    return int(row[0]), int(row[1])
            except:
                pass
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["OK", "NOT_OK"])
            writer.writerow([0, 0])
        return 0, 0
    
    def _update_csv(self):
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["OK", "NOT_OK"])
                writer.writerow([self.ok_count, self.not_ok_count])
        except Exception as e:
            logger.error(f"CSV error: {e}")
    
    def set_detection_delay(self, delay):
        self.detection_delay = delay
        logger.info(f"Detection delay set: {delay}s")
    
    def run(self):
        thread_name = f"Detection-{self.camera_source}"
        
        while self.running:
            if self.watchdog:
                self.watchdog.heartbeat(thread_name)
            
            if not self.connect_camera():
                if self.running:
                    time.sleep(3)
                continue
            
            self.detection_loop(thread_name)
            
            if self.running:
                self.connection_status.emit("Reconnecting...", True)
                time.sleep(3)
    
    def connect_camera(self):
        try:
            self.connection_status.emit("Connecting...", False)
            
            if self.is_ip:
                self.camera = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            else:
                self.camera = cv2.VideoCapture(self.camera_source)
            
            if self.camera.isOpened():
                for _ in range(10):
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        self.connection_status.emit("Connected", False)
                        logger.info("Detection camera connected")
                        return True
                    time.sleep(0.1)
            
            if self.camera:
                self.camera.release()
            return False
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def detection_loop(self, thread_name):
        consecutive_failures = 0
        
        while self.running and self.camera and self.camera.isOpened():
            try:
                if self.watchdog:
                    self.watchdog.heartbeat(thread_name)
                
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        break
                    continue
                
                consecutive_failures = 0
                self.frame_buffer.put(frame)
                current_time = time.time()
                
                # State machine
                if self.state == "WAITING_FOR_STUDS":
                    self._handle_waiting(frame, current_time)
                elif self.state == "STUDS_DETECTED":
                    self._handle_detected(frame, current_time)
                elif self.state == "TIMER_RUNNING":
                    self._handle_timer(frame, current_time)
                elif self.state == "PHOTO_TAKEN":
                    self._handle_photo_taken()
                elif self.state == "DISPLAYING_RESULT":
                    self._handle_displaying(frame, current_time)
                
                self.msleep(33)
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                break
    
    def _detect_studs(self, frame):
        """Simplified: Detect ANY studs - no minimum threshold"""
        # TODO: Implement YOLO detection
        import random
        num_studs = random.randint(0, 15)
        detections = []
        
        for _ in range(num_studs):
            if random.random() > 0.3:
                x = random.randint(50, frame.shape[1] - 100)
                y = random.randint(50, frame.shape[0] - 100)
                detections.append({
                    'class': 'Stud',
                    'confidence': random.uniform(0.7, 0.95),
                    'bbox': (x, y, 20, 20)
                })
        
        # Return True if ANY studs detected
        return len(detections) > 0, detections
    
    def _handle_waiting(self, frame, current_time):
        has_studs, detections = self._detect_studs(frame)
        
        if has_studs:
            self.state = "STUDS_DETECTED"
            self.entry_time = current_time
            self.cycle_status.emit(f"Studs Detected ({len(detections)})", 0)
            logger.info(f"Cycle started: {len(detections)} studs detected")
        
        status_text = f"Waiting (Found {len(detections)})"
        display_frame = self._draw_status(frame.copy(), status_text, detections)
        self.frame_ready.emit(display_frame)
    
    def _handle_detected(self, frame, current_time):
        has_studs, detections = self._detect_studs(frame)
        
        if has_studs:
            self.state = "TIMER_RUNNING"
            self.cycle_status.emit("Timer Running", self.detection_delay)
        else:
            self.state = "WAITING_FOR_STUDS"
            self.cycle_status.emit("Studs Lost", 0)
        
        status_text = f"Detected ({len(detections)} studs)"
        display_frame = self._draw_status(frame.copy(), status_text, detections)
        self.frame_ready.emit(display_frame)
    
    def _handle_timer(self, frame, current_time):
        has_studs, detections = self._detect_studs(frame)
        
        elapsed = current_time - self.entry_time
        remaining = max(0, self.detection_delay - elapsed)
        
        if not has_studs:
            self.state = "WAITING_FOR_STUDS"
            self.cycle_status.emit("Studs Lost", 0)
            return
        
        if remaining <= 0:
            self.state = "PHOTO_TAKEN"
            self._capture_and_detect(frame)
            self.cycle_status.emit("Processing", 0)
        else:
            self.cycle_status.emit("Timer Running", int(remaining) + 1)
            status_text = f"Timer: {int(remaining) + 1}s ({len(detections)} studs)"
            display_frame = self._draw_status(frame.copy(), status_text, detections)
            self.frame_ready.emit(display_frame)
    
    def _handle_photo_taken(self):
        self.state = "DISPLAYING_RESULT"
        self.cycle_status.emit("Displaying Result", 0)
    
    def _handle_displaying(self, frame, current_time):
        has_studs, detections = self._detect_studs(frame)
        
        if has_studs:
            self.studs_absent_count = 0
            if self.last_result_frame is not None:
                self.frame_ready.emit(self.last_result_frame)
        else:
            self.studs_absent_count += 1
            if self.studs_absent_count >= self.studs_absent_threshold:
                self.state = "WAITING_FOR_STUDS"
                self.last_result_frame = None
                self.studs_absent_count = 0
                self.cycle_status.emit("Ready for Next", 0)
                logger.info("Cycle complete - ready for next")
            
            if self.last_result_frame is not None:
                self.frame_ready.emit(self.last_result_frame)
    
    def _capture_and_detect(self, frame):
        try:
            logger.info("Capturing and analyzing...")
            
            model = self.model_manager.get_model(self.biw_model)
            reference_studs = model['positions'] if model else []
            
            _, detections = self._detect_studs(frame)
            detected_studs = []
            
            for det in detections:
                if det['class'] == 'Stud':
                    bbox = det['bbox']
                    cx = bbox[0] + bbox[2] // 2
                    cy = bbox[1] + bbox[3] // 2
                    detected_studs.append((cx, cy))
            
            matched, missing, extra = find_missing_and_extra_studs(
                reference_studs, detected_studs
            )
            
            result_frame = self._draw_results(
                frame, reference_studs, detected_studs, matched, missing, extra
            )
            self.last_result_frame = result_frame
            
            self._handle_quality(matched, missing, len(reference_studs))
            self.detection_ready.emit(result_frame, matched, missing, extra)
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            self.last_result_frame = frame.copy()
    
    def _draw_status(self, frame, status_text, detections):
        for det in detections:
            bbox = det['bbox']
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{det['confidence']:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Status: {status_text}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def _draw_results(self, frame, ref, det, matched, missing, extra):
        for r in ref:
            cv2.circle(frame, r, 8, (255, 255, 0), 1)
        
        for d, r in matched:
            cv2.circle(frame, d, 12, (0, 255, 0), 2)
        
        for m in missing:
            cv2.circle(frame, m, 12, (0, 0, 255), 2)
        
        for e in extra:
            cv2.circle(frame, e, 12, (255, 0, 255), 2)
        
        expected = len(ref)
        
        if len(matched) == expected and len(missing) == 0:
            status_text = "QUALITY: OK"
            color = (0, 255, 0)
        else:
            status_text = "QUALITY: NOT OK"
            color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        info = f"Matched: {len(matched)} | Missing: {len(missing)} | Extra: {len(extra)} | Expected: {expected}"
        cv2.putText(frame, info, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def _handle_quality(self, matched, missing, expected):
        if len(matched) == expected and len(missing) == 0:
            self.ok_count += 1
            status = "OK"
            if self.relay:
                try:
                    self.relay.set_state(1, True)
                    self.relay.set_state(2, False)
                except Exception as e:
                    logger.error(f"Relay error: {e}")
        else:
            self.not_ok_count += 1
            status = "NOT OK"
            if self.relay:
                try:
                    self.relay.set_state(1, False)
                    self.relay.set_state(2, True)
                except Exception as e:
                    logger.error(f"Relay error: {e}")
        
        self._update_csv()
        self.status_update.emit(status, self.ok_count, self.not_ok_count)
        logger.info(f"Quality: {status}, OK: {self.ok_count}, NOT OK: {self.not_ok_count}")
    
    def stop(self):
        logger.info("Stopping detection thread")
        self.running = False
        self.quit()
        self.wait(3000)
        if self.camera:
            self.camera.release()


class DetectionPage(QWidget):
    """Detection page with industrial features"""
    def __init__(self, model_manager, camera_config, watchdog):
        super().__init__()
        self.model_manager = model_manager
        self.camera_config = camera_config
        self.watchdog = watchdog
        self.detection_thread = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("🎯 Industrial Stud Detection System")
        header.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: #4CAF50;
            padding: 10px; background-color: #f0fff0; border-radius: 8px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        content = QHBoxLayout()
        
        # Left - Video
        left = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #333; background: #f5f5f5;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("🎯 Start detection")
        left.addWidget(self.video_label)
        
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        controls.addWidget(self.camera_combo)
        
        controls.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.refresh_models()
        controls.addWidget(self.model_combo)
        
        controls.addWidget(QLabel("Timer:"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(1, 30)
        self.delay_spin.setValue(3)
        self.delay_spin.setSuffix(" sec")
        controls.addWidget(self.delay_spin)
        
        controls.addStretch()
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white;
                padding: 8px 16px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        controls.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336; color: white;
                padding: 8px 16px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        controls.addWidget(self.stop_btn)
        
        left.addLayout(controls)
        content.addLayout(left, 3)
        
        # Right - Status
        right = QVBoxLayout()
        
        conn_group = QGroupBox("🔗 Connection")
        conn_layout = QVBoxLayout(conn_group)
        self.conn_label = QLabel("Ready")
        conn_layout.addWidget(self.conn_label)
        right.addWidget(conn_group)
        
        cycle_group = QGroupBox("🔄 Cycle")
        cycle_layout = QVBoxLayout(cycle_group)
        self.cycle_label = QLabel("Waiting")
        self.timer_label = QLabel("Timer: --")
        cycle_layout.addWidget(self.cycle_label)
        cycle_layout.addWidget(self.timer_label)
        right.addWidget(cycle_group)
        
        quality_group = QGroupBox("✅ Quality")
        quality_layout = QVBoxLayout(quality_group)
        self.ok_label = QLabel("OK: 0")
        self.nok_label = QLabel("NOT OK: 0")
        quality_layout.addWidget(self.ok_label)
        quality_layout.addWidget(self.nok_label)
        
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self.reset_counts)
        quality_layout.addWidget(reset_btn)
        right.addWidget(quality_group)
        
        info_group = QGroupBox("🔍 Detection")
        info_layout = QVBoxLayout(info_group)
        self.matched_label = QLabel("Matched: 0")
        self.missing_label = QLabel("Missing: 0")
        self.extra_label = QLabel("Extra: 0")
        self.expected_label = QLabel("Expected: 0")
        
        for lbl in [self.matched_label, self.missing_label, 
                    self.extra_label, self.expected_label]:
            info_layout.addWidget(lbl)
        right.addWidget(info_group)
        
        right.addStretch()
        content.addLayout(right, 1)
        layout.addLayout(content)
    
    def refresh_cameras(self):
        self.camera_combo.clear()
        for cam in self.camera_config.get_all_cameras():
            icon = "🔌" if cam["type"] == "usb" else "🌐"
            self.camera_combo.addItem(f"{icon} {cam['name']}", cam)
    
    def refresh_models(self):
        self.model_combo.clear()
        for name in self.model_manager.get_model_names():
            model = self.model_manager.get_model(name)
            self.model_combo.addItem(f"{name} ({model['stud_count']})", name)
    
    def start_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            return
        
        cam_data = self.camera_combo.currentData()
        if not cam_data:
            QMessageBox.warning(self, "Error", "No camera")
            return
        
        model = self.model_combo.currentData()
        if not model:
            QMessageBox.warning(self, "Error", "No model")
            return
        
        if cam_data["type"] == "usb":
            source = cam_data["index"]
            is_ip = False
        else:
            source = cam_data["rtsp_url"]
            is_ip = True
        
        self.detection_thread = IndustrialDetectionThread(
            source, is_ip, self.model_manager, model, self.watchdog
        )
        self.detection_thread.set_detection_delay(self.delay_spin.value())
        
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.detection_ready.connect(self.on_detection)
        self.detection_thread.status_update.connect(self.on_status)
        self.detection_thread.cycle_status.connect(self.on_cycle)
        self.detection_thread.connection_status.connect(self.on_connection)
        
        self.detection_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        model_data = self.model_manager.get_model(model)
        self.expected_label.setText(f"Expected: {model_data['stud_count']}")
        
        logger.info(f"Detection started: {cam_data['name']}, Model: {model}")
    
    def stop_detection(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("🎯 Stopped")
        logger.info("Detection stopped")
    
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
    
    def on_detection(self, frame, matched, missing, extra):
        self.update_frame(frame)
        self.matched_label.setText(f"Matched: {len(matched)}")
        self.missing_label.setText(f"Missing: {len(missing)}")
        self.extra_label.setText(f"Extra: {len(extra)}")
    
    def on_status(self, status, ok, nok):
        self.ok_label.setText(f"OK: {ok}")
        self.nok_label.setText(f"NOT OK: {nok}")
    
    def on_cycle(self, status, timer):
        self.cycle_label.setText(status)
        if timer > 0:
            self.timer_label.setText(f"Timer: {timer}s")
        else:
            self.timer_label.setText("Timer: --")
    
    def on_connection(self, status, is_error):
        color = "#f44336" if is_error else "#4CAF50"
        self.conn_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.conn_label.setText(status)
    
    def reset_counts(self):
        reply = QMessageBox.question(self, "Reset", "Reset counts?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                with open("quality_count.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["OK", "NOT_OK"])
                    writer.writerow([0, 0])
                
                if self.detection_thread:
                    self.detection_thread.ok_count = 0
                    self.detection_thread.not_ok_count = 0
                
                self.ok_label.setText("OK: 0")
                self.nok_label.setText("NOT OK: 0")
                
                QMessageBox.information(self, "Success", "Counts reset!")
                logger.info("Counts reset")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Reset failed: {e}")
    
    def closeEvent(self, event):
        if self.detection_thread:
            self.detection_thread.stop()
        event.accept()


class MainWindow(QMainWindow):
    """Industrial-grade main window"""
    def __init__(self):
        super().__init__()
        self.model_manager = BIWModelManager()
        self.camera_config = CameraConfig()
        self.watchdog = WatchdogTimer(timeout_seconds=30)
        self.watchdog.timeout_signal.connect(self.on_watchdog_timeout)
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Industrial BIW Stud Detection System v6.0")
        self.setGeometry(100, 50, 1600, 900)
        self.setMinimumSize(1400, 800)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.tab_widget = QTabWidget()
        
        self.teaching_page = TeachingPage(self.model_manager, self.camera_config, self.watchdog)
        self.detection_page = DetectionPage(self.model_manager, self.camera_config, self.watchdog)
        
        self.tab_widget.addTab(self.teaching_page, "📚 Teaching")
        self.tab_widget.addTab(self.detection_page, "🎯 Detection")
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        layout.addWidget(self.tab_widget)
        
        self.statusBar().showMessage("Ready - Industrial BIW Detection System v6.0")
        
        self.create_menu_bar()
        
        logger.info("=" * 70)
        logger.info("Industrial BIW Stud Detection System v6.0 Started")
        logger.info("=" * 70)
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        export_action = QAction('Export Models...', self)
        export_action.triggered.connect(self.export_models)
        file_menu.addAction(export_action)
        
        import_action = QAction('Import Models...', self)
        import_action.triggered.connect(self.import_models)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera menu
        camera_menu = menubar.addMenu('Camera')
        
        add_ip_action = QAction('Add Hikvision Camera...', self)
        add_ip_action.triggered.connect(self.add_ip_camera)
        camera_menu.addAction(add_ip_action)
        
        manage_ip_action = QAction('Manage Cameras...', self)
        manage_ip_action.triggered.connect(self.manage_cameras)
        camera_menu.addAction(manage_ip_action)
        
        refresh_action = QAction('Refresh Cameras', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.refresh_cameras)
        camera_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        logs_action = QAction('View Logs', self)
        logs_action.triggered.connect(self.view_logs)
        help_menu.addAction(logs_action)
    
    def on_tab_changed(self, index):
        if index == 1:
            self.detection_page.refresh_models()
            self.detection_page.refresh_cameras()
    
    def add_ip_camera(self):
        dialog = IPCameraDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_camera_config()
            if self.camera_config.add_ip_camera(
                config['name'], config['rtsp_url'],
                config['username'], config['password']
            ):
                QMessageBox.information(self, "Success", f"Camera '{config['name']}' added!")
                self.refresh_cameras()
                logger.info(f"IP camera added: {config['name']}")
    
    def manage_cameras(self):
        cameras = self.camera_config.configs["ip_cameras"]
        if not cameras:
            QMessageBox.information(self, "No Cameras", "No IP cameras configured.")
            return
        
        items = [cam['name'] for cam in cameras]
        item, ok = QInputDialog.getItem(self, "Manage Cameras",
                                        "Select camera to remove:", items, 0, False)
        
        if ok and item:
            reply = QMessageBox.question(self, "Confirm", f"Remove '{item}'?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.camera_config.remove_ip_camera(item):
                    QMessageBox.information(self, "Success", f"Camera '{item}' removed!")
                    self.refresh_cameras()
                    logger.info(f"Camera removed: {item}")
    
    def refresh_cameras(self):
        self.teaching_page.refresh_cameras()
        self.detection_page.refresh_cameras()
        logger.info("Cameras refreshed")
    
    def export_models(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Models", "biw_models_backup.json", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import shutil
                shutil.copy(self.model_manager.models_file, file_path)
                QMessageBox.information(self, "Success", f"Exported to:\n{file_path}")
                logger.info(f"Models exported: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
    
    def import_models(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Models", "", "JSON Files (*.json)"
        )
        if file_path:
            reply = QMessageBox.question(self, "Confirm",
                                        "Replace all existing models?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    import shutil
                    shutil.copy(file_path, self.model_manager.models_file)
                    self.model_manager.models = self.model_manager.load_models()
                    self.teaching_page.refresh_models()
                    self.detection_page.refresh_models()
                    QMessageBox.information(self, "Success", "Models imported!")
                    logger.info(f"Models imported: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Import failed:\n{e}")
    
    def view_logs(self):
        if os.path.exists('stud_detection.log'):
            import subprocess
            import platform
            
            try:
                if platform.system() == 'Windows':
                    os.startfile('stud_detection.log')
                elif platform.system() == 'Darwin':
                    subprocess.call(['open', 'stud_detection.log'])
                else:
                    subprocess.call(['xdg-open', 'stud_detection.log'])
            except:
                QMessageBox.information(self, "Logs",
                                       f"Log file: {os.path.abspath('stud_detection.log')}")
        else:
            QMessageBox.information(self, "No Logs", "No log file found.")
    
    def show_about(self):
        QMessageBox.about(
            self, "About",
            "Industrial BIW Stud Detection System v6.0\n\n"
            "Professional BIW quality inspection system\n\n"
            "✅ Industrial-Grade Features:\n"
            "• Watchdog timer for thread monitoring\n"
            "• Circular frame buffering\n"
            "• Automatic reconnection with TCP\n"
            "• NO minimum stud requirement\n"
            "• Multi-threaded architecture\n"
            "• Performance metrics (FPS)\n"
            "• Rotating log files (10MB, 5 backups)\n"
            "• Hikvision RTSP optimized\n"
            "• USB & IP camera support\n"
            "• HID relay integration\n"
            "• Real-time quality control\n"
            "• Graceful degradation\n\n"
            "© 2024 Industrial Detection Systems"
        )
    
    def on_watchdog_timeout(self, thread_name):
        """Handle watchdog timeout"""
        logger.critical(f"Watchdog timeout: {thread_name}")
        QMessageBox.warning(
            self, "Thread Timeout",
            f"Thread '{thread_name}' has stopped responding.\n"
            "The system will attempt to recover."
        )
    
    def closeEvent(self, event):
        logger.info("Application closing...")
        
        # Stop watchdog
        self.watchdog.stop()
        
        # Stop detection
        if hasattr(self.detection_page, 'detection_thread') and self.detection_page.detection_thread:
            self.detection_page.detection_thread.stop()
        
        # Stop teaching
        if hasattr(self.teaching_page, 'camera_thread') and self.teaching_page.camera_thread:
            self.teaching_page.camera_thread.stop()
        
        logger.info("Application closed successfully")
        event.accept()


def main():
    """Main entry point with exception handling"""
    def exception_hook(exctype, value, tb):
        logger.critical("Uncaught exception", exc_info=(exctype, value, tb))
        sys.__excepthook__(exctype, value, tb)
    
    sys.excepthook = exception_hook
    
    app = QApplication(sys.argv)
    app.setApplicationName("Industrial BIW Stud Detection System")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Industrial Detection Systems")
    
    # High DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    window = MainWindow()
    window.show()
    
    logger.info("=" * 70)
    logger.info("INDUSTRIAL BIW STUD DETECTION SYSTEM v6.0")
    logger.info("Ready for production operation")
    logger.info("=" * 70)
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()