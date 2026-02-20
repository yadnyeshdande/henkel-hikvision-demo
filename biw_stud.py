import sys 
import os 
import csv 
import datetime 
import time 
import json 
import threading 
import queue 
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
import cv2 
import numpy as np 
import logging 
 
# Configure logging 
logging.basicConfig( 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[ 
        logging.FileHandler('stud_detection.log'), 
        logging.StreamHandler() 
    ] 
) 
logger = logging.getLogger(__name__) 
 
# Import your existing modules 
try: 
    from logic.stud_detection import detect_studs 
    from logic.reference_positions import get_reference_positions 
    from logic.stud_analysis import find_missing_and_extra_studs 
except ImportError as e: 
    logger.warning(f"Could not import detection modules: {e}") 
    # Provide fallback functions 
    def detect_studs(frame): 
        return [] 
    def get_reference_positions(): 
        return [] 
    def find_missing_and_extra_studs(ref, det): 
        return [], ref, det 
 
try: 
    import pyhid_usb_relay 
except ImportError: 
    logger.warning("pyhid_usb_relay not available. Relay control disabled.") 
    pyhid_usb_relay = None 
 
 
class CameraConfig: 
    """Camera configuration storage""" 
    def __init__(self): 
        self.config_file = "camera_config.json" 
        self.configs = self.load_configs() 
     
    def load_configs(self): 
        """Load camera configurations""" 
        if os.path.exists(self.config_file): 
            try: 
                with open(self.config_file, 'r') as f: 
                    return json.load(f) 
            except Exception as e: 
                logger.error(f"Error loading camera configs: {e}") 
        return {"usb_cameras": [], "ip_cameras": []} 
     
    def save_configs(self): 
        """Save camera configurations""" 
        try: 
            with open(self.config_file, 'w') as f: 
                json.dump(self.configs, f, indent=2) 
            return True 
        except Exception as e: 
            logger.error(f"Error saving camera configs: {e}") 
            return False 
     
    def add_ip_camera(self, name, rtsp_url, username="", password=""): 
        """Add IP camera configuration""" 
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
        """Remove IP camera by name""" 
        self.configs["ip_cameras"] = [ 
            cam for cam in self.configs["ip_cameras"] if cam["name"] != name 
        ] 
        return self.save_configs() 
     
    def get_all_cameras(self): 
        """Get all camera configurations""" 
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
    """Dialog for adding IP camera""" 
    def __init__(self, parent=None): 
        super().__init__(parent) 
        self.setWindowTitle("Add IP Camera") 
        self.setModal(True) 
        self.resize(500, 400) 
        self.init_ui() 
     
    def init_ui(self): 
        layout = QVBoxLayout(self) 
         
        # Title 
        title = QLabel("Add IP/RTSP Camera📹Add IP/RTSP Camera") 
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3; padding: 10px;") 
        layout.addWidget(title) 
         
        # Camera name 
        layout.addWidget(QLabel("Camera Name:")) 
        self.name_input = QLineEdit() 
        self.name_input.setPlaceholderText("e.g., Front Camera, Station 1, etc.") 
        self.name_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;") 
        layout.addWidget(self.name_input) 
         
        # RTSP URL 
        layout.addWidget(QLabel("RTSP URL:")) 
        self.rtsp_input = QLineEdit() 
        self.rtsp_input.setPlaceholderText("rtsp://192.168.1.64:554/Streaming/Channels/101") 
        self.rtsp_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;") 
        layout.addWidget(self.rtsp_input) 
         
        # Username 
        layout.addWidget(QLabel("Username (optional):")) 
        self.username_input = QLineEdit() 
        self.username_input.setPlaceholderText("admin") 
        self.username_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;") 
        layout.addWidget(self.username_input) 
         
        # Password 
        layout.addWidget(QLabel("Password (optional):")) 
        self.password_input = QLineEdit() 
        self.password_input.setEchoMode(QLineEdit.Password) 
        self.password_input.setPlaceholderText("password") 
        self.password_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;") 
        layout.addWidget(self.password_input) 
         
        # URL format help 
        help_text = QLabel("RTSP URL Format Examples:\n" 
            "• Hikvision: rtsp://username:password@ip:554/Streaming/Channels/101\n" 
            "• Generic: rtsp://ip:port/stream\n" 
            "• With auth: rtsp://user:pass@ip:port/stream" 
        ) 
        help_text.setStyleSheet("background-color: #f0f8ff; padding: 10px; border-radius: 5px; font-size: 11px;") 
        help_text.setWordWrap(True) 
        layout.addWidget(help_text) 
         
        # Test connection button 
        test_btn = QPushButton("Test Connection") 
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
            QPushButton:hover { 
                background-color: #f57c00; 
            } 
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
        rtsp_url = self.get_rtsp_url() 
        if not rtsp_url: 
            QMessageBox.warning(self, "Invalid Input", "Please enter RTSP URL.") 
            return 
         
        QApplication.setOverrideCursor(Qt.WaitCursor) 
        try: 
            cap = cv2.VideoCapture(rtsp_url) 
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
             
            # Try to read a frame 
            for _ in range(30):  # Try up to 30 attempts 
                ret, frame = cap.read() 
                if ret and frame is not None: 
                    cap.release() 
                    QApplication.restoreOverrideCursor() 
                    QMessageBox.information(self, "Success", "✓ Connection successful!\nCamera is responding.") 
                    return 
                time.sleep(0.1) 
             
            cap.release() 
            QApplication.restoreOverrideCursor() 
            QMessageBox.warning(self, "Connection Failed", "Could not read frames from camera.\nPlease check the URL and credentials.") 
             
        except Exception as e: 
            QApplication.restoreOverrideCursor() 
            QMessageBox.critical(self, "Error", f"Connection test failed:\n{str(e)}") 
     
    def get_rtsp_url(self): 
        """Build RTSP URL with credentials if provided""" 
        base_url = self.rtsp_input.text().strip() 
        username = self.username_input.text().strip() 
        password = self.password_input.text().strip() 
         
        if not base_url: 
            return "" 
         
        # If username and password provided, inject them 
        if username and password: 
            if "://" in base_url: 
                protocol, rest = base_url.split("://", 1) 
                # Remove existing credentials if any 
                if "@" in rest: 
                    rest = rest.split("@", 1)[1] 
                return f"{protocol}://{username}:{password}@{rest}" 
         
        return base_url 
     
    def get_camera_config(self): 
        """Get camera configuration""" 
        return { 
            "name": self.name_input.text().strip(), 
            "rtsp_url": self.get_rtsp_url(), 
            "username": self.username_input.text().strip(), 
            "password": self.password_input.text().strip() 
        } 
 
 
class CameraThread(QThread): 
    """Robust camera thread supporting USB and IP cameras""" 
    frame_ready = pyqtSignal(object) 
    connection_status = pyqtSignal(str, bool)  # status message, is_error 
     
    def __init__(self, camera_source, is_ip=False): 
        super().__init__() 
        self.camera_source = camera_source 
        self.is_ip = is_ip 
        self.running = True 
        self.camera = None 
        self.reconnect_attempts = 0 
        self.max_reconnect_attempts = 5 
        self.reconnect_delay = 2 
        self.frame_queue = queue.Queue(maxsize=2) 
     
    def run(self): 
        """Main camera loop with reconnection logic""" 
        while self.running: 
            if not self.connect_camera(): 
                if self.running: 
                    time.sleep(self.reconnect_delay) 
                continue 
             
            # Camera connected, start reading frames 
            self.read_frames() 
             
            # If we exit read_frames and still running, attempt reconnect 
            if self.running: 
                logger.warning(f"Camera disconnected. Attempting to reconnect...") 
                self.connection_status.emit("Camera disconnected. Reconnecting...", True) 
                time.sleep(self.reconnect_delay) 
     
    def connect_camera(self): 
        """Connect to camera with retry logic""" 
        try: 
            logger.info(f"Connecting to camera: {self.camera_source}") 
            self.connection_status.emit("Connecting to camera...", False) 
             
            if self.is_ip: 
                self.camera = cv2.VideoCapture(self.camera_source) 
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency 
            else: 
                self.camera = cv2.VideoCapture(self.camera_source) 
             
            # Set camera properties for better performance 
            if self.camera.isOpened(): 
                # Try to set resolution and FPS 
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
                self.camera.set(cv2.CAP_PROP_FPS, 30) 
                 
                # Verify we can read a frame 
                for _ in range(10): 
                    ret, frame = self.camera.read() 
                    if ret and frame is not None: 
                        logger.info("Camera connected successfully") 
                        self.connection_status.emit("Camera connected", False) 
                        self.reconnect_attempts = 0 
                        return True 
                    time.sleep(0.1) 
             
            # Connection failed 
            self.reconnect_attempts += 1 
            error_msg = f"Failed to connect (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})" 
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
            logger.error(f"Camera connection error: {e}") 
            self.connection_status.emit(f"Connection error: {str(e)}", True) 
            if self.camera: 
                self.camera.release() 
                self.camera = None 
            return False 
     
    def read_frames(self): 
        """Read frames from camera""" 
        frame_skip_count = 0 
        consecutive_failures = 0 
         
        while self.running and self.camera and self.camera.isOpened(): 
            try: 
                ret, frame = self.camera.read() 
                 
                if not ret or frame is None: 
                    consecutive_failures += 1 
                    if consecutive_failures > 30:  # 30 consecutive failures 
                        logger.error("Too many consecutive frame read failures") 
                        break 
                    time.sleep(0.01) 
                    continue 
                 
                consecutive_failures = 0 
                 
                # Skip frames for IP cameras to reduce latency 
                if self.is_ip: 
                    frame_skip_count += 1 
                    if frame_skip_count % 2 != 0:  # Skip every other frame 
                        continue 
                 
                # Emit frame 
                self.frame_ready.emit(frame.copy()) 
                 
                # Small delay to prevent overloading 
                self.msleep(33)  # ~30 FPS 
                 
            except Exception as e: 
                logger.error(f"Frame read error: {e}") 
                break 
     
    def stop(self): 
        """Stop the camera thread""" 
        logger.info("Stopping camera thread") 
        self.running = False 
        self.quit() 
        self.wait(3000)  # Wait up to 3 seconds 
        if self.camera: 
            self.camera.release() 
            self.camera = None 
 
 
class BIWModelManager: 
    """Manager for BIW car models and their stud configurations""" 
    def __init__(self): 
        self.models_file = "biw_models.json" 
        self.models = self.load_models() 
     
    def load_models(self): 
        """Load BIW models from file""" 
        if os.path.exists(self.models_file): 
            try: 
                with open(self.models_file, 'r') as f: 
                    return json.load(f) 
            except Exception as e: 
                logger.error(f"Error loading models: {e}") 
        return {} 
     
    def save_models(self): 
        """Save BIW models to file""" 
        try: 
            with open(self.models_file, 'w') as f: 
                json.dump(self.models, f, indent=2) 
            return True 
        except Exception as e: 
            logger.error(f"Error saving models: {e}") 
            return False 
     
    def add_model(self, name, stud_count, positions, description=""): 
        """Add a new BIW model""" 
        self.models[name] = { 
            "stud_count": stud_count, 
            "positions": positions, 
            "description": description, 
            "created_at": datetime.datetime.now().isoformat() 
        } 
        return self.save_models() 
     
    def get_model(self, name): 
        """Get a specific BIW model""" 
        return self.models.get(name) 
     
    def get_model_names(self): 
        """Get list of all model names""" 
        return list(self.models.keys()) 
     
    def delete_model(self, name): 
        """Delete a BIW model""" 
        if name in self.models: 
            del self.models[name] 
            return self.save_models() 
        return False 
 
 
class TeachingCanvas(QLabel): 
    """Interactive canvas for teaching stud positions with camera preview""" 
    positions_updated = pyqtSignal(list) 
     
    def __init__(self): 
        super().__init__() 
        self.setMinimumSize(640, 480) 
        self.setStyleSheet("border: 2px solid #333; background-color: white;") 
        self.setAlignment(Qt.AlignCenter) 
         
        # Drawing state 
        self.drawing = False 
        self.current_rect = None 
        self.start_point = None 
        self.rectangles = [] 
        self.current_frame = None 
        self.captured_image = None 
         
        # Set up for mouse events 
        self.setMouseTracking(True) 
         
        # Display initial message 
        self.setText("📷Start camera preview to begin teaching stud positions") 
     
    def update_camera_frame(self, frame): 
        """Update with camera frame""" 
        self.current_frame = frame.copy() 
        if self.captured_image is None: 
            self.update_display() 
     
    def capture_teaching_image(self): 
        """Capture current frame for teaching""" 
        if self.current_frame is not None: 
            self.captured_image = self.current_frame.copy() 
            self.rectangles = [] 
            self.update_display() 
            return True 
        return False 
     
    def clear_capture(self): 
        """Clear captured image and return to live preview""" 
        self.captured_image = None 
        self.rectangles = [] 
        if self.current_frame is not None: 
            self.update_display() 
     
    def update_display(self): 
        """Update the display with current image and rectangles""" 
        display_image = None 
         
        if self.captured_image is not None: 
            display_image = self.captured_image.copy() 
             
            # Draw existing rectangles 
            for i, rect in enumerate(self.rectangles): 
                cv2.rectangle(display_image, 
                            (rect['x'], rect['y']), 
                            (rect['x'] + rect['width'], rect['y'] + rect['height']), 
                            (0, 255, 0), 2) 
                cv2.putText(display_image, f"S{i+1}", 
                          (rect['x'] + 5, rect['y'] + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
             
            # Draw current rectangle being drawn 
            if self.current_rect: 
                cv2.rectangle(display_image, 
                            (self.current_rect['x'], self.current_rect['y']), 
                            (self.current_rect['x'] + self.current_rect['width'], 
                             self.current_rect['y'] + self.current_rect['height']), 
                            (255, 0, 0), 2) 
             
            cv2.putText(display_image, "Draw rectangles around studs - Teaching Mode",  
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) 
                        
        elif self.current_frame is not None: 
            display_image = self.current_frame.copy() 
            cv2.putText(display_image, "Live Preview - Click 'Capture Image' to start teaching",  
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
         
        if display_image is not None: 
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) 
            h, w, ch = rgb_image.shape 
            bytes_per_line = ch * w 
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888) 
             
            pixmap = QPixmap.fromImage(qt_image) 
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) 
            self.setPixmap(scaled_pixmap) 
             
            if self.captured_image is not None: 
                positions = [(rect['x'] + rect['width']//2, rect['y'] + rect['height']//2) 
                           for rect in self.rectangles] 
                self.positions_updated.emit(positions) 
     
    def mousePressEvent(self, event): 
        """Handle mouse press for drawing rectangles""" 
        if event.button() == Qt.LeftButton and self.captured_image is not None: 
            pos = self.map_to_image_coords(event.pos()) 
            if pos: 
                self.drawing = True 
                self.start_point = pos 
                self.current_rect = { 
                    'x': pos.x(), 
                    'y': pos.y(), 
                    'width': 0, 
                    'height': 0 
                } 
     
    def mouseMoveEvent(self, event): 
        """Handle mouse move for drawing rectangles""" 
        if self.drawing and self.current_rect and self.start_point: 
            pos = self.map_to_image_coords(event.pos()) 
            if pos: 
                self.current_rect['width'] = pos.x() - self.start_point.x() 
                self.current_rect['height'] = pos.y() - self.start_point.y() 
                self.update_display() 
     
    def mouseReleaseEvent(self, event): 
        """Handle mouse release to finish drawing rectangle""" 
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
        """Map widget coordinates to image coordinates""" 
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
        else: 
            return None 
             
        return QPoint(img_x, img_y) 
     
    def clear_rectangles(self): 
        """Clear all drawn rectangles""" 
        self.rectangles = [] 
        if self.captured_image is not None: 
            self.update_display() 
     
    def get_positions(self): 
        """Get center positions of all rectangles""" 
        return [(rect['x'] + rect['width']//2, rect['y'] + rect['height']//2) 
                for rect in self.rectangles] 
 
 
class TeachingPage(QWidget): 
    """Page for teaching BIW car models with camera preview""" 
    def __init__(self, model_manager, camera_config): 
        super().__init__() 
        self.model_manager = model_manager 
        self.camera_config = camera_config 
        self.camera_thread = None 
        self.selected_camera = None 
        self.init_ui() 
     
    def init_ui(self): 
        """Initialize the teaching page UI""" 
        layout = QVBoxLayout(self) 
        layout.setSpacing(10) 
        layout.setContentsMargins(10, 10, 10, 10) 
         
        # Header 
        header = QLabel("📚 BIW Model Teaching") 
        header.setStyleSheet(""" 
            font-size: 20px; 
            font-weight: bold; 
            color: #2196F3; 
            padding: 10px; 
            background-color: #f0f8ff; 
            border-radius: 8px; 
            border: 2px solid #2196F3; 
        """) 
        header.setAlignment(Qt.AlignCenter) 
        layout.addWidget(header) 
         
        # Main content 
        content_layout = QHBoxLayout() 
         
        # Left side - Canvas 
        left_layout = QVBoxLayout() 
         
        # Canvas controls 
        canvas_controls = QHBoxLayout() 
         
        canvas_controls.addWidget(QLabel("Camera:")) 
        self.camera_combo = QComboBox() 
        self.camera_combo.setMinimumWidth(200) 
        self.refresh_cameras() 
        canvas_controls.addWidget(self.camera_combo) 

        self.start_camera_btn = QPushButton("📹 Start") 
        self.start_camera_btn.clicked.connect(self.start_camera_preview) 
        self.start_camera_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                padding: 8px 15px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #45a049; } 
        """) 
        canvas_controls.addWidget(self.start_camera_btn) 
         
        self.stop_camera_btn = QPushButton("⏹Stop") 
        self.stop_camera_btn.clicked.connect(self.stop_camera_preview) 
        self.stop_camera_btn.setEnabled(False) 
        self.stop_camera_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                padding: 8px 15px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #da190b; } 
        """) 
        canvas_controls.addWidget(self.stop_camera_btn) 
         
        canvas_controls.addStretch() 
        left_layout.addLayout(canvas_controls) 
         
        # Connection status 
        self.connection_status_label = QLabel("Status: Ready") 
        self.connection_status_label.setStyleSheet("padding: 5px; font-size: 11px; color: #666;") 
        left_layout.addWidget(self.connection_status_label) 
         
        # Teaching controls 
        teaching_controls = QHBoxLayout() 
         
        self.capture_btn = QPushButton("📸Capture Image") 
        self.capture_btn.clicked.connect(self.capture_teaching_image) 
        self.capture_btn.setEnabled(False) 
        self.capture_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #FF9800; 
                color: white; 
                padding: 8px 15px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #f57c00; } 
        """) 
        teaching_controls.addWidget(self.capture_btn) 
         
        clear_capture_btn = QPushButton("🔄Clear Capture") 
        clear_capture_btn.clicked.connect(self.clear_capture) 
        clear_capture_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #9C27B0; 
                color: white; 
                padding: 8px 15px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #7B1FA2; } 
        """) 
        teaching_controls.addWidget(clear_capture_btn) 
         
        clear_btn = QPushButton("🗑Clear All") 
        clear_btn.clicked.connect(self.clear_all_positions) 
        clear_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                padding: 8px 15px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #da190b; } 
        """) 
        teaching_controls.addWidget(clear_btn) 
         
        teaching_controls.addStretch() 
        left_layout.addLayout(teaching_controls) 
         
        # Teaching canvas 
        self.teaching_canvas = TeachingCanvas() 
        self.teaching_canvas.positions_updated.connect(self.update_position_count) 
        left_layout.addWidget(self.teaching_canvas) 
         
        content_layout.addLayout(left_layout, 3) 
         
        # Right side - Controls 
        right_layout = QVBoxLayout() 
         
        # Current positions info 
        positions_group = QGroupBox("📍Current Positions") 
        positions_group.setStyleSheet(self.get_group_style()) 
        positions_layout = QVBoxLayout(positions_group) 
         
        self.position_count_label = QLabel("Studs marked: 0") 
        self.position_count_label.setStyleSheet("font-size: 13px; color: #333; font-weight: bold;") 
        positions_layout.addWidget(self.position_count_label) 
         
        instructions = QLabel( 
            "• Start camera preview\n" 
            "• Capture image when BIW positioned\n" 
            "• Draw rectangles around studs\n" 
            "• Save model when complete" 
        ) 
        instructions.setStyleSheet("font-size: 11px; color: #666; margin: 8px 0;") 
        instructions.setWordWrap(True) 
        positions_layout.addWidget(instructions) 
         
        right_layout.addWidget(positions_group) 
         
        # Model configuration 
        config_group = QGroupBox("⚙ Model Configuration") 
        config_group.setStyleSheet(self.get_group_style()) 
        config_layout = QVBoxLayout(config_group) 
         
        config_layout.addWidget(QLabel("Model Name:")) 
        self.model_name_input = QLineEdit() 
        self.model_name_input.setPlaceholderText("e.g., Non-AC, AC, Luxury") 
        self.model_name_input.setStyleSheet("padding: 6px; border: 1px solid #ccc; border-radius: 4px;") 
        config_layout.addWidget(self.model_name_input) 
         
        config_layout.addWidget(QLabel("Description:")) 
        self.description_input = QTextEdit() 
        self.description_input.setMaximumHeight(60) 
        self.description_input.setPlaceholderText("Brief description...") 
        self.description_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 4px;") 
        config_layout.addWidget(self.description_input) 
         
        save_btn = QPushButton("💾 Save Model") 
        save_btn.clicked.connect(self.save_current_model) 
        save_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                padding: 10px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #1976D2; } 
        """) 
        config_layout.addWidget(save_btn) 
         
        right_layout.addWidget(config_group) 
         
        # Existing models 
        models_group = QGroupBox("📋Existing Models") 
        models_group.setStyleSheet(self.get_group_style()) 
        models_layout = QVBoxLayout(models_group) 
         
        self.models_list = QListWidget() 
        self.models_list.setMaximumHeight(150) 
        self.models_list.setStyleSheet("border: 1px solid #ccc; border-radius: 4px;") 
        self.refresh_models_list() 
        models_layout.addWidget(self.models_list) 
         
        models_buttons = QHBoxLayout() 
         
        delete_btn = QPushButton(" 🗑 Delete") 
        delete_btn.clicked.connect(self.delete_selected_model) 
        delete_btn.setStyleSheet("background-color: #f44336; color: white; padding: 5px 10px; border-radius: 3px;") 
        models_buttons.addWidget(delete_btn) 
         
        models_layout.addLayout(models_buttons) 
        right_layout.addWidget(models_group) 
         
        right_layout.addStretch() 
        content_layout.addLayout(right_layout, 1) 
        layout.addLayout(content_layout) 
     
    def get_group_style(self): 
        """Get consistent group box style""" 
        return """ 
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
        """ 
     
    def refresh_cameras(self): 
        """Refresh available cameras""" 
        self.camera_combo.clear() 
        cameras = self.camera_config.get_all_cameras() 
         
        for cam in cameras: 
            if cam["type"] == "usb": 
                self.camera_combo.addItem(f"🔌 {cam['name']}", cam) 
            else: 
                self.camera_combo.addItem(f"🌐 {cam['name']}", cam) 
         
        if self.camera_combo.count() == 0: 
            self.camera_combo.addItem("No cameras available") 
     
    def start_camera_preview(self): 
        """Start camera preview""" 
        if self.camera_thread and self.camera_thread.isRunning(): 
            return 
         
        camera_data = self.camera_combo.currentData() 
        if not camera_data: 
            QMessageBox.warning(self, "Error", "No camera selected!") 
            return 
         
        self.selected_camera = camera_data 
         
        if camera_data["type"] == "usb": 
            camera_source = camera_data["index"] 
            is_ip = False 
        else: 
            camera_source = camera_data["rtsp_url"] 
            is_ip = True 
         
        # Create and start camera thread 
        self.camera_thread = CameraThread(camera_source, is_ip) 
        self.camera_thread.frame_ready.connect(self.teaching_canvas.update_camera_frame) 
        self.camera_thread.connection_status.connect(self.update_connection_status) 
        self.camera_thread.start() 
         
        # Update UI 
        self.start_camera_btn.setEnabled(False) 
        self.stop_camera_btn.setEnabled(True) 
        self.capture_btn.setEnabled(True) 
         
        logger.info(f"Started camera preview: {camera_data['name']}") 
     
    def stop_camera_preview(self): 
        """Stop camera preview""" 
        if self.camera_thread: 
            self.camera_thread.stop() 
            self.camera_thread = None 
         
        self.start_camera_btn.setEnabled(True) 
        self.stop_camera_btn.setEnabled(False) 
        self.capture_btn.setEnabled(False) 
        self.teaching_canvas.setText("📷 Start camera preview to begin") 
        self.connection_status_label.setText("Status: Stopped") 
        self.connection_status_label.setStyleSheet("padding: 5px; font-size: 11px; color: #666;") 
         
        logger.info("Stopped camera preview") 
     
    def update_connection_status(self, status, is_error): 
        """Update connection status label""" 
        if is_error: 
            self.connection_status_label.setStyleSheet("padding: 5px; font-size: 11px; color: #f44336; font-weight: bold;") 
        else: 
            self.connection_status_label.setStyleSheet("padding: 5px; font-size: 11px; color: #4CAF50; font-weight: bold;") 
        self.connection_status_label.setText(f"Status: {status}") 
     
    def capture_teaching_image(self): 
        """Capture current frame for teaching""" 
        if self.teaching_canvas.capture_teaching_image(): 
            QMessageBox.information(self, "Image Captured", "Image captured! Draw rectangles around studs.") 
            logger.info("Teaching image captured") 
        else: 
            QMessageBox.warning(self, "Error", "Failed to capture. Ensure camera is running.") 
     
    def clear_capture(self): 
        """Clear captured image""" 
        self.teaching_canvas.clear_capture() 
     
    def clear_all_positions(self): 
        """Clear all marked positions""" 
        self.teaching_canvas.clear_rectangles() 
     
    def update_position_count(self, positions): 
        """Update position count display""" 
        self.position_count_label.setText(f"Studs marked: {len(positions)}") 
     
    def save_current_model(self): 
        """Save current model configuration""" 
        model_name = self.model_name_input.text().strip() 
        if not model_name: 
            QMessageBox.warning(self, "Error", "Please enter model name.") 
            return 
         
        positions = self.teaching_canvas.get_positions() 
        if not positions: 
            QMessageBox.warning(self, "Error", "Please mark at least one stud.") 
            return 
         
        description = self.description_input.toPlainText().strip() 
         
        if model_name in self.model_manager.get_model_names(): 
            reply = QMessageBox.question( 
                self, "Model Exists", 
                f"Overwrite '{model_name}'?", 
                QMessageBox.Yes | QMessageBox.No 
            ) 
            if reply != QMessageBox.Yes: 
                return 
         
        if self.model_manager.add_model(model_name, len(positions), positions, description): 
            QMessageBox.information(self, "Success", f"Model '{model_name}' saved with {len(positions)} studs!") 
            self.refresh_models_list() 
            self.model_name_input.clear() 
            self.description_input.clear() 
            logger.info(f"Saved model: {model_name} with {len(positions)} studs") 
        else: 
            QMessageBox.critical(self, "Error", "Failed to save model.") 
     
    def refresh_models_list(self): 
        """Refresh models list""" 
        self.models_list.clear() 
        for model_name in self.model_manager.get_model_names(): 
            model = self.model_manager.get_model(model_name) 
            self.models_list.addItem(f"{model_name} ({model['stud_count']} studs)") 
     
    def delete_selected_model(self): 
        """Delete selected model""" 
        current_item = self.models_list.currentItem() 
        if not current_item: 
            QMessageBox.information(self, "No Selection", "Please select a model.") 
            return 
         
        model_name = current_item.text().split(" (")[0] 
         
        reply = QMessageBox.question( 
            self, "Confirm Delete", 
            f"Delete model '{model_name}'?", 
            QMessageBox.Yes | QMessageBox.No 
        ) 
         
        if reply == QMessageBox.Yes: 
            if self.model_manager.delete_model(model_name): 
                QMessageBox.information(self, "Success", f"Model '{model_name}' deleted!") 
                self.refresh_models_list() 
                logger.info(f"Deleted model: {model_name}") 
     
    def closeEvent(self, event): 
        """Handle page closing""" 
        if self.camera_thread: 
            self.camera_thread.stop() 
        event.accept() 
 
 
class ROISelectionDialog(QDialog): 
    """Dialog for setting detection ROI""" 
    def __init__(self, parent=None): 
        super().__init__(parent) 
        self.setWindowTitle("Set Detection ROI") 
        self.setModal(True) 
        self.resize(400, 250) 
        self.init_ui() 
     
    def init_ui(self): 
        layout = QVBoxLayout(self) 
         
        instructions = QLabel("Set Region of Interest for detection:") 
        instructions.setStyleSheet("padding: 8px; background-color: #f0f8ff; border-radius: 5px;") 
        layout.addWidget(instructions) 
         
        roi_group = QGroupBox("ROI Coordinates") 
        roi_layout = QGridLayout(roi_group) 
         
        roi_layout.addWidget(QLabel("X:"), 0, 0) 
        self.x_spin = QSpinBox() 
        self.x_spin.setRange(0, 3840) 
        self.x_spin.setValue(100) 
        roi_layout.addWidget(self.x_spin, 0, 1) 
         
        roi_layout.addWidget(QLabel("Y:"), 0, 2) 
        self.y_spin = QSpinBox() 
        self.y_spin.setRange(0, 2160) 
        self.y_spin.setValue(100) 
        roi_layout.addWidget(self.y_spin, 0, 3) 
         
        roi_layout.addWidget(QLabel("Width:"), 1, 0) 
        self.width_spin = QSpinBox() 
        self.width_spin.setRange(50, 3840) 
        self.width_spin.setValue(640) 
        roi_layout.addWidget(self.width_spin, 1, 1) 
         
        roi_layout.addWidget(QLabel("Height:"), 1, 2) 
        self.height_spin = QSpinBox() 
        self.height_spin.setRange(50, 2160) 
        self.height_spin.setValue(480) 
        roi_layout.addWidget(self.height_spin, 1, 3) 
         
        layout.addWidget(roi_group) 
         
        preset_group = QGroupBox("Presets") 
        preset_layout = QHBoxLayout(preset_group) 
         
        for name, func in [("Center", self.set_center_roi), ("Full", self.set_full_roi)]: 
            btn = QPushButton(name) 
            btn.clicked.connect(func) 
            preset_layout.addWidget(btn) 
         
        layout.addWidget(preset_group) 
         
        button_layout = QHBoxLayout() 
        ok_btn = QPushButton("OK") 
        ok_btn.clicked.connect(self.accept) 
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 20px; border-radius: 4px;") 
        button_layout.addWidget(ok_btn) 
         
        cancel_btn = QPushButton("Cancel") 
        cancel_btn.clicked.connect(self.reject) 
        cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px 20px; border-radius: 4px;") 
        button_layout.addWidget(cancel_btn) 
         
        layout.addLayout(button_layout) 
     
    def set_center_roi(self): 
        self.x_spin.setValue(320) 
        self.y_spin.setValue(240) 
        self.width_spin.setValue(640) 
        self.height_spin.setValue(480) 
     
    def set_full_roi(self): 
        self.x_spin.setValue(0) 
        self.y_spin.setValue(0) 
        self.width_spin.setValue(1280) 
        self.height_spin.setValue(720) 
     
    def get_roi(self): 
        return (self.x_spin.value(), self.y_spin.value(),  
                self.width_spin.value(), self.height_spin.value()) 
 
 
class StatusWidget(QWidget): 
    """Custom status display widget""" 
    def __init__(self, title="Status"): 
        super().__init__() 
        self.title = title 
        self.status = "Unknown" 
        self.count = 0 
        self.init_ui() 
     
    def init_ui(self): 
        layout = QVBoxLayout() 
        layout.setSpacing(5) 
        layout.setContentsMargins(5, 5, 5, 5) 
         
        self.title_label = QLabel(self.title) 
        self.title_label.setAlignment(Qt.AlignCenter) 
        self.title_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #333;") 
        layout.addWidget(self.title_label) 
         
        self.status_label = QLabel(self.status) 
        self.status_label.setAlignment(Qt.AlignCenter) 
        self.status_label.setStyleSheet( 
            "font-size: 14px; font-weight: bold; padding: 8px; " 
            "background-color: #cccccc; border-radius: 6px; color: white;" 
        ) 
        layout.addWidget(self.status_label) 
         
        self.count_label = QLabel(f"Count: {self.count}") 
        self.count_label.setAlignment(Qt.AlignCenter) 
        self.count_label.setStyleSheet("font-size: 11px; color: #666;") 
        layout.addWidget(self.count_label) 
         
        self.setLayout(layout) 
        self.setMaximumWidth(140) 
     
    def update_status(self, status, count=None): 
        self.status = status 
        if count is not None: 
            self.count = count 
         
        self.status_label.setText(status) 
        self.count_label.setText(f"Count: {self.count}") 
         
        color = "#4CAF50" if status == "OK" else "#F44336" if status == "NOT OK" else "#cccccc" 
        self.status_label.setStyleSheet( 
            f"font-size: 14px; font-weight: bold; padding: 8px; " 
            f"background-color: {color}; border-radius: 6px; color: white;" 
        ) 
 
 
class DetectionVisualizationWidget(QLabel): 
    """Visualization widget for detection results""" 
    def __init__(self): 
        super().__init__() 
        self.setMinimumSize(640, 480) 
        self.setStyleSheet("border: 2px solid #333; background-color: #f5f5f5;") 
        self.setAlignment(Qt.AlignCenter) 
        self.setText("🎯 Starting detection...") 
     
    def update_frame_with_detection(self, frame, matched, missing, extra): 
        """Update with detection results""" 
        self.display_frame(frame) 
     
    def update_frame(self, frame): 
        """Update with regular frame""" 
        self.display_frame(frame) 
     
    def display_frame(self, frame): 
        """Display frame""" 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        h, w, ch = rgb_frame.shape 
        bytes_per_line = ch * w 
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888) 
         
        pixmap = QPixmap.fromImage(qt_image) 
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) 
        self.setPixmap(scaled_pixmap) 
 
 
class DetectionThread(QThread): 
    """Detection thread with stud-triggered logic""" 
    frame_ready = pyqtSignal(object) 
    detection_ready = pyqtSignal(object, list, list, list) 
    status_update = pyqtSignal(str, int, int) 
    stud_status_update = pyqtSignal(str, int) 
    connection_status = pyqtSignal(str, bool) 
     
    def __init__(self, camera_source, is_ip, model_manager, biw_model, yolo_model_path): 
        super().__init__() 
        self.camera_source = camera_source 
        self.is_ip = is_ip 
        self.model_manager = model_manager 
        self.biw_model = biw_model 
        self.yolo_model_path = yolo_model_path 
        self.running = True 
        self.camera = None 
         
        # Detection state 
        self.state = "WAITING_FOR_STUDS" 
        self.entry_time = 0 
        self.detection_delay = 3 
        self.last_result_frame = None 
        self.studs_absent_count = 0 
        self.studs_absent_threshold = 10 
        self.min_studs_threshold = 5 
        self.detection_roi = None 
         
        # Quality tracking 
        self.csv_path = "quality_count.csv" 
        self.ok_count, self.not_ok_count = self._load_csv() 
         
        # Relay 
        self.relay = self._init_relay() 
         
        # Camera reconnection 
        self.reconnect_attempts = 0 
        self.max_reconnect_attempts = 5 
     
    def _init_relay(self): 
        """Initialize relay""" 
        try: 
            if pyhid_usb_relay: 
                return pyhid_usb_relay.find() 
        except Exception as e: 
            logger.warning(f"Relay init failed: {e}") 
        return None 
     
    def _load_csv(self): 
        """Load quality counts""" 
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
        """Update quality counts""" 
        try: 
            with open(self.csv_path, 'w', newline='') as f: 
                writer = csv.writer(f) 
                writer.writerow(["OK", "NOT_OK"]) 
                writer.writerow([self.ok_count, self.not_ok_count]) 
        except Exception as e: 
            logger.error(f"CSV update error: {e}") 
     
    def set_detection_delay(self, delay): 
        self.detection_delay = delay 
     
    def set_min_studs_threshold(self, threshold): 
        self.min_studs_threshold = threshold 
     
    def set_detection_roi(self, roi): 
        self.detection_roi = roi 
     
    def run(self): 
        """Main detection loop""" 
        while self.running: 
            if not self.connect_camera(): 
                if self.running: 
                    time.sleep(2) 
                continue 
             
            self.detection_loop() 
             
            if self.running: 
                self.connection_status.emit("Reconnecting...", True) 
                time.sleep(2) 
     
    def connect_camera(self): 
        """Connect to camera""" 
        try: 
            self.connection_status.emit("Connecting...", False) 
            self.camera = cv2.VideoCapture(self.camera_source) 
             
            if self.is_ip: 
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
             
            if self.camera.isOpened(): 
                for _ in range(10): 
                    ret, frame = self.camera.read() 
                    if ret and frame is not None: 
                        self.connection_status.emit("Connected", False) 
                        self.reconnect_attempts = 0 
                        logger.info("Camera connected") 
                        return True 
                    time.sleep(0.1) 
             
            self.reconnect_attempts += 1 
            if self.reconnect_attempts >= self.max_reconnect_attempts: 
                self.connection_status.emit("Max reconnect attempts", True) 
                self.running = False 
             
            return False 
             
        except Exception as e: 
            logger.error(f"Connection error: {e}") 
            return False 
     
    def detection_loop(self): 
        """Main detection loop""" 
        consecutive_failures = 0 
         
        while self.running and self.camera and self.camera.isOpened(): 
            try: 
                ret, frame = self.camera.read() 
                 
                if not ret or frame is None: 
                    consecutive_failures += 1 
                    if consecutive_failures > 30: 
                        break 
                    continue 
                 
                consecutive_failures = 0 
                current_time = time.time() 
                 
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
        """Detect studs in frame""" 
        # TODO: Implement actual YOLO detection 
        # Placeholder simulation 
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
         
        return len(detections) >= self.min_studs_threshold, detections 
     
    def _handle_waiting(self, frame, current_time): 
        """Handle waiting state""" 
        sufficient, detections = self._detect_studs(frame) 
         
        if sufficient: 
            self.state = "STUDS_DETECTED" 
            self.entry_time = current_time 
            self.stud_status_update.emit(f"Studs Detected ({len(detections)})", 0) 
            logger.info(f"Studs detected: {len(detections)}") 
         
        status_text = f"Waiting (Need {self.min_studs_threshold}, Found {len(detections)})" 
        display_frame = self._draw_status(frame.copy(), status_text, detections) 
        self.frame_ready.emit(display_frame) 
     
    def _handle_detected(self, frame, current_time): 
        """Handle detected state""" 
        sufficient, detections = self._detect_studs(frame) 
         
        if sufficient: 
            self.state = "TIMER_RUNNING" 
            self.stud_status_update.emit("Timer Running", self.detection_delay) 
        else: 
            self.state = "WAITING_FOR_STUDS" 
            self.stud_status_update.emit("Studs Lost", 0) 
         
        status_text = f"Detected ({len(detections)} studs)" 
        display_frame = self._draw_status(frame.copy(), status_text, detections) 
        self.frame_ready.emit(display_frame) 
     
    def _handle_timer(self, frame, current_time): 
        """Handle timer state""" 
        sufficient, detections = self._detect_studs(frame) 
         
        elapsed = current_time - self.entry_time 
        remaining = max(0, self.detection_delay - elapsed) 
         
        if not sufficient: 
            self.state = "WAITING_FOR_STUDS" 
            self.stud_status_update.emit("Studs Lost", 0) 
            return 
         
        if remaining <= 0: 
            self.state = "PHOTO_TAKEN" 
            self._capture_and_detect(frame) 
            self.stud_status_update.emit("Processing", 0) 
        else: 
            self.stud_status_update.emit("Timer Running", int(remaining) + 1) 
            status_text = f"Timer: {int(remaining) + 1}s ({len(detections)} studs)" 
            display_frame = self._draw_status(frame.copy(), status_text, detections) 
            self.frame_ready.emit(display_frame) 
     
    def _handle_photo_taken(self): 
        """Handle photo taken state""" 
        self.state = "DISPLAYING_RESULT" 
        self.stud_status_update.emit("Displaying Result", 0) 
     
    def _handle_displaying(self, frame, current_time): 
        """Handle displaying result state""" 
        sufficient, detections = self._detect_studs(frame) 
         
        if sufficient: 
            self.studs_absent_count = 0 
            if self.last_result_frame is not None: 
                self.frame_ready.emit(self.last_result_frame) 
        else: 
            self.studs_absent_count += 1 
            if self.studs_absent_count >= self.studs_absent_threshold: 
                self.state = "WAITING_FOR_STUDS" 
                self.last_result_frame = None 
                self.studs_absent_count = 0 
                self.stud_status_update.emit("Ready for Next", 0) 
                logger.info("Cycle complete") 
             
            if self.last_result_frame is not None: 
                self.frame_ready.emit(self.last_result_frame) 
     
    def _capture_and_detect(self, frame): 
        """Capture and detect""" 
        try: 
            logger.info("Capturing and detecting...") 
             
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
             
            matched, missing, extra = find_missing_and_extra_studs(reference_studs, detected_studs) 
             
            result_frame = self._draw_results(frame, reference_studs, detected_studs, matched, missing, extra) 
            self.last_result_frame = result_frame 
             
            self._handle_quality(matched, missing, len(reference_studs)) 
            self.detection_ready.emit(result_frame, matched, missing, extra) 
             
        except Exception as e: 
            logger.error(f"Detection error: {e}") 
            self.last_result_frame = frame.copy() 
     
    def _draw_status(self, frame, status_text, detections): 
        """Draw status on frame""" 
        if self.detection_roi: 
            x, y, w, h = self.detection_roi 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) 
         
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
        """Draw detection results""" 
        for r in ref: 
            cv2.circle(frame, r, 8, (255, 255, 0), 1) 
 
        for d, r in matched: 
            cv2.circle(frame, d, 12, (0, 255, 0), 2) 
 
        for m in missing: 
            cv2.circle(frame, m, 12, (0, 0, 255), 2) 
 
        for e in extra: 
            cv2.circle(frame, e, 12, (255, 0, 255), 2) 
 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        expected = len(ref) 
 
        if len(matched) == expected and len(missing) == 0: 
            status_text = "QUALITY: OK" 
            color = (0, 255, 0) 
        else: 
            status_text = "QUALITY: NOT OK" 
            color = (0, 0, 255) 
 
        cv2.putText(frame, status_text, (20, 50), font, 1.2, color, 3, cv2.LINE_AA) 
 
        info = f"Matched: {len(matched)} | Missing: {len(missing)} | Extra: {len(extra)} | Expected: {expected}" 
        cv2.putText(frame, info, (20, 90), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA) 
 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 

        return frame 
 
    def _handle_quality(self, matched, missing, expected): 
        """Handle quality control""" 
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
        """Stop thread""" 
        logger.info("Stopping detection thread") 
        self.running = False 
        self.quit() 
        self.wait(3000) 
        if self.camera: 
            self.camera.release() 
 
 
class DetectionPage(QWidget): 
    """Detection page with USB and IP camera support""" 
    def __init__(self, model_manager, camera_config): 
        super().__init__() 
        self.model_manager = model_manager 
        self.camera_config = camera_config 
        self.detection_thread = None 
        self.yolo_model_path = None 
        self.init_ui() 
     
    def init_ui(self): 
        """Initialize detection page UI""" 
        layout = QVBoxLayout(self) 
        layout.setSpacing(10) 
        layout.setContentsMargins(10, 10, 10, 10) 
         
        # Header 
        header = QLabel("🎯 Stud Detection System") 
        header.setStyleSheet(""" 
            font-size: 20px; 
            font-weight: bold; 
            color: #4CAF50; 
            padding: 10px; 
            background-color: #f0fff0; 
            border-radius: 8px; 
            border: 2px solid #4CAF50; 
        """) 
        header.setAlignment(Qt.AlignCenter) 
        layout.addWidget(header) 
         
        # Main content 
        content_layout = QHBoxLayout() 
         
        # Left side - Video 
        left_layout = QVBoxLayout() 
        self.create_video_display(left_layout) 
        content_layout.addLayout(left_layout, 3) 
         
        # Right side - Controls 
        right_layout = QVBoxLayout() 
        self.create_status_panel(right_layout) 
        content_layout.addLayout(right_layout, 1) 
         
        layout.addLayout(content_layout) 
     
    def create_video_display(self, layout): 
        """Create video display area""" 
        video_title = QLabel("📹 Live Detection Feed") 
        video_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;") 
        layout.addWidget(video_title) 
         
        self.video_widget = DetectionVisualizationWidget() 
        layout.addWidget(self.video_widget) 
         
        # Model controls 
        model_layout = QHBoxLayout() 
         
        model_layout.addWidget(QLabel("BIW Model:")) 
        self.biw_model_combo = QComboBox() 
        self.biw_model_combo.setMinimumWidth(150) 
        self.refresh_biw_models() 
        model_layout.addWidget(self.biw_model_combo) 
         
        model_layout.addWidget(QLabel("YOLO Model:")) 
        self.yolo_model_label = QLabel("No model") 
        self.yolo_model_label.setStyleSheet(""" 
            padding: 6px 10px; 
            border: 1px solid #ccc; 
            border-radius: 4px; 
            background-color: #f9f9f9; 
            min-width: 150px; 
            color: #666; 
        """) 
        model_layout.addWidget(self.yolo_model_label) 
         
        load_yolo_btn = QPushButton("📁 Load") 
        load_yolo_btn.clicked.connect(self.load_yolo_model) 
        load_yolo_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                padding: 6px 12px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #1976D2; } 
        """) 
        model_layout.addWidget(load_yolo_btn) 
         
        model_layout.addStretch() 
        layout.addLayout(model_layout) 
         
        # Detection controls 
        det_controls = QHBoxLayout() 
        roi_btn = QPushButton("📍 Set ROI") 
        roi_btn.clicked.connect(self.set_detection_roi) 
        roi_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #FF9800; 
                color: white; 
                padding: 6px 12px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #f57c00; } 
        """) 
        det_controls.addWidget(roi_btn) 
         
        det_controls.addWidget(QLabel("Timer:")) 
        self.delay_spin = QSpinBox() 
        self.delay_spin.setRange(1, 30) 
        self.delay_spin.setValue(3) 
        self.delay_spin.setSuffix(" sec") 
        det_controls.addWidget(self.delay_spin) 
         
        det_controls.addWidget(QLabel("Min Studs:")) 
        self.min_studs_spin = QSpinBox() 
        self.min_studs_spin.setRange(1, 50) 
        self.min_studs_spin.setValue(5) 
        det_controls.addWidget(self.min_studs_spin) 
         
        det_controls.addStretch() 
        layout.addLayout(det_controls) 
         
        # Camera and control buttons 
        controls = QHBoxLayout() 
         
        controls.addWidget(QLabel("Camera:")) 
        self.camera_combo = QComboBox() 
        self.camera_combo.setMinimumWidth(200) 
        self.refresh_cameras() 
        controls.addWidget(self.camera_combo) 
         
        controls.addStretch() 
         
        self.start_btn = QPushButton("▶ Start Detection") 
        self.start_btn.clicked.connect(self.start_detection) 
        self.start_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                padding: 8px 16px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #45a049; } 
        """) 
        controls.addWidget(self.start_btn) 

        self.stop_btn = QPushButton("⏹ Stop") 
        self.stop_btn.clicked.connect(self.stop_detection) 
        self.stop_btn.setEnabled(False) 
        self.stop_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                padding: 8px 16px; 
                border: none; 
                border-radius: 5px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #da190b; } 
        """) 
        controls.addWidget(self.stop_btn) 
         
        layout.addLayout(controls) 
     
    def create_status_panel(self, layout): 
        """Create status panel""" 
        status_title = QLabel("📊 Detection Status") 
        status_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 8px;") 
        layout.addWidget(status_title) 
         
        # Connection status 
        conn_group = QGroupBox("🔗 Connection") 
        conn_group.setStyleSheet(self.get_group_style()) 
        conn_layout = QVBoxLayout(conn_group) 
        self.connection_label = QLabel("Status: Ready") 
        self.connection_label.setStyleSheet("font-size: 11px; padding: 5px;") 
        conn_layout.addWidget(self.connection_label) 
        layout.addWidget(conn_group) 
         
        # Cycle status 
        cycle_group = QGroupBox("🔄 Cycle Status") 
        cycle_group.setStyleSheet(self.get_group_style()) 
        cycle_layout = QVBoxLayout(cycle_group) 
         
        self.cycle_status = QLabel("Status: Waiting") 
        self.cycle_status.setStyleSheet("font-size: 12px; font-weight: bold; color: #333; padding: 5px;") 
        cycle_layout.addWidget(self.cycle_status) 
         
        self.timer_status = QLabel("Timer: Ready") 
        self.timer_status.setStyleSheet("font-size: 11px; color: #666; padding: 3px;") 
        cycle_layout.addWidget(self.timer_status) 
         
        layout.addWidget(cycle_group) 
         
        # Quality status 
        self.ok_status = StatusWidget("✅ Quality OK") 
        self.not_ok_status = StatusWidget("❌ NOT OK") 
        layout.addWidget(self.ok_status) 
        layout.addWidget(self.not_ok_status) 
         
        # Detection info 
        info_group = QGroupBox("🔍 Detection Info") 
        info_group.setStyleSheet(self.get_group_style()) 
        info_layout = QVBoxLayout(info_group) 
         
        self.matched_label = QLabel("Matched: 0") 
        self.missing_label = QLabel("Missing: 0") 
        self.extra_label = QLabel("Extra: 0") 
        self.expected_label = QLabel("Expected: 0") 
         
        for label in [self.matched_label, self.missing_label, self.extra_label, self.expected_label]: 
            label.setStyleSheet("padding: 4px; font-size: 11px;") 
            info_layout.addWidget(label) 
         
        layout.addWidget(info_group) 
         
        # Settings 
        settings_group = QGroupBox("⚙ Settings") 
        settings_group.setStyleSheet(self.get_group_style()) 
        settings_layout = QVBoxLayout(settings_group) 
         
        self.roi_label = QLabel("ROI: Not Set") 
        self.roi_label.setStyleSheet("padding: 4px; font-size: 10px; color: #666;") 
        settings_layout.addWidget(self.roi_label) 
         
        reset_btn = QPushButton(" 🔄 Reset Counts") 
        reset_btn.clicked.connect(self.reset_counts) 
        reset_btn.setStyleSheet(""" 
            QPushButton { 
                background-color: #FF9800; 
                color: white; 
                padding: 6px; 
                border: none; 
                border-radius: 4px; 
                font-weight: bold; 
            } 
            QPushButton:hover { background-color: #f57c00; } 
        """) 
        settings_layout.addWidget(reset_btn) 
         
        layout.addWidget(settings_group) 
        layout.addStretch() 
     
    def get_group_style(self): 
        """Get consistent group style""" 
        return """ 
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
        """ 
     
    def refresh_cameras(self): 
        """Refresh camera list""" 
        self.camera_combo.clear() 
        cameras = self.camera_config.get_all_cameras() 
         
        for cam in cameras: 
            if cam["type"] == "usb": 
                self.camera_combo.addItem(f" 🔌 {cam['name']}", cam) 
            else: 
                self.camera_combo.addItem(f" 🌐 {cam['name']}", cam) 
         
        if self.camera_combo.count() == 0: 
            self.camera_combo.addItem("No cameras") 
     
    def refresh_biw_models(self): 
        """Refresh BIW models""" 
        self.biw_model_combo.clear() 
         
        for name in self.model_manager.get_model_names(): 
            model = self.model_manager.get_model(name) 
            self.biw_model_combo.addItem(f"{name} ({model['stud_count']} studs)", name) 
         
        if self.biw_model_combo.count() == 0: 
            self.biw_model_combo.addItem("No models") 
     
    def load_yolo_model(self): 
        """Load YOLO model""" 
        file_path, _ = QFileDialog.getOpenFileName( 
            self, "Load YOLO Model", "", 
            "YOLO Models (*.pt *.onnx *.engine);;All Files (*)" 
        ) 
         
        if file_path and os.path.exists(file_path): 
            self.yolo_model_path = file_path 
            model_name = os.path.basename(file_path) 
            self.yolo_model_label.setText(model_name) 
            self.yolo_model_label.setStyleSheet(""" 
                padding: 6px 10px; 
                border: 1px solid #4CAF50; 
                border-radius: 4px; 
                background-color: #f0fff0; 
                min-width: 150px; 
                color: #2e7d32; 
                font-weight: bold; 
            """) 
            self.yolo_model_label.setToolTip(file_path) 
            QMessageBox.information(self, "Success", f"Model loaded:\n{model_name}") 
            logger.info(f"Loaded YOLO model: {file_path}") 
     
    def set_detection_roi(self): 
        """Set detection ROI""" 
        dialog = ROISelectionDialog(self) 
        if dialog.exec_() == QDialog.Accepted: 
            roi = dialog.get_roi() 
            if self.detection_thread: 
                self.detection_thread.set_detection_roi(roi) 
            self.roi_label.setText(f"ROI: {roi[0]},{roi[1]} {roi[2]}x{roi[3]}") 
            logger.info(f"ROI set: {roi}") 
     
    def start_detection(self): 
        """Start detection""" 
        if self.detection_thread and self.detection_thread.isRunning(): 
            return 
         
        camera_data = self.camera_combo.currentData() 
        if not camera_data: 
            QMessageBox.warning(self, "Error", "No camera selected!") 
            return 
         
        biw_model = self.biw_model_combo.currentData() 
        if not biw_model: 
            QMessageBox.warning(self, "Error", "No BIW model! Create one in Teaching tab.") 
            return 
         
        if not self.yolo_model_path: 
            QMessageBox.warning(self, "Error", "No YOLO model loaded!") 
            return 
         
        # Get camera source 
        if camera_data["type"] == "usb": 
            camera_source = camera_data["index"] 
            is_ip = False 
        else: 
            camera_source = camera_data["rtsp_url"] 
            is_ip = True 
         
        # Create detection thread 
        self.detection_thread = DetectionThread( 
            camera_source, is_ip, self.model_manager, biw_model, self.yolo_model_path 
        ) 
         
        self.detection_thread.set_detection_delay(self.delay_spin.value()) 
        self.detection_thread.set_min_studs_threshold(self.min_studs_spin.value()) 
         
        # Connect signals 
        self.detection_thread.frame_ready.connect(self.video_widget.update_frame) 
        self.detection_thread.detection_ready.connect(self.on_detection_ready) 
        self.detection_thread.status_update.connect(self.on_status_update) 
        self.detection_thread.stud_status_update.connect(self.on_stud_status_update) 
        self.detection_thread.connection_status.connect(self.update_connection_status) 
         
        self.detection_thread.start() 
         
        # Update UI 
        self.start_btn.setText("🔄 Active") 
        self.start_btn.setEnabled(False) 
        self.stop_btn.setEnabled(True) 
         
        model = self.model_manager.get_model(biw_model) 
        self.expected_label.setText(f"Expected: {model['stud_count']}") 
         
        logger.info(f"Started detection: {camera_data['name']}, BIW: {biw_model}") 
     
    def stop_detection(self): 
        """Stop detection""" 
        if self.detection_thread: 
            self.detection_thread.stop() 
            self.detection_thread = None 
         
        self.start_btn.setText("▶ Start Detection") 
        self.start_btn.setEnabled(True) 
        self.stop_btn.setEnabled(False) 
        self.video_widget.setText("🎯 Detection stopped") 
         
        self.cycle_status.setText("Status: Stopped") 
        self.timer_status.setText("Timer: Inactive") 
         
        logger.info("Stopped detection") 
     
    def update_connection_status(self, status, is_error): 
        """Update connection status""" 
        color = "#f44336" if is_error else "#4CAF50" 
        self.connection_label.setStyleSheet(f"font-size: 11px; padding: 5px; color: {color}; font-weight: bold;") 
        self.connection_label.setText(f"Status: {status}") 
     
    @pyqtSlot(str, int) 
    def on_stud_status_update(self, status, timer): 
        """Handle stud status update""" 
        self.cycle_status.setText(f"Status: {status}") 
        if timer > 0: 
            self.timer_status.setText(f"Timer: {timer}s") 
        else: 
            self.timer_status.setText("Timer: Ready") 
     
    @pyqtSlot(object, list, list, list) 
    def on_detection_ready(self, frame, matched, missing, extra): 
        """Handle detection ready""" 
        self.video_widget.update_frame_with_detection(frame, matched, missing, extra) 
        self.matched_label.setText(f"Matched: {len(matched)}") 
        self.missing_label.setText(f"Missing: {len(missing)}") 
        self.extra_label.setText(f"Extra: {len(extra)}") 
     
    @pyqtSlot(str, int, int) 
    def on_status_update(self, status, ok, not_ok): 
        """Handle status update""" 
        self.ok_status.update_status("OK", ok) 
        self.not_ok_status.update_status("NOT OK", not_ok) 
     
    def reset_counts(self): 
        """Reset quality counts""" 
        reply = QMessageBox.question(self, "Reset", "Reset all counts?",  
                                     QMessageBox.Yes | QMessageBox.No) 
        if reply == QMessageBox.Yes: 
            try: 
                with open("quality_count.csv", 'w', newline='') as f: 
                    writer = csv.writer(f) 
                    writer.writerow(["OK", "NOT_OK"]) 
                    writer.writerow([0, 0]) 
                 
                self.ok_status.update_status("OK", 0) 
                self.not_ok_status.update_status("NOT OK", 0) 
                 
                if self.detection_thread: 
                    self.detection_thread.ok_count = 0 
                    self.detection_thread.not_ok_count = 0 
                 
                QMessageBox.information(self, "Success", "Counts reset!") 
                logger.info("Reset quality counts") 
            except Exception as e: 
                QMessageBox.critical(self, "Error", f"Reset failed: {e}") 
 
 
class MainWindow(QMainWindow): 
    """Main application window""" 
    def __init__(self): 
        super().__init__() 
        self.model_manager = BIWModelManager() 
        self.camera_config = CameraConfig() 
        self.init_ui() 
     
    def init_ui(self): 
        """Initialize UI""" 
        self.setWindowTitle("Industrial Stud Detection System v5.0") 
        self.setGeometry(100, 50, 1600, 900) 
        self.setMinimumSize(1400, 800) 
         
        # Set style 
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
        """) 
         
        central_widget = QWidget() 
        self.setCentralWidget(central_widget) 
        layout = QVBoxLayout(central_widget) 
        layout.setContentsMargins(10, 10, 10, 10) 
         
        # Tab widget 
        self.tab_widget = QTabWidget() 
         
        # Create pages 
        self.teaching_page = TeachingPage(self.model_manager, self.camera_config) 
        self.detection_page = DetectionPage(self.model_manager, self.camera_config) 
         
        self.tab_widget.addTab(self.teaching_page, "📚 Teaching") 
        self.tab_widget.addTab(self.detection_page, "🎯 Detection") 
         
        self.tab_widget.currentChanged.connect(self.on_tab_changed) 
         
        layout.addWidget(self.tab_widget) 
         
        # Status bar 
        self.statusBar().showMessage("Ready - Industrial Stud Detection System v5.0") 
         
        # Menu bar 
        self.create_menu_bar() 
         
        logger.info("Application started") 
     
    def create_menu_bar(self): 
        """Create menu bar""" 
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
         
        add_ip_action = QAction('Add IP Camera...', self) 
        add_ip_action.triggered.connect(self.add_ip_camera) 
        camera_menu.addAction(add_ip_action) 
         
        manage_ip_action = QAction('Manage IP Cameras...', self) 
        manage_ip_action.triggered.connect(self.manage_ip_cameras) 
        camera_menu.addAction(manage_ip_action) 
         
        refresh_action = QAction('Refresh Cameras', self) 
        refresh_action.setShortcut('F5') 
        refresh_action.triggered.connect(self.refresh_all_cameras) 
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
        """Handle tab change""" 
        if index == 1: 
            self.detection_page.refresh_biw_models() 
            self.detection_page.refresh_cameras() 
     
    def add_ip_camera(self): 
        """Add IP camera""" 
        dialog = IPCameraDialog(self) 
        if dialog.exec_() == QDialog.Accepted: 
            config = dialog.get_camera_config() 
            if self.camera_config.add_ip_camera( 
                config['name'], config['rtsp_url'],  
                config['username'], config['password'] 
            ): 
                QMessageBox.information(self, "Success", f"Camera '{config['name']}' added!") 
                self.refresh_all_cameras() 
                logger.info(f"Added IP camera: {config['name']}") 
     
    def manage_ip_cameras(self): 
        """Manage IP cameras""" 
        cameras = self.camera_config.configs["ip_cameras"] 
        if not cameras: 
            QMessageBox.information(self, "No Cameras", "No IP cameras configured.") 
            return 
         
        items = [cam['name'] for cam in cameras] 
        item, ok = QInputDialog.getItem(self, "Manage IP Cameras",  
                                        "Select camera to remove:", items, 0, False) 
         
        if ok and item: 
            reply = QMessageBox.question(self, "Confirm", f"Remove '{item}'?", 
                                        QMessageBox.Yes | QMessageBox.No) 
            if reply == QMessageBox.Yes: 
                if self.camera_config.remove_ip_camera(item): 
                    QMessageBox.information(self, "Success", f"Camera '{item}' removed!") 
                    self.refresh_all_cameras() 
                    logger.info(f"Removed IP camera: {item}") 
     
    def refresh_all_cameras(self): 
        """Refresh all camera lists""" 
        self.teaching_page.refresh_cameras() 
        self.detection_page.refresh_cameras() 
        logger.info("Refreshed camera lists") 
     
    def export_models(self): 
        """Export models""" 
        file_path, _ = QFileDialog.getSaveFileName( 
            self, "Export Models", "biw_models_backup.json", "JSON Files (*.json)" 
        ) 
        if file_path: 
            try: 
                import shutil 
                shutil.copy(self.model_manager.models_file, file_path) 
                QMessageBox.information(self, "Success", f"Exported to:\n{file_path}") 
                logger.info(f"Exported models to: {file_path}") 
            except Exception as e: 
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}") 
     
    def import_models(self): 
        """Import models""" 
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
                    self.teaching_page.refresh_models_list() 
                    self.detection_page.refresh_biw_models() 
                    QMessageBox.information(self, "Success", "Models imported!") 
                    logger.info(f"Imported models from: {file_path}") 
                except Exception as e: 
                    QMessageBox.critical(self, "Error", f"Import failed:\n{e}") 
     
    def view_logs(self): 
        """View application logs""" 
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
                QMessageBox.information(self, "Logs", f"Log file: {os.path.abspath('stud_detection.log')}") 
        else: 
            QMessageBox.information(self, "No Logs", "No log file found yet.") 
     
    def show_about(self): 
        """Show about dialog""" 
        QMessageBox.about( 
            self, "About", 
            "Industrial Stud Detection System v5.0\n\n" 
            "Professional BIW quality inspection system\n\n" 
            "Features:\n" 
            "✓ USB & IP camera support (Hikvision compatible)\n" 
            "✓ Camera-based model teaching\n" 
            "✓ Stud-triggered detection cycles\n" 
            "✓ Custom YOLO model integration\n" 
            "✓ Real-time quality control\n" 
            "✓ HID relay integration\n" 
            "✓ Automatic reconnection\n" 
            "✓ Comprehensive logging\n" 
            "✓ Export/Import capabilities\n\n" 
            "© 2024 Industrial Detection Systems" 
        ) 
     
    def closeEvent(self, event): 
        """Handle application closing""" 
        logger.info("Application closing...") 
        # Stop detection thread
        if hasattr(self.detection_page, 'detection_thread') and self.detection_page.detection_thread:
            self.detection_page.detection_thread.stop()

        # Stop teaching camera
        if hasattr(self.teaching_page, 'camera_thread') and self.teaching_page.camera_thread:
            self.teaching_page.camera_thread.stop()
         
        logger.info("Application closed") 
        event.accept() 
 
 
def main(): 
    """Main application entry point""" 
    # Set up exception handling 
    def exception_hook(exctype, value, tb): 
        logger.error("Uncaught exception", exc_info=(exctype, value, tb)) 
        sys.__excepthook__(exctype, value, tb) 
     
    sys.excepthook = exception_hook 
     
    # Create application 
    app = QApplication(sys.argv) 
     
    # Set application properties 
    app.setApplicationName("Industrial Stud Detection System") 
    app.setApplicationVersion("5.0") 
    app.setOrganizationName("Industrial Detection Systems") 
     
    # Set application icon (if available) 
    # app.setWindowIcon(QIcon('icon.png')) 
     
    # Create and show main window 
    window = MainWindow() 
    window.show() 
     
    logger.info("=" * 50) 
    logger.info("Industrial Stud Detection System v5.0 Started") 
    logger.info("=" * 50) 
     
    # Run application 
    sys.exit(app.exec_()) 
 
 
if __name__ == '__main__': 
    main() 
 