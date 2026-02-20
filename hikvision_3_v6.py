import sys
from ultralytics import YOLO
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import os
import pyhid_usb_relay
import threading
import queue
import time
import logging
from datetime import datetime
from collections import deque
import traceback

# Configuration
CONFIG = {
    "detection": {
        "confidence_thresholds": {
            "oil_can": 0.45,
            "bunk_hole": 0.40
        },
        "watchdog_timeout_base": 10,
        "max_reconnect_attempts": 10,
        "reconnect_backoff_max": 60
    },
    "camera": {
        "rtsp_timeout_ms": 5000,
        "buffer_size": 1,
        "default_fps": 30
    },
    "required_classes": {
        0: "bunk_hole",
        1: "teflon",
        2: "lance",
        3: "oil_can"
    }
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'detection_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FrameBuffer:
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

class WatchdogTimer(QThread):
    timeout_signal = pyqtSignal(str)
    
    def __init__(self, name, timeout_seconds=10):
        super().__init__()
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = time.time()
        self.running = True
        self.lock = threading.Lock()
        
    def heartbeat(self):
        with self.lock:
            self.last_heartbeat = time.time()
    
    def run(self):
        logger.info(f"Watchdog started for {self.name} (timeout: {self.timeout_seconds}s)")
        while self.running:
            time.sleep(1)
            with self.lock:
                elapsed = time.time() - self.last_heartbeat
            
            if elapsed > self.timeout_seconds:
                logger.error(f"Watchdog timeout for {self.name}: {elapsed:.1f}s")
                self.timeout_signal.emit(self.name)
                with self.lock:
                    self.last_heartbeat = time.time()
    
    def stop(self):
        self.running = False
        logger.info(f"Watchdog stopped for {self.name}")

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, camera_source):
        super().__init__()
        self.camera_source = camera_source
        self.running = False
        self.camera = None
        self.frame_buffer = FrameBuffer(maxsize=5)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = CONFIG["detection"]["max_reconnect_attempts"]
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.is_ip_camera = isinstance(camera_source, str)
        
    def run(self):
        self.running = True
        camera_type = "IP Camera (RTSP)" if self.is_ip_camera else f"USB Camera {self.camera_source}"
        logger.info(f"Camera thread started for {camera_type}")
        
        while self.running:
            try:
                if self.camera is None or not self.camera.isOpened():
                    self.connect_camera()
                
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    
                    if ret and frame is not None:
                        self.frame_buffer.put(frame.copy(), block=False)
                        self.frame_ready.emit(frame)
                        self.reconnect_attempts = 0
                        self.frame_count += 1
                        
                        current_time = time.time()
                        if current_time - self.last_frame_time >= 5.0:
                            fps = self.frame_count / (current_time - self.last_frame_time)
                            logger.debug(f"Camera FPS: {fps:.1f}")
                            self.frame_count = 0
                            self.last_frame_time = current_time
                    else:
                        logger.warning("Failed to read frame, attempting reconnect...")
                        self.reconnect_camera()
                else:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Camera thread error: {e}\n{traceback.format_exc()}")
                self.error_signal.emit(str(e))
                self.reconnect_camera()
                time.sleep(1)
        
        self.cleanup()
        logger.info("Camera thread stopped")
    
    def connect_camera(self):
        for attempt in range(self.max_reconnect_attempts):
            try:
                camera_desc = f"IP camera (RTSP)" if self.is_ip_camera else f"USB camera {self.camera_source}"
                logger.info(f"Connecting to {camera_desc}, attempt {attempt + 1}")
                
                if self.is_ip_camera:
                    self.camera = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG["camera"]["buffer_size"])
                    self.camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CONFIG["camera"]["rtsp_timeout_ms"])
                    self.camera.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, CONFIG["camera"]["rtsp_timeout_ms"])
                else:
                    self.camera = cv2.VideoCapture(self.camera_source)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG["camera"]["buffer_size"])
                    self.camera.set(cv2.CAP_PROP_FPS, CONFIG["camera"]["default_fps"])
                
                if self.camera and self.camera.isOpened():
                    ret, test_frame = self.camera.read()
                    if ret and test_frame is not None:
                        logger.info(f"{camera_desc} connected successfully - Frame size: {test_frame.shape}")
                        self.status_signal.emit(f"Camera Connected ({camera_desc})")
                        self.reconnect_attempts = 0
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
            
            time.sleep(2)
        
        self.error_signal.emit(f"Failed to connect camera after {self.max_reconnect_attempts} attempts")
        return False
    
    def reconnect_camera(self):
        self.reconnect_attempts += 1
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        
        # Exponential backoff with max cap
        delay = min(2 ** self.reconnect_attempts, CONFIG["detection"]["reconnect_backoff_max"])
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.status_signal.emit(f"Reconnecting in {delay}s... (attempt {self.reconnect_attempts})")
            time.sleep(delay)
        else:
            # Reset counter for perpetual retries
            logger.warning("Max attempts reached, resetting reconnection counter")
            self.reconnect_attempts = 0
            time.sleep(60)
    
    def cleanup(self):
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        logger.info("Camera resources cleaned up")
    
    def stop(self):
        self.running = False
        self.wait(3000)

class DetectionThread(QThread):
    detection_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, model_path, boundaries):
        super().__init__()
        self.model_path = model_path
        self.boundaries = boundaries
        self.oil_can_boundaries = [b for b in boundaries if b.get('type') == 'oil_can']
        self.bunk_hole_boundaries = [b for b in boundaries if b.get('type') == 'bunk_hole']
        self.running = False
        self.model = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.fps_deque = deque(maxlen=30)
        self.last_detection_time = time.time()
        self.detection_count = 0
        self.class_names = {}
        
    def validate_model_classes(self):
        """Validate YOLO model has expected classes"""
        if not self.class_names:
            logger.warning("Model has no class names defined")
            return
        
        logger.info(f"Model classes: {self.class_names}")
        
        # Check if required classes exist anywhere in the model
        has_oil_can = any('oil' in str(name).lower() or 'can' in str(name).lower() 
                        for name in self.class_names.values())
        has_bunk_hole = any('bunk' in str(name).lower() or 'hole' in str(name).lower() 
                            for name in self.class_names.values())
        
        if not has_oil_can or not has_bunk_hole:
            raise ValueError(
                f"⚠️ Model missing required classes!\n"
                f"Model classes: {self.class_names}\n"
                f"Need: oil_can and bunk_hole"
            )
        
        logger.info("✅ Model has required classes (flexible validation)")
        
    def run(self):
        self.running = True
        logger.info("Detection thread started")
        
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            try:
                self.class_names = getattr(self.model, 'names', None) or {}
                self.validate_model_classes()
            except Exception as e:
                logger.critical(f"Model validation failed: {e}")
                self.error_signal.emit(str(e))
                return
            
            logger.info("YOLO model loaded successfully")
            
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    if frame is not None:
                        start_time = time.time()
                        result = self.process_frame(frame)
                        inference_time = time.time() - start_time
                        
                        self.fps_deque.append(1.0 / inference_time if inference_time > 0 else 0)
                        avg_fps = sum(self.fps_deque) / len(self.fps_deque)
                        
                        self.detection_ready.emit(result)
                        self.fps_signal.emit(avg_fps)
                        self.detection_count += 1
                        
                        if self.detection_count % 100 == 0:
                            logger.info(f"Detection performance - FPS: {avg_fps:.1f}, Total: {self.detection_count}")
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Detection error: {e}\n{traceback.format_exc()}")
                    self.error_signal.emit(str(e))
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.critical(f"Fatal detection thread error: {e}\n{traceback.format_exc()}")
            self.error_signal.emit(f"Fatal error: {str(e)}")
        
        logger.info("Detection thread stopped")
    
    def process_frame(self, frame):
        try:
            # Use configurable confidence thresholds
            results = self.model(frame, verbose=False, conf=0.3)
            
            oil_cans = []
            bunk_holes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detected_obj = {
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'confidence': float(confidence),
                            'center': (float((x1 + x2) / 2), float((y1 + y2) / 2)),
                            'class_id': class_id
                        }

                        # Map by model class name
                        class_name = str(self.class_names.get(class_id, '')).lower()

                        # Apply category-specific confidence thresholds
                        if 'oil' in class_name or 'can' in class_name:
                            if confidence >= CONFIG["detection"]["confidence_thresholds"]["oil_can"]:
                                oil_cans.append(detected_obj)
                        elif 'bunk' in class_name or 'hole' in class_name:
                            if confidence >= CONFIG["detection"]["confidence_thresholds"]["bunk_hole"]:
                                bunk_holes.append(detected_obj)
                        else:
                            # Fallback to index mapping - UPDATED FOR YOUR MODEL
                            if class_id == 3 and confidence >= CONFIG["detection"]["confidence_thresholds"]["oil_can"]:
                                oil_cans.append(detected_obj)
                            elif class_id == 0 and confidence >= CONFIG["detection"]["confidence_thresholds"]["bunk_hole"]:
                                bunk_holes.append(detected_obj)
            
            # NEW PAIRING LOGIC: Check oil can + bunk hole pairs
            pair_statuses = []
            mismatched_pairs = []
            
            for i in range(3):
                if i >= len(self.oil_can_boundaries) or i >= len(self.bunk_hole_boundaries):
                    break
                    
                # Check oil can (partial overlap allowed)
                oc_objects = self.check_objects_in_boundary(
                    oil_cans, self.oil_can_boundaries[i], strict=False
                )
                oc_present = len(oc_objects) > 0
                
                # Check bunk hole (strict boundary check)
                bh_objects = self.check_objects_in_boundary(
                    bunk_holes, self.bunk_hole_boundaries[i], strict=True
                )
                bh_present = len(bh_objects) > 0
                
                # Pairing logic
                if not oc_present and not bh_present:
                    # Both missing - ignore this pair (no alarm)
                    pair_status = 'ignored'
                    pair_ok = True
                elif oc_present and bh_present:
                    # Both present - all good
                    pair_status = 'ok'
                    pair_ok = True
                else:
                    # Mismatch - one present, one missing
                    pair_status = 'mismatch'
                    pair_ok = False
                    mismatched_pairs.append({
                        'pair_index': i + 1,
                        'oc_present': oc_present,
                        'bh_present': bh_present
                    })
                
                pair_statuses.append({
                    'index': i,
                    'oc_present': oc_present,
                    'bh_present': bh_present,
                    'status': pair_status,
                    'ok': pair_ok
                })
            
            # System is OK if all checked pairs have no mismatches
            all_ok = len(mismatched_pairs) == 0
            
            return {
                'frame': frame,
                'oil_cans': oil_cans,
                'bunk_holes': bunk_holes,
                'pair_statuses': pair_statuses,
                'mismatched_pairs': mismatched_pairs,
                'all_ok': all_ok,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            raise
    
    def check_objects_in_boundary(self, objects, boundary, strict=False):
        """Optimized boundary check with vectorization"""
        objects_in_boundary = []
        
        try:
            if 'polygon' in boundary and boundary.get('polygon'):
                contour = np.array(boundary['polygon'], dtype=np.float32)
                
                for obj in objects:
                    center_x, center_y = obj['center']
                    
                    try:
                        if strict:
                            # Strict: require center strictly inside (val > 0)
                            val = cv2.pointPolygonTest(contour, (float(center_x), float(center_y)), False)
                            if val > 0:
                                objects_in_boundary.append(obj)
                        else:
                            # Non-strict: allow partial overlap
                            val_center = cv2.pointPolygonTest(contour, (float(center_x), float(center_y)), False)
                            if val_center >= 0:
                                objects_in_boundary.append(obj)
                                continue

                            # Check bbox corners for partial overlap
                            x1, y1, x2, y2 = obj['bbox']
                            corners = np.array([
                                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
                            ], dtype=np.float32)
                            
                            for corner in corners:
                                if cv2.pointPolygonTest(contour, tuple(corner), False) >= 0:
                                    objects_in_boundary.append(obj)
                                    break
                    except Exception as e:
                        logger.debug(f"Polygon test error: {e}")
                        pass
            else:
                # Rectangle boundary format
                for obj in objects:
                    center_x, center_y = obj['center']
                    x1 = boundary.get('x1', 0)
                    x2 = boundary.get('x2', 0)
                    y1 = boundary.get('y1', 0)
                    y2 = boundary.get('y2', 0)
                    
                    if strict:
                        if x1 < center_x < x2 and y1 < center_y < y2:
                            objects_in_boundary.append(obj)
                    else:
                        if (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                            objects_in_boundary.append(obj)
                            continue

                        bx1, by1, bx2, by2 = obj['bbox']
                        if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                            objects_in_boundary.append(obj)
        except Exception as e:
            logger.error(f"Boundary check error: {e}")

        return objects_in_boundary
    
    def add_frame(self, frame):
        try:
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            pass
    
    def stop(self):
        self.running = False
        self.wait(3000)

class RelayControlThread(QThread):
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.command_queue = queue.Queue()
        self.running = False
        self.relay = None
        self.last_state = None
        
    def run(self):
        self.running = True
        logger.info("Relay control thread started")
        
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                
                if command is not None:
                    self.execute_command(command)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Relay control error: {e}")
                self.error_signal.emit(str(e))
        
        logger.info("Relay control thread stopped")
    
    def execute_command(self, all_ok):
        try:
            if all_ok == self.last_state:
                return
            
            if self.relay is None:
                self.relay = pyhid_usb_relay.find()
            
            if all_ok:
                self.relay.set_state(1, True)
                self.relay.set_state(2, False)
                logger.info("✅ Relay: OK state (R1=ON, R2=OFF)")
                self.status_signal.emit("Relay: All Pairs OK")
            else:
                self.relay.set_state(1, False)
                self.relay.set_state(2, True)
                logger.warning("⚠️ Relay: MISMATCH state (R1=OFF, R2=ON)")
                self.status_signal.emit("Relay: Pair Mismatch Detected")
            
            self.last_state = all_ok
            
        except Exception as e:
            logger.error(f"Relay execution error: {e}")
            self.relay = None
            self.error_signal.emit(f"Relay error: {str(e)}")
    
    def set_state(self, all_ok):
        try:
            self.command_queue.put(all_ok, block=False)
        except queue.Full:
            pass
    
    def stop(self):
        self.running = False
        self.wait(3000)

class DrawingWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 2px solid #333; background-color: white;")
        self.setAlignment(Qt.AlignCenter)
        self.image = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.polygons = []
        self.boundaries = []
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        
        self.oil_can_color = QColor(0, 0, 255)
        self.bunk_hole_color = QColor(255, 165, 0)
        self.preview_color = QColor(128, 128, 128, 100)
        self.line_width = 3

    def set_image(self, cv_image):
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

    def resizeEvent(self, event):
        if hasattr(self, 'cv_image') and self.cv_image is not None:
            self.set_image(self.cv_image)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image and len(self.boundaries) < 6:
            img_pos = self.widget_to_image_coords(event.pos())
            if not img_pos:
                return

            if not self.drawing:
                self.drawing = True
                self.start_point = img_pos
                self.end_point = img_pos
                self.current_points = [img_pos]
            else:
                first = self.current_points[0]
                dist_sq = (first.x() - img_pos.x()) ** 2 + (first.y() - img_pos.y()) ** 2
                if dist_sq <= 100:
                    self.finish_polygon()
                    self.drawing = False
                else:
                    self.current_points.append(img_pos)
                    self.end_point = img_pos
            self.update_display()

    def mouseMoveEvent(self, event):
        if self.drawing and self.image:
            img_pos = self.widget_to_image_coords(event.pos())
            if img_pos:
                self.end_point = img_pos
                self.update_display()

    def widget_to_image_coords(self, widget_point):
        if not self.image:
            return None
        
        img_rect = QRect(self.offset_x, self.offset_y, self.image.width(), self.image.height())
        if not img_rect.contains(widget_point):
            return None
        
        x = widget_point.x() - self.offset_x
        y = widget_point.y() - self.offset_y
        return QPoint(x, y)

    def finish_polygon(self):
        if not hasattr(self, 'current_points') or len(self.current_points) < 3:
            logger.info("Polygon too small, ignored")
            return

        oil_can_count = len([b for b in self.boundaries if b.get('type') == 'oil_can'])
        bunk_hole_count = len([b for b in self.boundaries if b.get('type') == 'bunk_hole'])

        if oil_can_count < 3:
            boundary_type = 'oil_can'
        else:
            boundary_type = 'bunk_hole'

        pts_copy = list(self.current_points)
        self.polygons.append({'points': pts_copy, 'type': boundary_type})

        scaled_polygon = []
        for p in pts_copy:
            ox = p.x() * self.scale_x
            oy = p.y() * self.scale_y
            scaled_polygon.append([float(ox), float(oy)])

        boundary = {
            'type': boundary_type,
            'polygon': scaled_polygon
        }
        self.boundaries.append(boundary)
        logger.info(f"Polygon boundary added: {boundary_type} - Total: {len(self.boundaries)}")

        self.current_points = []
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.update_display()

    def update_display(self):
        if not self.image:
            return
        
        display_pixmap = self.image.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        oil_can_idx = 0
        bunk_hole_idx = 0

        for poly_item in self.polygons:
            pts = poly_item['points']
            poly_type = poly_item['type']

            if poly_type == 'oil_can':
                color = self.oil_can_color
                oil_can_idx += 1
                label = f"OC{oil_can_idx}"
            else:
                color = self.bunk_hole_color
                bunk_hole_idx += 1
                label = f"BH{bunk_hole_idx}"

            pen = QPen(color, self.line_width)
            painter.setPen(pen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 50))

            qpoly = QPolygon(pts)
            painter.drawPolygon(qpoly)

            first = pts[0]
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            label_pos = QPoint(first.x() + 5, first.y() - 5)
            painter.setPen(QPen(Qt.white, 2))
            painter.drawText(label_pos, label)
        
        if self.drawing and hasattr(self, 'current_points') and len(self.current_points) > 0:
            pen = QPen(self.preview_color, self.line_width)
            painter.setPen(pen)
            pts = list(self.current_points)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

            if self.end_point:
                painter.drawLine(pts[-1], self.end_point)

            first = pts[0]
            painter.setBrush(QBrush(self.preview_color))
            painter.drawEllipse(first, 5, 5)
        
        painter.end()
        
        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(Qt.white)
        final_painter = QPainter(final_pixmap)
        final_painter.drawPixmap(self.offset_x, self.offset_y, display_pixmap)
        final_painter.end()
        
        self.setPixmap(final_pixmap)

    def clear_boundaries(self):
        if hasattr(self, 'polygons'):
            self.polygons.clear()
        self.boundaries.clear()
        if hasattr(self, 'current_points'):
            self.current_points = []
        self.start_point = None
        self.end_point = None
        self.drawing = False
        if self.image:
            self.update_display()
        logger.info("All boundaries cleared")

    def undo_last_polygon(self):
        removed = False
        if hasattr(self, 'polygons') and self.polygons:
            self.polygons.pop()
            removed = True

        if self.boundaries:
            self.boundaries.pop()
            removed = True

        if removed:
            self.update_display()
            logger.info("Undo: removed last polygon/boundary")
        else:
            logger.info("Undo: nothing to remove")

        return removed

    def get_instruction_text(self):
        oil_can_count = len([b for b in self.boundaries if b.get('type') == 'oil_can'])
        bunk_hole_count = len([b for b in self.boundaries if b.get('type') == 'bunk_hole'])
        
        if oil_can_count < 3:
            return f"Draw Oil Can boundary {oil_can_count + 1} of 3 (Blue)"
        elif bunk_hole_count < 3:
            return f"Draw Bunk Hole boundary {bunk_hole_count + 1} of 3 (Orange)"
        else:
            return "All 6 boundaries completed! Click 'Save Boundaries'"

class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_thread = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Training - Define Oil Can & Bunk Hole Pairs")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        instructions = QLabel(
            "Instructions:\n"
            "1. Draw 3 Oil Can boundaries (Blue) - OC1, OC2, OC3\n"
            "2. Draw 3 Bunk Hole boundaries (Orange) - BH1, BH2, BH3\n"
            "3. Each pair (OC1↔BH1, OC2↔BH2, OC3↔BH3) will be monitored\n"
            "4. System OK when: Both present OR both absent in each pair\n"
            "5. System ALARM when: One present, one absent in any pair\n"
            "Total: 6 boundaries required (3 pairs)"
        )
        instructions.setStyleSheet(
            "background-color: #E1F5FE; padding: 10px; border-radius: 5px; "
            "font-size: 11px; color: #0277BD; border: 1px solid #0288D1;"
        )
        layout.addWidget(instructions)
        
        camera_controls = QHBoxLayout()
        
        camera_type_layout = QVBoxLayout()
        self.camera_type_group = QButtonGroup()
        self.usb_radio = QRadioButton("USB Camera")
        self.ip_radio = QRadioButton("IP Camera (RTSP)")
        self.usb_radio.setChecked(True)
        self.camera_type_group.addButton(self.usb_radio)
        self.camera_type_group.addButton(self.ip_radio)
        camera_type_layout.addWidget(self.usb_radio)
        camera_type_layout.addWidget(self.ip_radio)
        camera_controls.addLayout(camera_type_layout)
        
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        camera_controls.addWidget(QLabel("USB:"))
        camera_controls.addWidget(self.camera_combo)
        
        camera_controls.addWidget(QLabel("RTSP URL:"))
        self.ip_url_input = QLineEdit()
        self.ip_url_input.setPlaceholderText("rtsp://admin:password@192.168.1.64:554/stream")
        self.ip_url_input.setText("rtsp://admin:Pass_123@192.168.1.64:554/stream")
        self.ip_url_input.setMinimumWidth(350)
        camera_controls.addWidget(self.ip_url_input)
        
        self.usb_radio.toggled.connect(self.on_camera_type_changed)
        self.ip_radio.toggled.connect(self.on_camera_type_changed)
        
        layout.addLayout(camera_controls)
        button_controls = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        button_controls.addWidget(self.start_camera_btn)
        
        self.capture_btn = QPushButton("📷 Capture Photo")
        self.capture_btn.clicked.connect(self.capture_frame)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        button_controls.addWidget(self.capture_btn)
        
        layout.addLayout(button_controls)
        
        self.image_area = QStackedWidget()
        
        self.camera_label = QLabel("📹 Start camera to see preview")
        self.camera_label.setMinimumSize(900, 600)
        self.camera_label.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5; font-size: 16px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.image_area.addWidget(self.camera_label)
        
        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setMinimumSize(900, 600)
        self.image_area.addWidget(self.drawing_widget)
        
        layout.addWidget(self.image_area)
        
        self.instruction_label = QLabel("📷 Click 'Capture Photo' to start")
        self.instruction_label.setStyleSheet(
            "background-color: #E3F2FD; padding: 10px; border-radius: 5px; "
            "font-size: 12px; color: #1976D2;"
        )
        layout.addWidget(self.instruction_label)
        
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_boundaries)
        self.clear_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
        button_layout.addWidget(self.clear_btn)
        
        self.undo_btn = QPushButton("↶ Undo Last")
        self.undo_btn.clicked.connect(self.undo_last_boundary)
        self.undo_btn.setStyleSheet("background-color: #FFB300; color: white; padding: 10px;")
        button_layout.addWidget(self.undo_btn)
        
        self.save_btn = QPushButton("💾 Save Boundaries")
        self.save_btn.clicked.connect(self.save_boundaries)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        self.status_label = QLabel("✅ Ready")
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: #E8F5E8; color: #2E7D32; "
            "border: 1px solid #4CAF50; border-radius: 5px; font-weight: bold;"
        )
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        self.on_camera_type_changed()

    def on_camera_type_changed(self):
        is_usb = self.usb_radio.isChecked()
        self.camera_combo.setEnabled(is_usb)
        self.ip_url_input.setEnabled(not is_usb)

    def refresh_cameras(self):
        self.camera_combo.clear()
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        else:
            if self.usb_radio.isChecked():
                camera_source = self.camera_combo.currentIndex()
            else:
                camera_source = self.ip_url_input.text().strip()
                if not camera_source:
                    QMessageBox.warning(self, "Error", "Please enter RTSP URL!")
                    return
                if not camera_source.startswith("rtsp://"):
                    QMessageBox.warning(self, "Error", "Invalid RTSP URL! Must start with rtsp://")
                    return
            
            self.camera_thread = CameraThread(camera_source)
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.error_signal.connect(self.handle_camera_error)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.start()
            
            self.start_camera_btn.setText("⏹️ Stop Camera")
            self.start_camera_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px;")
            self.capture_btn.setEnabled(True)
            self.image_area.setCurrentWidget(self.camera_label)

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.camera_label.clear()
        self.camera_label.setText("📹 Start camera to see preview")
        self.start_camera_btn.setText("Start Camera")
        self.start_camera_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.capture_btn.setEnabled(False)
        self.image_area.setCurrentWidget(self.camera_label)

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
        self.current_frame = frame.copy()

    def capture_frame(self):
        if hasattr(self, 'current_frame'):
            self.stop_camera()
            self.drawing_widget.set_image(self.current_frame)
            self.captured_frame = self.current_frame.copy()
            self.image_area.setCurrentWidget(self.drawing_widget)
            self.update_instruction_text()
            logger.info("Frame captured for training")
        else:
            QMessageBox.warning(self, "Error", "No frame available!")

    def update_instruction_text(self):
        instruction_text = self.drawing_widget.get_instruction_text()
        self.instruction_label.setText(f"✏️ {instruction_text}")

    def clear_boundaries(self):
        self.drawing_widget.clear_boundaries()
        self.update_instruction_text()

    def undo_last_boundary(self):
        if hasattr(self.drawing_widget, 'undo_last_polygon'):
            removed = self.drawing_widget.undo_last_polygon()
            if removed:
                logger.info("Last boundary removed")
            else:
                logger.info("No boundary to remove")
        self.update_instruction_text()

    def save_boundaries(self):
        oil_can_count = len([b for b in self.drawing_widget.boundaries if b.get('type') == 'oil_can'])
        bunk_hole_count = len([b for b in self.drawing_widget.boundaries if b.get('type') == 'bunk_hole'])
        
        if oil_can_count != 3 or bunk_hole_count != 3:
            QMessageBox.warning(self, "Error", "Please draw exactly 3 oil can and 3 bunk hole boundaries (6 total)!")
            return
        
        oil_can_boundaries = [b for b in self.drawing_widget.boundaries if b.get('type') == 'oil_can']
        bunk_hole_boundaries = [b for b in self.drawing_widget.boundaries if b.get('type') == 'bunk_hole']
        
        data = {
            'oil_can_boundaries': oil_can_boundaries,
            'bunk_hole_boundaries': bunk_hole_boundaries,
            'all_boundaries': self.drawing_widget.boundaries,
            'frame_shape': self.captured_frame.shape
        }
        
        try:
            with open('boundaries.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            cv2.imwrite('reference_frame.jpg', self.captured_frame)
            
            logger.info(f"✅ Boundaries saved: {oil_can_count} oil cans, {bunk_hole_count} bunk holes")
            
            QMessageBox.information(
                self, "Success",
                f"✅ Boundaries saved successfully!\n\n"
                f"🔵 Oil Cans: {oil_can_count}\n"
                f"🟠 Bunk Holes: {bunk_hole_count}\n\n"
                f"Pairs: OC1↔BH1, OC2↔BH2, OC3↔BH3"
            )
        except Exception as e:
            logger.error(f"Failed to save boundaries: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def handle_camera_error(self, error):
        logger.error(f"Camera error: {error}")

    def update_status(self, status):
        self.status_label.setText(f"📹 {status}")

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

class DetectionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_thread = None
        self.detection_thread = None
        self.relay_thread = None
        self.watchdog = None
        self.boundaries = []
        self.oil_can_boundaries = []
        self.bunk_hole_boundaries = []
        self.model_path = None
        self.running = False
        self.detection_count = 0
        self.error_count = 0
        self.mismatch_count = 0
        self.detection_history = deque(maxlen=100)
        self.uptime_start = None
        self.load_boundaries()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Detection - Oil Can & Bunk Hole Pairing System")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        controls_layout = QHBoxLayout()
        
        camera_type_layout = QVBoxLayout()
        self.camera_type_group = QButtonGroup()
        self.usb_radio = QRadioButton("USB")
        self.ip_radio = QRadioButton("IP Camera")
        self.usb_radio.setChecked(True)
        self.camera_type_group.addButton(self.usb_radio)
        self.camera_type_group.addButton(self.ip_radio)
        camera_type_layout.addWidget(self.usb_radio)
        camera_type_layout.addWidget(self.ip_radio)
        controls_layout.addLayout(camera_type_layout)
        
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        controls_layout.addWidget(QLabel("USB:"))
        controls_layout.addWidget(self.camera_combo)
        
        controls_layout.addWidget(QLabel("RTSP:"))
        self.ip_url_input = QLineEdit()
        self.ip_url_input.setPlaceholderText("rtsp://admin:pass@192.168.1.64:554/stream")
        self.ip_url_input.setText("rtsp://admin:Pass_123@192.168.1.64:554/stream")
        self.ip_url_input.setMinimumWidth(300)
        controls_layout.addWidget(self.ip_url_input)
        
        self.usb_radio.toggled.connect(self.on_camera_type_changed)
        self.ip_radio.toggled.connect(self.on_camera_type_changed)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_btn)
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        controls_layout.addWidget(self.start_btn)
        
        layout.addLayout(controls_layout)
        
        self.detection_label = QLabel("Detection View")
        self.detection_label.setMinimumSize(800, 600)
        self.detection_label.setStyleSheet("border: 2px solid #333;")
        self.detection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detection_label)
        
        metrics_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("padding: 5px; font-weight: bold; background-color: #E3F2FD;")
        metrics_layout.addWidget(self.fps_label)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setStyleSheet("padding: 5px; font-weight: bold; background-color: #E3F2FD;")
        metrics_layout.addWidget(self.detection_count_label)
        
        self.mismatch_count_label = QLabel("Mismatches: 0")
        self.mismatch_count_label.setStyleSheet("padding: 5px; font-weight: bold; background-color: #FFEBEE;")
        metrics_layout.addWidget(self.mismatch_count_label)
        
        layout.addLayout(metrics_layout)
        
        # Pair status display
        pair_status_layout = QHBoxLayout()
        pair_label = QLabel("Pair Status:")
        pair_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        pair_status_layout.addWidget(pair_label)
        
        self.pair_status_labels = []
        for i in range(3):
            status_label = QLabel(f"Pair {i + 1}: --")
            status_label.setStyleSheet(
                "padding: 10px; font-size: 11px; font-weight: bold; "
                "background-color: #cccccc; border-radius: 5px; min-width: 150px;"
            )
            status_label.setAlignment(Qt.AlignCenter)
            pair_status_layout.addWidget(status_label)
            self.pair_status_labels.append(status_label)
        
        layout.addLayout(pair_status_layout)
        
        self.overall_status = QLabel("System Status: Idle")
        self.overall_status.setStyleSheet(
            "padding: 15px; font-size: 16px; font-weight: bold; "
            "background-color: #f0f0f0; border-radius: 5px;"
        )
        self.overall_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.overall_status)
        
        self.health_label = QLabel("System Health: Ready")
        self.health_label.setStyleSheet(
            "padding: 10px; font-size: 12px; "
            "background-color: #E8F5E8; color: #2E7D32; border-radius: 5px;"
        )
        layout.addWidget(self.health_label)
        
        stats_layout = QHBoxLayout()
        
        self.uptime_label = QLabel("Uptime: 00:00:00")
        self.uptime_label.setStyleSheet("padding: 5px; font-size: 11px; background-color: #F5F5F5;")
        stats_layout.addWidget(self.uptime_label)
        
        self.success_rate_label = QLabel("Success Rate: 100%")
        self.success_rate_label.setStyleSheet("padding: 5px; font-size: 11px; background-color: #F5F5F5;")
        stats_layout.addWidget(self.success_rate_label)
        
        self.relay_state_label = QLabel("Relay: Idle")
        self.relay_state_label.setStyleSheet("padding: 5px; font-size: 11px; background-color: #F5F5F5;")
        stats_layout.addWidget(self.relay_state_label)
        
        layout.addLayout(stats_layout)
        
        self.setLayout(layout)
        
        self.uptime_timer = QTimer()
        self.uptime_timer.timeout.connect(self.update_uptime)
        
        self.health_check_timer = QTimer()
        self.health_check_timer.timeout.connect(self.check_system_health)
        self.health_check_timer.start(5000)
        
        self.on_camera_type_changed()

    def on_camera_type_changed(self):
        is_usb = self.usb_radio.isChecked()
        self.camera_combo.setEnabled(is_usb)
        self.ip_url_input.setEnabled(not is_usb)

    def refresh_cameras(self):
        self.camera_combo.clear()
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def load_boundaries(self):
        try:
            if os.path.exists('boundaries.json'):
                with open('boundaries.json', 'r') as f:
                    data = json.load(f)
                
                self.oil_can_boundaries = data.get('oil_can_boundaries', [])
                self.bunk_hole_boundaries = data.get('bunk_hole_boundaries', [])
                self.boundaries = data.get('all_boundaries', [])
                self.reference_frame_shape = data.get('frame_shape', None)
                
                logger.info(f"✅ Boundaries loaded: {len(self.oil_can_boundaries)} oil cans, {len(self.bunk_hole_boundaries)} bunk holes")
            else:
                logger.warning("No boundaries file found")
                QMessageBox.warning(self, "Warning", "No boundaries found. Please train first!")
        except Exception as e:
            logger.error(f"Failed to load boundaries: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load boundaries: {str(e)}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "YOLO Model (*.pt *.onnx *.engine);;All Files (*)"
        )
        
        if file_path:
            self.model_path = file_path
            self.start_btn.setEnabled(True)
            logger.info(f"Model selected: {file_path}")
            QMessageBox.information(self, "Success", f"Model loaded: {os.path.basename(file_path)}")

    def toggle_detection(self):
        if self.running:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self):
        if not self.model_path:
            QMessageBox.warning(self, "Error", "Please load a model first!")
            return
        
        if not self.boundaries:
            QMessageBox.warning(self, "Error", "No boundaries defined!")
            return
        
        try:
            logger.info("="*60)
            logger.info("STARTING PAIRING DETECTION SYSTEM")
            logger.info("="*60)
            
            self.detection_count = 0
            self.error_count = 0
            self.mismatch_count = 0
            self.detection_history.clear()
            self.uptime_start = time.time()
            
            if self.usb_radio.isChecked():
                camera_source = self.camera_combo.currentIndex()
                camera_desc = f"USB Camera {camera_source}"
            else:
                camera_source = self.ip_url_input.text().strip()
                if not camera_source:
                    QMessageBox.warning(self, "Error", "Please enter RTSP URL!")
                    return
                if not camera_source.startswith("rtsp://"):
                    QMessageBox.warning(self, "Error", "Invalid RTSP URL! Must start with rtsp://")
                    return
                camera_desc = "IP Camera (RTSP)"
            
            self.camera_thread = CameraThread(camera_source)
            self.camera_thread.frame_ready.connect(self.on_frame_ready)
            self.camera_thread.error_signal.connect(self.handle_error)
            self.camera_thread.status_signal.connect(self.update_camera_status)
            self.camera_thread.start()

            camera_wait_deadline = time.time() + 10.0
            camera_ok = False
            while time.time() < camera_wait_deadline:
                if getattr(self.camera_thread, 'camera', None) and self.camera_thread.camera.isOpened():
                    camera_ok = True
                    break
                QThread.msleep(200)

            if not camera_ok:
                raise Exception("Camera failed to initialize")
            
            # Calculate watchdog timeout based on model size
            model_size_mb = os.path.getsize(self.model_path) / (1024 ** 2)
            watchdog_timeout = max(CONFIG["detection"]["watchdog_timeout_base"], int(model_size_mb / 10))
            
            self.detection_thread = DetectionThread(self.model_path, self.boundaries)
            self.detection_thread.detection_ready.connect(self.on_detection_ready)
            self.detection_thread.error_signal.connect(self.handle_error)
            self.detection_thread.fps_signal.connect(self.update_fps)
            self.detection_thread.start()

            model_wait_deadline = time.time() + 15.0
            model_ok = False
            while time.time() < model_wait_deadline:
                if getattr(self.detection_thread, 'model', None):
                    model_ok = True
                    break
                QThread.msleep(200)

            if not model_ok:
                raise Exception("YOLO model failed to load")
            
            self.relay_thread = RelayControlThread()
            self.relay_thread.error_signal.connect(self.handle_relay_error)
            self.relay_thread.status_signal.connect(self.update_relay_status)
            self.relay_thread.start()
            
            self.watchdog = WatchdogTimer("Detection System", timeout_seconds=watchdog_timeout)
            self.watchdog.timeout_signal.connect(self.handle_watchdog_timeout)
            self.watchdog.start()
            
            self.uptime_timer.start(1000)
            
            self.running = True
            self.start_btn.setText("⏹️ Stop Detection")
            self.start_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
            self.overall_status.setText("System Status: Running")
            self.overall_status.setStyleSheet(
                "padding: 15px; font-size: 16px; font-weight: bold; "
                "background-color: #4CAF50; color: white; border-radius: 5px;"
            )
            self.health_label.setText("✅ System Health: All threads running")
            self.health_label.setStyleSheet(
                "padding: 10px; font-size: 12px; "
                "background-color: #E8F5E8; color: #2E7D32; border-radius: 5px;"
            )
            
            logger.info("✅ Detection system started successfully")
            logger.info(f"Camera: {camera_desc}")
            logger.info(f"Model: {os.path.basename(self.model_path)}")
            logger.info(f"Watchdog timeout: {watchdog_timeout}s")
            logger.info(f"Boundaries: {len(self.oil_can_boundaries)} oil cans, {len(self.bunk_hole_boundaries)} bunk holes")
            
        except Exception as e:
            logger.critical(f"Failed to start detection: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Startup Error", f"Failed to start system:\n{str(e)}\n\nCheck logs for details.")
            self.stop_detection()

    def stop_detection(self):
        logger.info("="*60)
        logger.info("STOPPING DETECTION SYSTEM")
        logger.info("="*60)
        
        self.running = False
        
        if hasattr(self, 'uptime_timer'):
            self.uptime_timer.stop()
        
        if self.uptime_start:
            uptime_seconds = time.time() - self.uptime_start
            logger.info(f"Session uptime: {uptime_seconds:.0f} seconds")
            logger.info(f"Total detections: {self.detection_count}")
            logger.info(f"Total mismatches: {self.mismatch_count}")
            logger.info(f"Total errors: {self.error_count}")
            if self.detection_count > 0:
                success_rate = ((self.detection_count - self.mismatch_count) / self.detection_count) * 100
                logger.info(f"Success rate: {success_rate:.1f}%")
        
        threads_to_stop = [
            ("Watchdog", self.watchdog),
            ("Camera", self.camera_thread),
            ("Detection", self.detection_thread),
            ("Relay", self.relay_thread)
        ]
        
        for name, thread in threads_to_stop:
            if thread:
                try:
                    logger.info(f"Stopping {name} thread...")
                    thread.stop()
                    if not thread.wait(3000):
                        logger.warning(f"{name} thread did not stop gracefully")
                except Exception as e:
                    logger.error(f"Error stopping {name} thread: {e}")
        
        self.watchdog = None
        self.camera_thread = None
        self.detection_thread = None
        self.relay_thread = None
        
        self.start_btn.setText("Start Detection")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.detection_label.clear()
        self.detection_label.setText("Detection View")
        self.overall_status.setText("System Status: Stopped")
        self.overall_status.setStyleSheet(
            "padding: 15px; font-size: 16px; font-weight: bold; "
            "background-color: #f0f0f0; border-radius: 5px;"
        )
        self.health_label.setText("⏹️ System Health: Stopped")
        self.health_label.setStyleSheet(
            "padding: 10px; font-size: 12px; "
            "background-color: #F5F5F5; color: #666; border-radius: 5px;"
        )
        
        for label in self.pair_status_labels:
            label.setText(label.text().split(':')[0] + ": --")
            label.setStyleSheet(
                "padding: 10px; font-size: 11px; font-weight: bold; "
                "background-color: #cccccc; border-radius: 5px; min-width: 150px;"
            )
        
        logger.info("Detection system stopped successfully")
        logger.info("="*60)

    def on_frame_ready(self, frame):
        if self.detection_thread and self.running:
            try:
                if hasattr(self, 'reference_frame_shape') and self.reference_frame_shape:
                    ref_h, ref_w = int(self.reference_frame_shape[0]), int(self.reference_frame_shape[1])
                    h, w = frame.shape[0], frame.shape[1]
                    
                    # Check aspect ratio
                    ref_aspect = ref_w / ref_h
                    curr_aspect = w / h
                    
                    if abs(ref_aspect - curr_aspect) > 0.05:
                        logger.warning(
                            f"⚠️ Aspect ratio mismatch! Reference: {ref_aspect:.2f}, "
                            f"Current: {curr_aspect:.2f}. Boundaries may be inaccurate."
                        )
                    
                    if (h, w) != (ref_h, ref_w):
                        resized = cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
                        self.detection_thread.add_frame(resized)
                    else:
                        self.detection_thread.add_frame(frame)
                else:
                    self.detection_thread.add_frame(frame)
            except Exception as e:
                logger.error(f"Frame resize/forward error: {e}")
            
            if self.watchdog:
                self.watchdog.heartbeat()

    def on_detection_ready(self, result):
        try:
            self.detection_count += 1
            self.detection_count_label.setText(f"Detections: {self.detection_count}")
            
            all_ok = result['all_ok']
            pair_statuses = result['pair_statuses']
            mismatched_pairs = result['mismatched_pairs']
            
            if not all_ok:
                self.mismatch_count += 1
                self.mismatch_count_label.setText(f"Mismatches: {self.mismatch_count}")
            
            self.detection_history.append(all_ok)
            
            if len(self.detection_history) > 0:
                success_count = sum(self.detection_history)
                success_rate = (success_count / len(self.detection_history)) * 100
                self.success_rate_label.setText(f"Success Rate: {success_rate:.1f}%")
            
            # Update pair status labels
            for i, pair_status in enumerate(pair_statuses):
                if i < len(self.pair_status_labels):
                    oc_present = pair_status['oc_present']
                    bh_present = pair_status['bh_present']
                    status = pair_status['status']
                    
                    if status == 'ignored':
                        text = f"Pair {i + 1}: Both Absent"
                        color = "#9E9E9E"
                        text_color = "white"
                    elif status == 'ok':
                        text = f"Pair {i + 1}: ✓ Both Present"
                        color = "#4CAF50"
                        text_color = "white"
                    else:  # mismatch
                        if oc_present and not bh_present:
                            text = f"Pair {i + 1}: ⚠️ OC only"
                        else:
                            text = f"Pair {i + 1}: ⚠️ BH only"
                        color = "#F44336"
                        text_color = "white"
                    
                    self.pair_status_labels[i].setText(text)
                    self.pair_status_labels[i].setStyleSheet(
                        f"padding: 10px; font-size: 11px; font-weight: bold; "
                        f"background-color: {color}; color: {text_color}; "
                        f"border-radius: 5px; min-width: 150px;"
                    )
            
            if all_ok:
                overall_text = "✅ All Pairs OK"
                overall_color = "#4CAF50"
                logger.debug(f"Detection #{self.detection_count}: All Pairs OK")
            else:
                mismatch_details = []
                for mismatch in mismatched_pairs:
                    pair_idx = mismatch['pair_index']
                    if mismatch['oc_present'] and not mismatch['bh_present']:
                        mismatch_details.append(f"Pair{pair_idx}:OC✓BH✗")
                    else:
                        mismatch_details.append(f"Pair{pair_idx}:OC✗BH✓")
                
                overall_text = f"⚠️ Mismatch: {', '.join(mismatch_details)}"
                overall_color = "#F44336"
                logger.warning(f"Detection #{self.detection_count}: Mismatches {mismatch_details}")
            
            self.overall_status.setText(overall_text)
            self.overall_status.setStyleSheet(
                f"padding: 15px; font-size: 16px; font-weight: bold; "
                f"background-color: {overall_color}; color: white; border-radius: 5px;"
            )
            
            if self.relay_thread:
                self.relay_thread.set_state(all_ok)
            
            self.draw_frame(result)
            
            if self.watchdog:
                self.watchdog.heartbeat()
            
            if self.detection_count % 100 == 0:
                logger.info(f"Milestone: {self.detection_count} detections completed")
            
        except Exception as e:
            logger.error(f"Error processing detection result: {e}\n{traceback.format_exc()}")
            self.error_count += 1

    def draw_frame(self, result):
        try:
            frame = result['frame'].copy()
            pair_statuses = result['pair_statuses']
            
            # Draw boundaries with status colors
            for i, boundary in enumerate(self.oil_can_boundaries):
                if i < len(pair_statuses):
                    status = pair_statuses[i]['status']
                    if status == 'ok':
                        color = (0, 255, 0)  # Green - OK
                    elif status == 'ignored':
                        color = (128, 128, 128)  # Gray - Ignored
                    else:
                        color = (0, 0, 255)  # Red - Mismatch
                else:
                    color = (255, 0, 0)  # Blue - Default
                
                if 'polygon' in boundary and boundary.get('polygon'):
                    pts = np.array(boundary['polygon'], dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)
                    x0, y0 = pts[0]
                    cv2.putText(frame, f"OC{i + 1}", (int(x0), int(y0) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            for i, boundary in enumerate(self.bunk_hole_boundaries):
                if i < len(pair_statuses):
                    status = pair_statuses[i]['status']
                    if status == 'ok':
                        color = (0, 255, 0)  # Green - OK
                    elif status == 'ignored':
                        color = (128, 128, 128)  # Gray - Ignored
                    else:
                        color = (0, 0, 255)  # Red - Mismatch
                else:
                    color = (0, 165, 255)  # Orange - Default
                
                if 'polygon' in boundary and boundary.get('polygon'):
                    pts = np.array(boundary['polygon'], dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)
                    x0, y0 = pts[0]
                    cv2.putText(frame, f"BH{i + 1}", (int(x0), int(y0) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw detected objects
            for obj in result.get('oil_cans', []):
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"Oil: {obj['confidence']:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            for obj in result.get('bunk_holes', []):
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                cv2.putText(frame, f"Bunk: {obj['confidence']:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # Add timestamp and status
            timestamp = datetime.fromtimestamp(result['timestamp']).strftime('%H:%M:%S')
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            status_text = "OK" if result['all_ok'] else "MISMATCH"
            status_color = (0, 255, 0) if result['all_ok'] else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.detection_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.detection_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error drawing frame: {e}")

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_relay_status(self, status):
        self.relay_state_label.setText(f"Relay: {status}")
        if "OK" in status:
            self.relay_state_label.setStyleSheet(
                "padding: 5px; font-size: 11px; background-color: #C8E6C9; color: #2E7D32;"
            )
        else:
            self.relay_state_label.setStyleSheet(
                "padding: 5px; font-size: 11px; background-color: #FFCDD2; color: #C62828;"
            )
        logger.debug(f"Relay status updated: {status}")
    
    def update_camera_status(self, status):
        logger.info(f"Camera status: {status}")

    def handle_error(self, error):
        self.error_count += 1
        logger.error(f"Thread error: {error}")
        self.health_label.setText(f"⚠️ Error: {error[:50]}...")
        self.health_label.setStyleSheet(
            "padding: 10px; font-size: 12px; "
            "background-color: #FFEBEE; color: #C62828; border-radius: 5px;"
        )
        
        if self.error_count > 0 and self.error_count % 3 == 0:
            logger.warning(f"Multiple errors detected ({self.error_count}), attempting recovery...")
            QTimer.singleShot(2000, self.attempt_recovery)
    
    def handle_relay_error(self, error):
        logger.error(f"Relay error: {error}")
        self.relay_state_label.setText("Relay: ERROR")
        self.relay_state_label.setStyleSheet(
            "padding: 5px; font-size: 11px; background-color: #FFCDD2; color: #C62828;"
        )

    def handle_watchdog_timeout(self, name):
        logger.critical(f"⚠️ WATCHDOG TIMEOUT: {name} - System frozen detected!")
        logger.critical("Initiating emergency recovery...")
        
        QMessageBox.critical(
            self, "System Frozen",
            f"Detection system has frozen!\n\n"
            f"Component: {name}\n"
            f"Action: Automatic restart in 3 seconds..."
        )
        
        self.stop_detection()
        QTimer.singleShot(3000, self.attempt_recovery)
    
    def attempt_recovery(self):
        if not self.running:
            logger.info("Attempting automatic system recovery...")
            try:
                self.start_detection()
                logger.info("✅ Recovery successful")
                QMessageBox.information(self, "Recovery", "System recovered and restarted successfully!")
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                QMessageBox.critical(self, "Recovery Failed", f"Automatic recovery failed:\n{str(e)}\n\nManual intervention required.")
    
    def update_uptime(self):
        if self.uptime_start:
            uptime_seconds = int(time.time() - self.uptime_start)
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            seconds = uptime_seconds % 60
            self.uptime_label.setText(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def check_system_health(self):
        if not self.running:
            return
        
        health_issues = []
        
        if not self.camera_thread or not self.camera_thread.isRunning():
            health_issues.append("Camera thread dead")
        
        if not self.detection_thread or not self.detection_thread.isRunning():
            health_issues.append("Detection thread dead")
        
        if not self.relay_thread or not self.relay_thread.isRunning():
            health_issues.append("Relay thread dead")
        
        if not self.watchdog or not self.watchdog.isRunning():
            health_issues.append("Watchdog dead")
        
        current_fps_text = self.fps_label.text()
        try:
            fps_value = float(current_fps_text.split(':')[1].strip())
            if fps_value < 1.0:
                health_issues.append("Low FPS")
        except:
            pass
        
        if health_issues:
            health_text = f"⚠️ Issues: {', '.join(health_issues)}"
            logger.warning(f"System health check: {health_text}")
            self.health_label.setText(health_text)
            self.health_label.setStyleSheet(
                "padding: 10px; font-size: 12px; "
                "background-color: #FFF3E0; color: #E65100; border-radius: 5px;"
            )
            
            if len(health_issues) >= 2:
                logger.error("Critical system health issues detected!")
                self.attempt_recovery()
        else:
            if self.mismatch_count == 0:
                self.health_label.setText("✅ System Health: Excellent")
                self.health_label.setStyleSheet(
                    "padding: 10px; font-size: 12px; "
                    "background-color: #E8F5E8; color: #2E7D32; border-radius: 5px;"
                )
            else:
                self.health_label.setText(f"✅ System Health: Good ({self.mismatch_count} mismatches)")
                self.health_label.setStyleSheet(
                    "padding: 10px; font-size: 12px; "
                    "background-color: #E8F5E8; color: #2E7D32; border-radius: 5px;"
                )

    def closeEvent(self, event):
        logger.info("Application close requested")
        
        if self.running:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Detection is running. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        self.stop_detection()
        
        if hasattr(self, 'health_check_timer'):
            self.health_check_timer.stop()
        
        event.accept()
        logger.info("Detection page closed")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_logs)
        self.auto_save_timer.start(300000)

    def init_ui(self):
        self.setWindowTitle("Industrial Detection System v4.0 - Oil Can & Bunk Hole Pairing")
        self.setGeometry(100, 100, 1400, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        self.training_page = TrainingPage()
        self.detection_page = DetectionPage()
        
        self.tab_widget.addTab(self.training_page, "🎯 Training")
        self.tab_widget.addTab(self.detection_page, "🔍 Detection")
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        self.create_menu_bar()
        
        self.statusBar().showMessage("System Ready - Draw 6 Boundaries (3 pairs)")
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_bar)
        self.status_timer.start(2000)
        
        logger.info("="*60)
        logger.info("APPLICATION STARTED - v4.0 PAIRING SYSTEM")
        logger.info("="*60)
        logger.info(f"System: {sys.platform}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"OpenCV: {cv2.__version__}")

    def on_tab_changed(self, index):
        if self.tab_widget.widget(index) is self.detection_page:
            self.detection_page.load_boundaries()

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        export_stats_action = QAction('Export Statistics', self)
        export_stats_action.triggered.connect(self.export_statistics)
        file_menu.addAction(export_stats_action)
        
        view_logs_action = QAction('View Logs', self)
        view_logs_action.triggered.connect(self.view_logs)
        file_menu.addAction(view_logs_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tools_menu = menubar.addMenu('Tools')
        
        test_relay_action = QAction('Test Relay', self)
        test_relay_action.triggered.connect(self.test_relay)
        tools_menu.addAction(test_relay_action)
        
        test_camera_action = QAction('Test All Cameras', self)
        test_camera_action.triggered.connect(self.test_cameras)
        tools_menu.addAction(test_camera_action)
        
        system_info_action = QAction('System Information', self)
        system_info_action.triggered.connect(self.show_system_info)
        tools_menu.addAction(system_info_action)
        
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        user_manual_action = QAction('User Manual', self)
        user_manual_action.triggered.connect(self.show_user_manual)
        help_menu.addAction(user_manual_action)

    def update_status_bar(self):
        if self.detection_page.running:
            uptime = self.detection_page.uptime_label.text()
            fps = self.detection_page.fps_label.text()
            detections = self.detection_page.detection_count
            mismatches = self.detection_page.mismatch_count
            self.statusBar().showMessage(f"🟢 RUNNING | {uptime} | {fps} | Detections: {detections} | Mismatches: {mismatches}")
        else:
            self.statusBar().showMessage("⚪ IDLE | System ready")

    def export_statistics(self):
        try:
            if not self.detection_page.detection_history:
                QMessageBox.information(self, "No Data", "No detection data available to export")
                return
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Statistics", 
                f"pairing_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write("Detection Number,Status,Timestamp\n")
                    for i, status in enumerate(self.detection_page.detection_history):
                        f.write(f"{i+1},{'OK' if status else 'MISMATCH'},{datetime.now().isoformat()}\n")
                
                QMessageBox.information(self, "Success", f"Statistics exported to:\n{filename}")
                logger.info(f"Statistics exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def auto_save_logs(self):
        logger.info("Auto-save checkpoint")

    def test_relay(self):
        try:
            relay = pyhid_usb_relay.find()
            
            reply = QMessageBox.question(
                self, 'Test Relay',
                'This will toggle both relays for testing.\nContinue?',
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                relay.set_state(1, True)
                relay.set_state(2, False)
                QThread.msleep(1000)
                
                relay.set_state(1, False)
                relay.set_state(2, True)
                QThread.msleep(1000)
                
                relay.set_state(1, False)
                relay.set_state(2, False)
                
                QMessageBox.information(self, "Test Complete", "Relay test completed successfully!")
                logger.info("Relay test completed")
        except Exception as e:
            logger.error(f"Relay test failed: {e}")
            QMessageBox.critical(self, "Test Failed", f"Relay test failed:\n{str(e)}")

    def test_cameras(self):
        try:
            available_cameras = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(f"Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                    cap.release()
            
            if available_cameras:
                QMessageBox.information(
                    self, "Camera Test",
                    f"Found {len(available_cameras)} camera(s):\n\n" + "\n".join(available_cameras)
                )
                logger.info(f"Camera test: {len(available_cameras)} cameras found")
            else:
                QMessageBox.warning(self, "Camera Test", "No cameras found!")
        except Exception as e:
            logger.error(f"Camera test failed: {e}")
            QMessageBox.critical(self, "Test Failed", f"Camera test failed:\n{str(e)}")

    def show_system_info(self):
        try:
            import platform
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                info = f"""System Information:

Platform: {platform.system()} {platform.release()}
Processor: {platform.processor()}
CPU Usage: {cpu_percent}%
RAM: {memory.percent}% used ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)
Disk: {disk.percent}% used ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)

Python: {sys.version.split()[0]}
OpenCV: {cv2.__version__}

Application:
Running: {'Yes' if self.detection_page.running else 'No'}
Total Detections: {self.detection_page.detection_count}
Total Mismatches: {self.detection_page.mismatch_count}
Total Errors: {self.detection_page.error_count}

Configuration:
Oil Can Threshold: {CONFIG['detection']['confidence_thresholds']['oil_can']}
Bunk Hole Threshold: {CONFIG['detection']['confidence_thresholds']['bunk_hole']}
"""
                QMessageBox.information(self, "System Information", info)
            except:
                info = f"""System Information:

Platform: {platform.system()} {platform.release()}
Python: {sys.version.split()[0]}
OpenCV: {cv2.__version__}

Application:
Running: {'Yes' if self.detection_page.running else 'No'}
Total Detections: {self.detection_page.detection_count}
Total Mismatches: {self.detection_page.mismatch_count}

(Install psutil for detailed system info)
"""
                QMessageBox.information(self, "System Information", info)
        except Exception as e:
            QMessageBox.information(
                self, "System Information",
                f"Basic Info:\nPython: {sys.version.split()[0]}\nOpenCV: {cv2.__version__}"
            )

    def show_user_manual(self):
        manual = """USER MANUAL - OIL CAN & BUNK HOLE PAIRING SYSTEM v4.0

=== MODEL REQUIREMENTS ===
YOLO Model must have these classes:
- Class 0: oil_can
- Class 1: bunk_hole

=== NEW PAIRING LOGIC ===
The system now monitors OIL CAN + BUNK HOLE PAIRS:
• Pair 1: OC1 ↔ BH1
• Pair 2: OC2 ↔ BH2  
• Pair 3: OC3 ↔ BH3

SYSTEM OK when:
✓ Both oil can AND bunk hole present in a pair, OR
✓ Both oil can AND bunk hole absent in a pair

SYSTEM ALARM when:
✗ Oil can present BUT bunk hole missing
✗ Bunk hole present BUT oil can missing

=== TRAINING MODE ===
1. Select camera type (USB or IP Camera)
2. For IP: Enter RTSP URL (e.g., rtsp://admin:pass@ip:554/stream)
3. Click "Start Camera" to preview
4. Click "Capture Photo" when ready
5. Draw boundaries in this order:
   a) 3 Oil Can boundaries (Blue) - OC1, OC2, OC3
   b) 3 Bunk Hole boundaries (Orange) - BH1, BH2, BH3
6. Total 6 boundaries required (3 pairs)
7. Click "Save Boundaries"

=== DETECTION MODE ===
1. Select camera type and source
2. Click "Load Model" and select YOLO model (.pt file)
3. Click "Start Detection"
4. Monitor real-time pair status
5. System triggers alarm only on pair mismatches

=== PAIR STATUS INDICATORS ===
• Gray "Both Absent" - Pair ignored (no alarm)
• Green "Both Present" - Pair OK
• Red "OC only" or "BH only" - MISMATCH (alarm)

=== CONFIDENCE THRESHOLDS ===
(Configurable in CONFIG at top of code)
- Oil Can: 0.45 (45%)
- Bunk Hole: 0.40 (40%)

Adjust these based on your detection accuracy needs.

=== RELAY CONTROL ===
- Relay R1=ON, R2=OFF: All pairs OK
- Relay R1=OFF, R2=ON: Pair mismatch detected

=== IP CAMERA SUPPORT ===
- Auto-reconnection with exponential backoff
- RTSP stream buffering optimization
- Supports Hikvision, Dahua, and other RTSP cameras

RTSP URL Formats:
Hikvision: rtsp://user:pass@ip:554/Streaming/Channels/101
Dahua: rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0
Generic: rtsp://user:pass@ip:554/stream

=== IMPROVEMENTS IN v4.0 ===
✅ Pairing logic: Only alarms on mismatches
✅ Handles partial production runs automatically
✅ Model class validation on startup
✅ Configurable confidence thresholds per category
✅ Exponential backoff for camera reconnection
✅ Adaptive watchdog timeout based on model size
✅ Aspect ratio validation for frame resize
✅ Removed unused lance/teflon detection
✅ Enhanced logging and error recovery

=== TROUBLESHOOTING ===
- Model validation error: Ensure model has correct classes
- Pair mismatch: Check boundary positions match physical layout
- Low FPS: Reduce camera resolution or confidence thresholds
- Camera reconnection loops: Check network and credentials
- False alarms: Adjust confidence thresholds in CONFIG

=== SYSTEM FEATURES ===
✓ Multi-threaded architecture
✓ Watchdog timer with adaptive timeout
✓ Auto error recovery
✓ Production-grade logging
✓ Performance monitoring
✓ Real-time FPS display
✓ Success rate tracking
✓ System health monitoring
✓ 24/7 operation ready

=== LOGS & SUPPORT ===
View logs: File → View Logs
Export stats: File → Export Statistics
System info: Tools → System Information
Test cameras: Tools → Test All Cameras
Test relay: Tools → Test Relay

=== KEYBOARD SHORTCUTS ===
Ctrl+Q: Exit application

=== CONFIGURATION ===
Edit CONFIG dictionary at top of code to adjust:
- Confidence thresholds per class
- Watchdog timeout
- Camera buffer settings
- Reconnection parameters

For technical support, check the log files in the application directory.
Log file format: detection_system_YYYYMMDD.log
"""
        
        msg = QMessageBox()
        msg.setWindowTitle("User Manual - v4.0 Pairing System")
        msg.setText(manual)
        msg.setStyleSheet("QLabel{min-width: 800px; min-height: 600px;}")
        msg.exec_()

    def view_logs(self):
        log_file = f'detection_system_{datetime.now().strftime("%Y%m%d")}.log'
        if os.path.exists(log_file):
            if sys.platform == "win32":
                os.startfile(log_file)
            elif sys.platform == "darwin":
                os.system(f"open {log_file}")
            else:
                os.system(f"xdg-open {log_file}")
        else:
            QMessageBox.information(self, "Logs", "No log file found for today")

    def show_about(self):
        QMessageBox.about(
            self, "About",
            "Industrial Detection System v4.0\n\n"
            "Oil Can & Bunk Hole Pairing System\n"
            "Supports USB and IP Cameras (RTSP)\n\n"
            "Detection Logic:\n"
            "• 3 Oil Can + Bunk Hole pairs monitored\n"
            "• Alarm only on pair mismatches\n"
            "• Handles partial production runs\n\n"
            "Model Requirements:\n"
            "• Class 0: oil_can\n"
            "• Class 1: bunk_hole\n\n"
            "New Features v4.0:\n"
            "✅ Intelligent pairing logic\n"
            "✅ Model class validation\n"
            "✅ Configurable thresholds\n"
            "✅ Exponential backoff reconnection\n"
            "✅ Adaptive watchdog timeout\n"
            "✅ Aspect ratio validation\n"
            "✅ Enhanced error recovery\n"
            "✅ Simplified UI (pairs only)\n\n"
            "Pairing Rules:\n"
            "✓ Both present = OK\n"
            "✓ Both absent = OK (ignored)\n"
            "✗ One present, one absent = ALARM\n\n"
            "Built for 24/7 industrial operation\n\n"
            "© 2025 Credence Technologies Pvt Ltd"
        )

    def closeEvent(self, event):
        logger.info("Application close requested")
        
        if self.training_page.camera_thread:
            self.training_page.camera_thread.stop()
        
        if self.detection_page.running:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Detection system is running!\n\nAre you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.detection_page.stop_detection()
        
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if hasattr(self, 'auto_save_timer'):
            self.auto_save_timer.stop()
        
        logger.info("="*60)
        logger.info("APPLICATION SHUTDOWN")
        logger.info("="*60)
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    sys.excepthook = lambda exc_type, exc_value, exc_tb: logger.critical(
        f"Unhandled exception: {exc_type.__name__}: {exc_value}\n{''.join(traceback.format_tb(exc_tb))}"
    )
    
    window = MainWindow()
    window.show()
    
    logger.info("="*60)
    logger.info("INDUSTRIAL DETECTION SYSTEM v4.0 STARTED")
    logger.info("OIL CAN & BUNK HOLE PAIRING SYSTEM")
    logger.info("="*60)
    
    return_code = app.exec_()
    
    logger.info(f"Application exited with code {return_code}")
    sys.exit(return_code)

if __name__ == '__main__':
    main()