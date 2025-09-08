import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import requests
import base64
import re
import os
import time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    vehicle_type: Optional[str] = None
    license_plate: Optional[str] = None
    parking_hours: Optional[int] = None
    confidence: float = 0.0
    status: str = "idle"
    message: str = ""
    current_phase: str = ""

class DetectionEngine:
    def __init__(self):
        self.detection_result = DetectionResult()
        self.vehicle_classes_mapping = {
            "motorcycle": "Bike",
            "bicycle": "Bike", 
            "car": "Car",
            "bus": "Truck",
            "truck": "Truck"
        }
        
        # Indian state codes for license plate validation
        self.indian_state_codes = {
            'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL',
            'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'OR', 'PB', 'RJ', 'SK', 'TN',
            'TG', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DD', 'DL', 'JK', 'LA', 'LD', 'PY'
        }
        
        # Initialize models
        self.vehicle_model = None
        self.plate_model = None
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        
        # OCR settings
        self.ocr_api_key = "K83315680088957"
        self.ocr_api_url = "https://api.ocr.space/parse/image"
        
        # Hand gesture timing variables
        self.previous_counts = []
        self.current_finger_number = 0
        self.finger_number_start_time = None
        self.finger_display_duration = 3.0  # 3 seconds for finger count stability
        self.confirmation_mode = False
        self.confirmed_finger_number = 0
        self.ok_gesture_counter = 0
        self.ok_gesture_threshold = 8  # Need consecutive OK detections
        self.ok_confirmation_duration = 2.0  # 2 seconds for OK gesture confirmation
        self.last_stable_finger_count = 0
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO models and MediaPipe"""
        try:
            # Load vehicle detection model (YOLO12n)
            self.vehicle_model = YOLO('yolo12n.pt')
            logger.info("Vehicle detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vehicle model: {e}")
        
        try:
            # Load license plate model
            if os.path.exists('models/best.pt'):
                self.plate_model = YOLO('models/best.pt')
                logger.info("License plate model loaded successfully")
            else:
                logger.warning("License plate model not found at models/best.pt")
        except Exception as e:
            logger.error(f"Failed to load license plate model: {e}")
        
        try:
            # Initialize MediaPipe
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.7,
                model_complexity=1
            )
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
    
    def detect_vehicle_in_frame(self, frame: np.ndarray) -> Tuple[bool, str, float]:
        """Detect vehicle in frame"""
        if self.vehicle_model is None:
            return False, None, 0.0
        
        try:
            results = self.vehicle_model(frame, verbose=False)
            best_confidence = 0.0
            best_vehicle_type = None
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.vehicle_model.names[cls]
                        
                        if conf > 0.15 and class_name in self.vehicle_classes_mapping:
                            if conf > best_confidence:
                                best_confidence = conf
                                best_vehicle_type = self.vehicle_classes_mapping[class_name]
                                
                                # Draw bounding box
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{best_vehicle_type} {conf:.2f}", 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return best_confidence > 0.25, best_vehicle_type, best_confidence
            
        except Exception as e:
            logger.error(f"Vehicle detection error: {e}")
            return False, None, 0.0
    
    def detect_license_plate_in_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """Detect license plate in frame"""
        if self.plate_model is None:
            return False, None, 0.0
        
        try:
            results = self.plate_model(frame, verbose=False)
            best_confidence = 0.0
            best_bbox = None
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = float(box.conf)
                    if conf > 0.3 and conf > best_confidence:
                        best_confidence = conf
                        best_bbox = box.xyxy[0].cpu().numpy()
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, best_bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Plate: {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if best_bbox is not None:
                x1, y1, x2, y2 = map(int, best_bbox)
                h, w = frame.shape[:2]
                padding = 20
                y1_pad = max(0, y1-padding)
                y2_pad = min(h, y2+padding)
                x1_pad = max(0, x1-padding)
                x2_pad = min(w, x2+padding)
                
                plate_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                return True, plate_roi, best_confidence
            
            return False, None, 0.0
            
        except Exception as e:
            logger.error(f"License plate detection error: {e}")
            return False, None, 0.0
    
    def process_license_plate_ocr(self, plate_image: np.ndarray) -> Optional[str]:
        """Process license plate using OCR"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        try:
            # Enhance image for better OCR
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
            
            # Apply image enhancement
            enhanced_images = []
            
            # Original
            enhanced_images.append(gray)
            
            # Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_images.append(thresh)
            
            # Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            enhanced_images.append(adaptive)
            
            best_result = None
            best_score = 0
            
            for enhanced_img in enhanced_images:
                try:
                    _, buffer = cv2.imencode('.jpg', enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    img_base64 = base64.b64encode(buffer).decode()
                    
                    payload = {
                        'apikey': self.ocr_api_key,
                        'language': 'eng',
                        'isOverlayRequired': False,
                        'base64Image': f'data:image/jpeg;base64,{img_base64}',
                        'OCREngine': '2',
                        'scale': 'true',
                        'isTable': 'false'
                    }
                    
                    response = requests.post(self.ocr_api_url, data=payload, timeout=15)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if not result.get("IsErroredOnProcessing", True):
                            parsed_results = result.get("ParsedResults", [])
                            for parsed_result in parsed_results:
                                text = parsed_result.get("ParsedText", "").strip()
                                if text:
                                    cleaned_text = self._clean_license_plate_text(text)
                                    if cleaned_text:
                                        score = self._score_license_plate_text(cleaned_text)
                                        if score > best_score:
                                            best_score = score
                                            best_result = cleaned_text
                
                except Exception as e:
                    logger.error(f"OCR processing error: {e}")
                    continue
            
            return best_result
            
        except Exception as e:
            logger.error(f"License plate OCR error: {e}")
            return None
    
    def _clean_license_plate_text(self, text: str) -> Optional[str]:
        """Clean and validate license plate text"""
        if not text:
            return None
        
        # Remove all non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', str(text).upper().strip())
        
        if len(cleaned) < 6:
            return None
        
        # Check if it looks like an Indian license plate
        if self._is_valid_indian_license_plate(cleaned):
            return cleaned
        
        return None
    
    def _is_valid_indian_license_plate(self, text: str) -> bool:
        """Validate Indian license plate format"""
        if len(text) < 8 or len(text) > 10:
            return False
        
        # Check if starts with valid state code
        for state_code in self.indian_state_codes:
            if text.startswith(state_code):
                return True
        
        return False
    
    def _score_license_plate_text(self, text: str) -> int:
        """Score license plate text quality"""
        if not text:
            return 0
        
        score = 0
        
        # Check Indian license plate patterns
        patterns = [
            (r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', 100),
            (r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$', 90),
            (r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$', 85),
        ]
        
        for pattern, pattern_score in patterns:
            if re.match(pattern, text):
                score += pattern_score
                break
        
        # Check for valid state code
        for state_code in self.indian_state_codes:
            if text.startswith(state_code):
                score += 50
                break
        
        # Length scoring
        if 8 <= len(text) <= 10:
            score += 20
        
        return score
    
    def smooth_detection(self, current_count: int, threshold: int = 3) -> int:
        """Smooth detection to reduce noise"""
        self.previous_counts.append(current_count)
        if len(self.previous_counts) > 5:
            self.previous_counts.pop(0)
        
        # Return most common count in recent frames
        if len(self.previous_counts) >= threshold:
            return max(set(self.previous_counts), key=self.previous_counts.count)
        return current_count

    def count_fingers(self, hand_landmarks, hand_label: str) -> int:
        """Count fingers from hand landmarks"""
        if not hand_landmarks:
            return 0
        
        count = 0
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        
        # Thumb detection based on hand orientation
        if hand_label == "Right":
            if landmarks[tip_ids[0]][0] < landmarks[pip_ids[0]][0]:
                count += 1
        else:
            if landmarks[tip_ids[0]][0] > landmarks[pip_ids[0]][0]:
                count += 1
        
        # Other four fingers
        for i in range(1, 5):
            if landmarks[tip_ids[i]][1] < landmarks[pip_ids[i]][1]:
                count += 1
        
        return count
    
    def is_ok_sign(self, hand_landmarks, hand_label: str) -> bool:
        """Detect OK sign gesture"""
        if not hand_landmarks:
            return False
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check if thumb and index finger are close (forming circle)
        distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
        circle_formed = distance < 0.05
        
        # Check if other fingers are extended
        middle_extended = landmarks[12][1] < landmarks[10][1] - 0.02
        ring_extended = landmarks[16][1] < landmarks[14][1] - 0.02
        pinky_extended = landmarks[20][1] < landmarks[18][1] - 0.02
        
        return circle_formed and middle_extended and ring_extended and pinky_extended
    
    def detect_hand_gesture_in_frame(self, frame: np.ndarray) -> Tuple[int, bool, np.ndarray]:
        """Detect hand gestures in frame with timing confirmation"""
        if self.hands is None:
            return 0, False, frame
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            total_fingers = 0
            ok_detected = False
            current_time = time.time()
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_handedness.classification[0].label
                    confidence = hand_handedness.classification[0].score
                    
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Display hand info
                    cv2.putText(frame, f'{hand_label} ({confidence:.2f})', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Check for OK sign
                    if self.is_ok_sign(hand_landmarks, hand_label):
                        ok_detected = True
                    
                    # Count fingers only if not in confirmation mode
                    if not self.confirmation_mode:
                        fingers = self.count_fingers(hand_landmarks, hand_label)
                        total_fingers += fingers
            
            # Handle OK gesture detection with timing
            if ok_detected and self.confirmation_mode:
                self.ok_gesture_counter += 1
                remaining = max(0, self.ok_gesture_threshold - self.ok_gesture_counter)
                
                cv2.putText(frame, 'OK SIGN DETECTED!', (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f'Confirming... {remaining} more frames', (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                
                # Confirmed after threshold consecutive detections
                if self.ok_gesture_counter >= self.ok_gesture_threshold:
                    logger.info(f'Hand gesture confirmed: {self.confirmed_finger_number} fingers')
                    self.detection_result.parking_hours = self.confirmed_finger_number
                    self.detection_result.message = f"Confirmed {self.confirmed_finger_number} parking hours"
                    self.detection_result.status = "confirmed"
                    
                    # Reset timing variables
                    self._reset_gesture_timing()
                    
                    return self.confirmed_finger_number, True, frame
            else:
                self.ok_gesture_counter = 0
            
            # Finger counting and timing logic
            if not self.confirmation_mode and results.multi_hand_landmarks:
                # Smooth the finger count
                smoothed_fingers = self.smooth_detection(total_fingers)
                
                # Check if finger number has changed
                if smoothed_fingers != self.current_finger_number:
                    self.current_finger_number = smoothed_fingers
                    self.finger_number_start_time = current_time
                    logger.info(f"New finger count detected: {smoothed_fingers}")
                
                # Check if same number has been displayed for required duration
                if (self.finger_number_start_time and 
                    (current_time - self.finger_number_start_time) >= self.finger_display_duration):
                    
                    self.confirmation_mode = True
                    self.confirmed_finger_number = self.current_finger_number
                    self.detection_result.message = f"Number {self.confirmed_finger_number} detected for {self.finger_display_duration}s. Show OK gesture to confirm."
                    self.detection_result.status = "awaiting_confirmation"
                    logger.info(f"Number {self.confirmed_finger_number} stable for {self.finger_display_duration}s, entering confirmation mode")
                
                # Display current finger count and timing info
                self.last_stable_finger_count = smoothed_fingers
                cv2.putText(frame, f'Fingers: {smoothed_fingers}', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                
                # Show stability progress
                if self.finger_number_start_time:
                    elapsed = current_time - self.finger_number_start_time
                    remaining_time = max(0, self.finger_display_duration - elapsed)
                    progress_text = f'Stable for {elapsed:.1f}s / {self.finger_display_duration}s'
                    cv2.putText(frame, progress_text, (50, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    
                    if remaining_time > 0:
                        cv2.putText(frame, f'Hold for {remaining_time:.1f}s more', (50, 170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            elif self.confirmation_mode:
                # Display confirmation message
                cv2.putText(frame, f'Confirm {self.confirmed_finger_number} hours?', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.putText(frame, 'Show OK gesture to confirm', (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 3)
                
                # Auto-reset if no hands detected for too long in confirmation mode
                if not results.multi_hand_landmarks:
                    # Could add timeout logic here if needed
                    pass
            
            elif not results.multi_hand_landmarks:
                cv2.putText(frame, 'Show your hand(s)', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 3)
                # Reset timing if no hands detected
                self._reset_gesture_timing()
            
            # Limit fingers to 0-10
            total_fingers = min(10, max(0, total_fingers))
            
            return total_fingers, ok_detected, frame
            
        except Exception as e:
            logger.error(f"Hand gesture detection error: {e}")
            return 0, False, frame
    
    def _reset_gesture_timing(self):
        """Reset all gesture timing variables"""
        self.current_finger_number = 0
        self.finger_number_start_time = None
        self.confirmation_mode = False
        self.confirmed_finger_number = 0
        self.ok_gesture_counter = 0
        self.previous_counts = []
    
    def reset_detection(self):
        """Reset detection state including gesture timing"""
        self.detection_result = DetectionResult()
        self._reset_gesture_timing()
        logger.info("Detection state and gesture timing reset")
    
    def get_gesture_status(self) -> Dict:
        """Get current gesture detection status"""
        return {
            'current_finger_count': self.current_finger_number,
            'confirmed_finger_count': self.confirmed_finger_number,
            'confirmation_mode': self.confirmation_mode,
            'ok_gesture_counter': self.ok_gesture_counter,
            'time_remaining': max(0, self.finger_display_duration - 
                                (time.time() - self.finger_number_start_time)) 
                                if self.finger_number_start_time else 0,
            'last_stable_count': self.last_stable_finger_count
        }
    
    def get_status(self) -> Dict:
        """Get current detection status"""
        return {
            'vehicle_type': self.detection_result.vehicle_type,
            'license_plate': self.detection_result.license_plate,
            'parking_hours': self.detection_result.parking_hours,
            'confidence': self.detection_result.confidence,
            'status': self.detection_result.status,
            'message': self.detection_result.message,
            'current_phase': self.detection_result.current_phase
        }

# Global instance
detection_engine = DetectionEngine()
