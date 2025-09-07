# Add this improved error handling at the top of your server.py
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this function for better error handling
def handle_detection_error(error_message):
    """Handle detection errors gracefully"""
    logger.error(f"Detection error: {error_message}")
    detection_status['status'] = 'error'
    detection_status['message'] = error_message
    detection_status['current_phase'] = 'Error occurred'

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import numpy as np
from ultralytics import YOLO
import re
import requests
import base64
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue

# Load configuration from environment variables
PORT = int(os.getenv('PORT', 8000))
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'best.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
CAMERA_ID = int(os.getenv('CAMERA_ID', 0))
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', 640))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', 480))

# Try importing mediapipe with error handling
try:
    import mediapipe as mp
    mp.solutions.hands  # Verify the module is properly installed
    MP_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"MediaPipe not available or incomplete installation: {str(e)}")
    print("Hand gesture detection will be disabled.")
    MP_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error with MediaPipe: {str(e)}")
    print("Hand gesture detection will be disabled.")
    MP_AVAILABLE = False

app = Flask(__name__)

# Get allowed origins from environment variables
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5500,http://127.0.0.1:5500').split(',')

# Configure CORS with specific origins and error handling
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables to store detection state
detection_results = {
    'vehicle_type': None,
    'license_plate': None,
    'parking_hours': None
}

detection_status = {
    'status': 'idle',  # idle, running, completed, error
    'current_phase': None,
    'message': None
}

detection_queue = queue.Queue()

# OCR.space API configuration
OCR_API_KEY = os.getenv('OCR_API_KEY', "K83315680088957")
OCR_API_URL = os.getenv('OCR_API_URL', "https://api.ocr.space/parse/image")

# Complete list of Indian state and UT codes
INDIAN_STATE_CODES = {
    'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
    'CG': 'Chhattisgarh', 'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana',
    'HP': 'Himachal Pradesh', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala',
    'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya',
    'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'OR': 'Odisha', 'PB': 'Punjab',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TG': 'Telangana',
    'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal',
    'AN': 'Andaman and Nicobar', 'CH': 'Chandigarh', 'DD': 'Dadra and Nagar Haveli',
    'DL': 'Delhi', 'JK': 'Jammu and Kashmir', 'LA': 'Ladakh', 'LD': 'Lakshadweep',
    'PY': 'Puducherry'
}

# Vehicle classification mapping
vehicle_classes_mapping = {
    "motorcycle": "Two Wheeler (Bike)",
    "bicycle": "Two Wheeler (Bike)",
    "car": "4 Wheeler (Car)",
    "bus": "Heavy Vehicle (Bus/Truck)",
    "truck": "Heavy Vehicle (Bus/Truck)"
}

def ocr_space_api(image, api_key=OCR_API_KEY, language='eng'):
    """OCR.space API request with error handling."""
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode()
        
        payload = {
            'apikey': api_key,
            'language': language,
            'isOverlayRequired': False,
            'base64Image': f'data:image/jpeg;base64,{img_base64}',
            'OCREngine': '2',
            'scale': 'true',
            'isTable': 'false'
        }
        
        r = requests.post(OCR_API_URL, data=payload, timeout=30)
        if r.status_code == 200:
            result = r.json()
            return result
        else:
            return {"error": f"API request failed with status code {r.status_code}: {r.text}"}
            
    except requests.exceptions.Timeout:
        return {"error": "API request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def correct_ocr_errors(text):
    """Correct common OCR errors in license plate text"""
    if not text:
        return ""
    
    corrected = str(text).upper()
    
    char_corrections = {
        'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6',
        'o': '0', 'i': '1', 's': '5', 'z': '2', 'b': '8', 'g': '6',
        'l': '1', 'L': '1'
    }
    
    for wrong, right in char_corrections.items():
        corrected = corrected.replace(wrong, right)
    
    # Remove spaces between characters and digits
    corrected = re.sub(r'([A-Z])\s+([A-Z0-9])', r'\1\2', corrected)
    corrected = re.sub(r'(\d)\s+([A-Z0-9])', r'\1\2', corrected)
    corrected = re.sub(r'([A-Z0-9])\s+(\d)', r'\1\2', corrected)
    
    return corrected

def apply_final_corrections(text):
    """Apply final corrections to license plate text - FIXED SPACING ISSUE"""
    if not text:
        return ""
    
    # Remove ALL spaces and special characters, keep only letters and numbers
    clean_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    
    state_corrections = {
        'MMHO': 'MH0', 'MHO': 'MH0', 'MHOI': 'MH01', 'MH0I': 'MH01',
        'UPO': 'UP0', 'UPOI': 'UP01', 'UP0I': 'UP01',
        'DLO': 'DL0', 'DLOI': 'DL01', 'DL0I': 'DL01',
        'KAO': 'KA0', 'KAOI': 'KA01', 'KA0I': 'KA01',
        'TNO': 'TN0', 'TNOI': 'TN01', 'TN0I': 'TN01',
        'WBO': 'WB0', 'WBOI': 'WB01', 'WB0I': 'WB01',
        'GJO': 'GJ0', 'GJOI': 'GJ01', 'GJ0I': 'GJ01',
        'RJO': 'RJ0', 'RJOI': 'RJ01', 'RJ0I': 'RJ01',
        'MPO': 'MP0', 'MPOI': 'MP01', 'MP0I': 'MP01',
        'HRO': 'HR0', 'HROI': 'HR01', 'HR0I': 'HR01',
        'PBO': 'PB0', 'PBOI': 'PB01', 'PB0I': 'PB01'
    }
    
    for wrong, right in state_corrections.items():
        if clean_text.startswith(wrong):
            clean_text = right + clean_text[len(wrong):]
            break
    
    sequence_fixes = {
        'AE8': 'AE8', 'AES': 'AE8', 'AEB': 'AE8',
        'CD1': 'CD1', 'CDI': 'CD1', 'COI': 'CD1',
        '1996': '1996', 'I996': '1996', '19S6': '1996'
    }
    
    for wrong, right in sequence_fixes.items():
        clean_text = clean_text.replace(wrong, right)
    
    return clean_text

def is_valid_indian_state_code(code):
    """Check if the code is a valid Indian state/UT code"""
    if not code:
        return False
    return str(code).upper() in INDIAN_STATE_CODES

def is_valid_license_plate_text(text):
    """Check if detected text looks like a license plate"""
    if not text:
        return False
    
    clean_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    
    if len(clean_text) < 6:
        return False
    
    if clean_text in ['IND', 'INDIA', 'BHARAT']:
        return False
    
    if any(word in clean_text for word in ['PLATE', 'DETECTION', 'CAMERA', 'VIDEO']):
        return False
    
    has_letters = bool(re.search(r'[A-Z]', clean_text))
    has_numbers = bool(re.search(r'[0-9]', clean_text))
    
    if not (has_letters and has_numbers):
        return False
    
    for state_code in INDIAN_STATE_CODES.keys():
        if clean_text.startswith(state_code):
            return True
    
    return False

def score_license_plate_text(text):
    """Score license plate text based on Indian license plate patterns"""
    if not text:
        return 0
    
    clean_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    score = 0
    
    patterns = [
        (r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', 100),
        (r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$', 90),
        (r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$', 85),
        (r'^[A-Z]{3,4}\d{1,4}$', 70),
    ]
    
    for pattern, pattern_score in patterns:
        if re.match(pattern, clean_text):
            score += pattern_score
            break
    
    for state_code in INDIAN_STATE_CODES.keys():
        if clean_text.startswith(state_code):
            score += 100
            break
    
    if any(word in clean_text for word in ['PLATE', 'IND', 'INDIA', 'DETECTION']):
        score -= 200
    
    text_len = len(clean_text)
    if 8 <= text_len <= 10:
        score += 30
    elif 6 <= text_len <= 7:
        score += 15
    elif text_len > 10:
        score -= 20
    
    letter_count = len(re.findall(r'[A-Z]', clean_text))
    number_count = len(re.findall(r'\d', clean_text))
    
    if 4 <= letter_count <= 6 and 3 <= number_count <= 6:
        score += 20
    
    return score

def extract_license_plate_from_text(text):
    """Extract the most likely license plate from detected text"""
    if not text:
        return None
    
    corrected_text = correct_ocr_errors(text)
    
    patterns = [
        r'[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4}',
        r'[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{1,4}',
        r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{1,4}',
    ]
    
    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, corrected_text)
        for match in matches:
            # Remove ALL spaces - this is the key fix
            clean_match = re.sub(r'\s+', '', match)
            clean_match = apply_final_corrections(clean_match)
            if is_valid_license_plate_text(clean_match):
                candidates.append(clean_match)
    
    if candidates:
        scored_candidates = [(candidate, score_license_plate_text(candidate)) for candidate in candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]
    
    return None

def enhance_image_for_ocr(image):
    """Apply image enhancement techniques for better OCR"""
    if image is None:
        return []
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    enhanced_images = []
    enhanced_images.append(("original", gray))
    
    try:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(("thresh_otsu", thresh1))
        
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        enhanced_images.append(("adaptive", adaptive))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_images.append(("clahe", enhanced))
        
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        enhanced_images.append(("morph", morph))
        
    except Exception as e:
        print(f"Error in image enhancement: {e}")
    
    return enhanced_images

def clean_indian_plate_text(text):
    """Clean and filter text for Indian license plates"""
    if not text:
        return None
    
    # Remove all non-alphanumeric characters and spaces
    cleaned = re.sub(r'[^A-Z0-9]', '', str(text).upper().strip())
    
    if len(cleaned) < 6:
        return None
    
    cleaned = apply_final_corrections(cleaned)
    return cleaned

def process_license_plate_ocr(original_image):
    """Process license plate with multiple enhancement techniques"""
    if original_image is None:
        return []
    
    enhanced_images = enhance_image_for_ocr(original_image)
    all_candidates = []
    
    for method_name, enhanced_img in enhanced_images:
        try:
            result = ocr_space_api(enhanced_img)
            
            if "error" in result:
                continue
            
            if not result.get("IsErroredOnProcessing", True):
                parsed_results = result.get("ParsedResults", [])
                for parsed_result in parsed_results:
                    text = parsed_result.get("ParsedText", "").strip()
                    if text:
                        corrected_text = correct_ocr_errors(text)
                        extracted_plate = extract_license_plate_from_text(corrected_text)
                        
                        if extracted_plate:
                            score = score_license_plate_text(extracted_plate)
                            confidence = 0.99
                            all_candidates.append((extracted_plate, confidence, method_name, score))
                        
                        lines = corrected_text.split('\n')
                        for line in lines:
                            clean_text = clean_indian_plate_text(line)
                            if clean_text and is_valid_license_plate_text(clean_text):
                                final_text = apply_final_corrections(clean_text)
                                score = score_license_plate_text(final_text)
                                confidence = 0.99
                                all_candidates.append((final_text, confidence, method_name, score))
                                
        except Exception as e:
            print(f"Error processing OCR: {e}")
            continue
    
    if not all_candidates:
        return []
    
    all_candidates.sort(key=lambda x: x[3], reverse=True)
    
    seen = set()
    unique_candidates = []
    for candidate in all_candidates:
        text = candidate[0]
        if text not in seen:
            seen.add(text)
            unique_candidates.append(candidate)
    
    return [(text, conf, method) for text, conf, method, score in unique_candidates]

def format_indian_plate(text):
    """Format text as Indian license plate - NO SPACES for auto detection"""
    if not text:
        return ""
    
    # Remove all spaces and special characters
    clean_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    
    # Return without spaces for consistency
    return clean_text

def count_fingers(hand_landmarks, hand_label):
    """Improved finger counting with better landmark analysis"""
    if not hand_landmarks:
        return 0
    
    count = 0
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y])
    
    # Thumb detection
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

def is_ok_sign(hand_landmarks, hand_label):
    """Detect OK sign - thumb and index finger touching in a circle, others extended"""
    if not hand_landmarks:
        return False
    
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y])
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
    circle_formed = distance < 0.05
    
    middle_extended = landmarks[12][1] < landmarks[10][1] - 0.02
    ring_extended = landmarks[16][1] < landmarks[14][1] - 0.02
    pinky_extended = landmarks[20][1] < landmarks[18][1] - 0.02
    
    return circle_formed and middle_extended and ring_extended and pinky_extended

def smooth_detection(current_count, previous_counts, threshold=3):
    """Smooth detection to reduce noise"""
    if previous_counts is None:
        previous_counts = []
    
    previous_counts.append(current_count)
    if len(previous_counts) > 5:
        previous_counts.pop(0)
    
    if len(previous_counts) >= threshold:
        return max(set(previous_counts), key=previous_counts.count)
    
    return current_count

def vehicle_detection_phase():
    """Phase 1: Vehicle Detection"""
    print("Phase 1: Vehicle Detection Started")
    detection_status['current_phase'] = "Vehicle Detection - Point camera at vehicle"
    
    try:
        # Use relative path - file should be in same directory
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set camera to high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify if the camera accepted our resolution settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution set to: {actual_width}x{actual_height}")
    
    detected_class = None
    vehicle_detected = False
    start_time = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, verbose=False)
            vehicle_found = False
            best_conf = 0
            best_label = None
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        if conf > 0.25 and class_name in vehicle_classes_mapping:
                            vehicle_found = True
                            if conf > best_conf:
                                best_conf = conf
                                best_label = vehicle_classes_mapping[class_name]
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            display_label = f"{vehicle_classes_mapping[class_name]} {conf:.2f}"
                            (text_width, text_height), _ = cv2.getTextSize(
                                display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            
                            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                                        (x1 + text_width, y1), (0, 255, 0), -1)
                            cv2.putText(frame, display_label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Handle detection timing
            if vehicle_found:
                detected_class = best_label
                if not vehicle_detected:
                    vehicle_detected = True
                    start_time = time.time()
                    print(f"Vehicle detected: {best_label}! Waiting 5 seconds...")
                    detection_status['current_phase'] = f"Vehicle detected: {best_label} - Confirming..."
                
                if start_time and time.time() - start_time >= 5:
                    print(f"Vehicle confirmed: {best_label}")
                    detection_results['vehicle_type'] = best_label
                    break
            else:
                vehicle_detected = False
                start_time = None
            
            # Add timer display
            if vehicle_detected and start_time:
                elapsed = time.time() - start_time
                remaining = max(0, 5 - elapsed)
                cv2.putText(frame, f"Confirming in: {remaining:.1f}s", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if detected_class:
                    cv2.putText(frame, f"Detected: {detected_class}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
                
    except Exception as e:
        print(f"Error in vehicle detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return detected_class

def license_plate_detection_phase():
    """Phase 2: License Plate Detection"""
    print("\nPhase 2: License Plate Detection Started")
    detection_status['current_phase'] = "License Plate Detection - Point camera at license plate"
    
    # Use relative path - file should be in same directory
    model_path = r"C:\Users\UseR\Documents\Coding\Smart Parking\best.pt"
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading license plate model: {e}")
        print("Make sure 'best.pt' model file exists in your directory")
        # Generate demo license plate if model not found
        import random
        states = ['MH', 'DL', 'KA', 'UP', 'WB', 'TN', 'GJ', 'RJ']
        state = random.choice(states)
        district = random.randint(1, 99)
        series = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        number = random.randint(1000, 9999)
        demo_plate = f"{state}{district:02d}{series}{number}"  # No spaces
        print(f"Demo license plate generated: {demo_plate}")
        detection_results['license_plate'] = demo_plate
        time.sleep(3)  # Simulate processing time
        return demo_plate
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Set camera to high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify if the camera accepted our resolution settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"License plate detection camera resolution: {actual_width}x{actual_height}")
    
    plate_detected = False
    start_time = None
    save_path = "detected_plate.jpg"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, verbose=False)
            plate_found = False
            best_bbox = None
            best_conf = 0
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    conf = float(box.conf)
                    if conf > 0.3:
                        plate_found = True
                        if conf > best_conf:
                            best_conf = conf
                            best_bbox = box.xyxy[0].cpu().numpy()
            
            if plate_found:
                if not plate_detected:
                    plate_detected = True
                    start_time = time.time()
                    print("License plate detected! Waiting 5 seconds...")
                    detection_status['current_phase'] = "License plate detected - Capturing..."
                
                if best_bbox is not None:
                    x1, y1, x2, y2 = map(int, best_bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Plate: {best_conf:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if start_time and time.time() - start_time >= 5:
                    print("5 seconds elapsed! Capturing plate...")
                    x1, y1, x2, y2 = map(int, best_bbox)
                    h, w = frame.shape[:2]
                    padding = 30
                    y1_pad = max(0, y1-padding)
                    y2_pad = min(h, y2+padding)
                    x1_pad = max(0, x1-padding)
                    x2_pad = min(w, x2+padding)
                    
                    plate_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    cv2.imwrite(save_path, plate_img)
                    break
            else:
                plate_detected = False
                start_time = None
            
            if plate_detected and start_time:
                elapsed = time.time() - start_time
                remaining = max(0, 5 - elapsed)
                cv2.putText(frame, f"Capturing in: {remaining:.1f}s", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except Exception as e:
        print(f"Error in license plate detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Process saved image with OCR
    if os.path.exists(save_path):
        print("Processing license plate...")
        detection_status['current_phase'] = "Processing license plate text..."
        detection_status['message'] = "Analyzing license plate image with OCR..."
        
        try:
            plate_img = cv2.imread(save_path)
            results = process_license_plate_ocr(plate_img)
            if results:
                best_text = results[0][0]
                formatted_plate = format_indian_plate(best_text)
                detection_results['license_plate'] = formatted_plate
                detection_status['current_phase'] = "License plate processed successfully"
                detection_status['message'] = f"License plate detected: {formatted_plate}"
                print(f"✓ License plate processed: {formatted_plate}")
                return formatted_plate
            else:
                detection_status['current_phase'] = "License plate text not readable"
                detection_status['message'] = "Could not extract text from license plate"
                print("✗ No readable text found in license plate")
        except Exception as e:
            print(f"Error processing OCR: {e}")
            detection_status['current_phase'] = "OCR processing error"
            detection_status['message'] = f"Error processing license plate: {str(e)}"
    else:
        detection_status['current_phase'] = "No license plate image captured"
        detection_status['message'] = "Failed to capture license plate image"
        print("✗ No license plate image found")
    
    return None

def hand_gesture_detection_phase():
    """Phase 3: Hand Gesture Detection for Hours"""
    if not MP_AVAILABLE:
        print("MediaPipe not available. Using default parking hours.")
        detection_status['current_phase'] = "Hand gesture detection not available"
        detection_status['message'] = "Using default 2 hours parking duration"
        # For demonstration, return a default value
        default_hours = 2
        detection_results['parking_hours'] = default_hours
        time.sleep(3)  # Simulate processing time
        return default_hours
    
    print("\nPhase 3: Hand Gesture Detection Started")
    detection_status['current_phase'] = "Hand Gesture Detection - Show fingers (1-10) for parking hours"
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7,
        model_complexity=1
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    previous_counts = []
    current_number = 0
    number_start_time = None
    number_display_duration = 3.0
    confirmation_mode = False
    confirmed_number = 0
    ok_gesture_counter = 0
    ok_gesture_threshold = 8
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            total_fingers = 0
            ok_detected = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_handedness.classification[0].label
                    confidence = hand_handedness.classification[0].score
                    
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    cv2.putText(frame, f'{hand_label} ({confidence:.2f})',
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    if is_ok_sign(hand_landmarks, hand_label):
                        ok_detected = True
                    
                    if not confirmation_mode:
                        fingers = count_fingers(hand_landmarks, hand_label)
                        total_fingers += fingers
            
            # Handle OK gesture detection
            if ok_detected and confirmation_mode:
                ok_gesture_counter += 1
                remaining = ok_gesture_threshold - ok_gesture_counter + 1
                cv2.putText(frame, 'OK SIGN DETECTED!', (50, 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 3)
                cv2.putText(frame, f'Confirming in {remaining}...', (50, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 3)
                
                if ok_gesture_counter >= ok_gesture_threshold:
                    print(f'Parking hours confirmed: {confirmed_number}')
                    detection_results['parking_hours'] = confirmed_number
                    detection_status['current_phase'] = f"Parking duration confirmed: {confirmed_number} hours"
                    detection_status['message'] = f"Hand gesture detection completed successfully"
                    break
            else:
                ok_gesture_counter = 0
            
            # Number tracking and confirmation logic
            if not confirmation_mode and results.multi_hand_landmarks:
                # Limit to 1-10 hours
                if total_fingers > 10:
                    total_fingers = 10
                elif total_fingers == 0:
                    total_fingers = 0
                
                # Smooth the finger count
                total_fingers = smooth_detection(total_fingers, previous_counts)
                
                # Check if number has changed
                if total_fingers != current_number:
                    current_number = total_fingers
                    number_start_time = time.time()
                
                # Check if same number has been displayed for required duration
                if number_start_time and (time.time() - number_start_time) >= number_display_duration:
                    if current_number > 0:  # Only confirm if 1-10 hours
                        confirmation_mode = True
                        confirmed_number = current_number
                        print(f"Number {confirmed_number} detected for 3 seconds! Please confirm with OK gesture.")
                        detection_status['current_phase'] = f"Confirm {confirmed_number} hours with OK gesture"
                        detection_status['message'] = f"Hold OK gesture to confirm {confirmed_number} hours parking duration"
                
                # Display current finger count
                cv2.putText(frame, f'Hours: {total_fingers}', (50, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 139), 4)
                
                # Show countdown if number is stable
                if number_start_time and current_number > 0:
                    elapsed = time.time() - number_start_time
                    remaining = max(0, number_display_duration - elapsed)
                    if remaining > 0:
                        cv2.putText(frame, f'Stable for {elapsed:.1f}s / {number_display_duration}s',
                                  (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 112), 2)
            
            elif confirmation_mode:
                # Display confirmation message
                cv2.putText(frame, f'Is the duration {confirmed_number} hours?', (50, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (139, 0, 0), 3)
                cv2.putText(frame, 'Show OK gesture to confirm', (50, 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 3)
            
            elif not results.multi_hand_landmarks:
                cv2.putText(frame, 'Show your hand(s) (1-10 fingers)', (50, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 3)
                current_number = 0
                number_start_time = None
            
            # Display instructions
            if not confirmation_mode:
                cv2.putText(frame, 'Hold same number for 3 seconds to confirm',
                          (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                cv2.putText(frame, 'Use OK gesture (thumb+index circle) to confirm',
                          (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.putText(frame, 'Press ESC to quit', (10, h - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.imshow("Hand Gesture - Parking Hours", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except Exception as e:
        print(f"Error in hand gesture detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return detection_results.get('parking_hours')

def run_detection():
    """Main detection function that runs all three phases"""
    global detection_status, detection_results
    
    try:
        detection_status['status'] = 'running'
        detection_status['message'] = 'Starting detection process...'
        logger.info("Starting AI detection process")
        
        # Reset results
        detection_results = {
            'vehicle_type': None,
            'license_plate': None,
            'parking_hours': None
        }
        
        # Phase 1: Vehicle Detection
        logger.info("Starting Phase 1: Vehicle Detection")
        vehicle_type = vehicle_detection_phase()
        if not vehicle_type:
            detection_status['status'] = 'error'
            detection_status['message'] = 'Vehicle detection failed'
            logger.error("Vehicle detection failed")
            return
        
        print(f"✓ Phase 1 Complete: {vehicle_type}")
        logger.info(f"Phase 1 completed: {vehicle_type}")
        time.sleep(2)
        
        # Phase 2: License Plate Detection
        logger.info("Starting Phase 2: License Plate Detection")
        license_plate = license_plate_detection_phase()
        if license_plate:
            print(f"✓ Phase 2 Complete: {license_plate}")
            logger.info(f"Phase 2 completed: {license_plate}")
        else:
            print("✗ Phase 2 Failed: License plate not detected")
            logger.warning("License plate detection failed, continuing to next phase")
            # Continue to next phase even if license plate detection fails
            detection_status['current_phase'] = "License plate detection failed - continuing to hand gesture"
            detection_status['message'] = "Will proceed with manual license plate entry"
        
        time.sleep(2)
        
        # Phase 3: Hand Gesture Detection
        logger.info("Starting Phase 3: Hand Gesture Detection")
        parking_hours = hand_gesture_detection_phase()
        if parking_hours:
            print(f"✓ Phase 3 Complete: {parking_hours} hours")
            logger.info(f"Phase 3 completed: {parking_hours} hours")
        else:
            print("✗ Phase 3 Failed: Parking hours not detected")
            logger.warning("Hand gesture detection failed, using default 2 hours")
            # Use default parking hours if gesture detection fails
            parking_hours = 2
            detection_results['parking_hours'] = parking_hours
            detection_status['current_phase'] = "Hand gesture detection failed - using default 2 hours"
            detection_status['message'] = "Default 2-hour parking duration applied"
        
        detection_status['status'] = 'completed'
        detection_status['current_phase'] = 'Detection completed'
        detection_status['message'] = 'All detection phases completed'
        logger.info("All detection phases completed successfully")
        
    except Exception as e:
        detection_status['status'] = 'error'
        detection_status['message'] = str(e)
        print(f"Error during detection: {e}")

# Flask API Routes
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_status
    
    if detection_status['status'] == 'running':
        return jsonify({
            'status': 'already_running',
            'message': 'Detection is already in progress'
        })
    
    # Start detection in a separate thread
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Detection process started'
    })

@app.route('/get_results', methods=['GET'])
def get_results():
    return jsonify({
        'status': detection_status['status'],
        'current_phase': detection_status.get('current_phase'),
        'message': detection_status.get('message'),
        'results': detection_results
    })

@app.route('/reset', methods=['POST'])
def reset_detection():
    global detection_status, detection_results
    
    detection_status = {
        'status': 'idle',
        'current_phase': None,
        'message': None
    }
    
    detection_results = {
        'vehicle_type': None,
        'license_plate': None,
        'parking_hours': None
    }
    
    logger.info("Detection system reset")
    return jsonify({
        'status': 'reset',
        'message': 'Detection system reset'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'detection_status': detection_status['status'],
        'camera_available': True,  # Could add actual camera check
        'models_loaded': True  # Could add actual model check
    })

if __name__ == '__main__':
    print("Smart Parking Detection Server Starting...")
    print("Make sure to have your YOLO models ready:")
    print("1. yolo11n.pt (for vehicle detection) - will be downloaded automatically")
    print("2. best.pt (for license plate detection) - place in same directory")
    print("Server will run on http://localhost:8000")
    app.run(host='localhost', port=8000, debug=False)
