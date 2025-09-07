import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import json
import uuid
from typing import Dict, List, Optional
import base64
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import time as time_module

# Import backend modules
from parking_manager import parking_manager, SlotStatus
from detection_engine import detection_engine

# Page configuration
st.set_page_config(
    page_title="Vehicle Vacancy Vault - Smart Parking System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global state for video processing (shared between threads)
class GlobalState:
    def __init__(self):
        self.auto_mode_active = False
        self.detection_phase = 'idle'
        self.auto_detection_results = {}
        self.plate_captured = False
        self.gesture_confirmed = False
        self.current_finger_count = 0
        self.ok_gesture_count = 0

global_state = GlobalState()

# Custom CSS (keeping exactly the same as provided)
def load_css():
    st.markdown("""
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #16213e;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --accent-primary: #6366f1;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-orange: #f59e0b;
        --accent-blue: #3b82f6;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #6366f1);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 0;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Parking grid styling */
    .parking-slot {
        aspect-ratio: 1;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin: 4px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        color: white;
        text-align: center;
        padding: 8px;
        height: 120px;
    }
    
    .slot-available {
        background: linear-gradient(135deg, #10b981, #059669);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .slot-occupied {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .slot-reserved {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .slot-number {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .slot-info {
        font-size: 0.75rem;
        opacity: 0.9;
    }
    
    /* Pricing card styling */
    .pricing-card {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6), rgba(22, 33, 62, 0.6));
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .pricing-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    .pricing-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .pricing-title {
        color: #6366f1;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .pricing-amount {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .pricing-note {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Legend styling */
    .legend {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 6px;
    }
    
    .legend-available { background: linear-gradient(135deg, #10b981, #059669); }
    .legend-occupied { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .legend-reserved { background: linear-gradient(135deg, #f59e0b, #d97706); }
    
    /* Auto Mode specific styles */
    .auto-mode-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .detection-phase {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .phase-completed { border-left-color: #10b981; }
    .phase-active { border-left-color: #f59e0b; }
    .phase-pending { border-left-color: rgba(255, 255, 255, 0.3); }
    
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'auto_mode_active' not in st.session_state:
        st.session_state.auto_mode_active = False
    if 'detection_phase' not in st.session_state:
        st.session_state.detection_phase = 'idle'
    if 'auto_detection_results' not in st.session_state:
        st.session_state.auto_detection_results = {}
    if 'vehicle_confirmed_time' not in st.session_state:
        st.session_state.vehicle_confirmed_time = None
    if 'plate_captured' not in st.session_state:
        st.session_state.plate_captured = False
    if 'gesture_confirmed' not in st.session_state:
        st.session_state.gesture_confirmed = False
    if 'finger_count_stable_time' not in st.session_state:
        st.session_state.finger_count_stable_time = None
    if 'current_finger_count' not in st.session_state:
        st.session_state.current_finger_count = 0
    if 'ok_gesture_count' not in st.session_state:
        st.session_state.ok_gesture_count = 0

# Auto Detection Video Processor (Fixed for new API)
class AutoDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detection_start_time = None
        self.last_vehicle_type = None
        self.last_finger_count = 0
        self.finger_count_history = []
        self.ok_gesture_counter = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Use global state instead of st.session_state
        if not global_state.auto_mode_active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        current_time = time_module.time()
        
        # Phase 1: Vehicle Detection
        if global_state.detection_phase == 'vehicle_detection':
            vehicle_detected, vehicle_type, confidence = detection_engine.detect_vehicle_in_frame(img)
            
            if vehicle_detected and confidence > 0.5:
                if self.last_vehicle_type != vehicle_type:
                    self.detection_start_time = current_time
                    self.last_vehicle_type = vehicle_type
                
                elapsed = current_time - self.detection_start_time if self.detection_start_time else 0
                remaining = max(0, 3 - elapsed)
                
                cv2.putText(img, f'Vehicle: {vehicle_type} ({confidence:.2f})', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f'Confirming in: {remaining:.1f}s', 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if elapsed >= 3:
                    global_state.auto_detection_results['vehicle_type'] = vehicle_type
                    global_state.detection_phase = 'license_plate_detection'
                    detection_engine.detection_result.current_phase = "License Plate Detection"
                    self.detection_start_time = None
            else:
                cv2.putText(img, 'Point camera at vehicle', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                self.detection_start_time = None
                self.last_vehicle_type = None
        
        # Phase 2: License Plate Detection  
        elif global_state.detection_phase == 'license_plate_detection':
            plate_detected, plate_roi, confidence = detection_engine.detect_license_plate_in_frame(img)
            
            if plate_detected and confidence > 0.4:
                if not self.detection_start_time:
                    self.detection_start_time = current_time
                
                elapsed = current_time - self.detection_start_time
                remaining = max(0, 3 - elapsed)
                
                cv2.putText(img, f'License plate detected ({confidence:.2f})', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, f'Capturing in: {remaining:.1f}s', 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if elapsed >= 3 and not global_state.plate_captured:
                    # Process license plate
                    license_text = detection_engine.process_license_plate_ocr(plate_roi)
                    if license_text:
                        global_state.auto_detection_results['license_plate'] = license_text
                        detection_engine.detection_result.current_phase = "Hand Gesture Detection"
                        global_state.detection_phase = 'gesture_detection'
                        global_state.plate_captured = True
                    self.detection_start_time = None
            else:
                cv2.putText(img, 'Point camera at license plate', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                self.detection_start_time = None
        
        # Phase 3: Hand Gesture Detection
        elif global_state.detection_phase == 'gesture_detection':
            finger_count, ok_detected, processed_img = detection_engine.detect_hand_gesture_in_frame(img)
            img = processed_img
            
            if not global_state.gesture_confirmed:
                if ok_detected:
                    self.ok_gesture_counter += 1
                    cv2.putText(img, f'OK detected! Confirming... ({self.ok_gesture_counter}/10)', 
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if self.ok_gesture_counter >= 10:
                        confirmed_hours = global_state.current_finger_count
                        if confirmed_hours > 0:
                            global_state.auto_detection_results['parking_hours'] = confirmed_hours
                            global_state.detection_phase = 'completed'
                            global_state.gesture_confirmed = True
                            detection_engine.detection_result.current_phase = "Detection Completed"
                else:
                    self.ok_gesture_counter = 0
                    
                    # Track finger count stability
                    if finger_count > 0:
                        self.finger_count_history.append(finger_count)
                        if len(self.finger_count_history) > 30:  # ~1 second at 30fps
                            self.finger_count_history.pop(0)
                        
                        # Check if finger count is stable
                        if len(self.finger_count_history) >= 30:
                            most_common = max(set(self.finger_count_history), 
                                            key=self.finger_count_history.count)
                            stability = self.finger_count_history.count(most_common) / len(self.finger_count_history)
                            
                            if stability > 0.8:  # 80% stability
                                global_state.current_finger_count = most_common
                                cv2.putText(img, f'Detected {most_common} hours - Show OK to confirm', 
                                          (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            else:
                                cv2.putText(img, 'Hold steady finger count for 1 second', 
                                          (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    else:
                        self.finger_count_history = []
                        cv2.putText(img, 'Show 1-10 fingers for parking hours', 
                                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Phase 4: Completed
        elif global_state.detection_phase == 'completed':
            cv2.putText(img, 'Detection Complete! Check results below.', 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Holiday data for 2025
HOLIDAYS_2025 = [
    {'date': '2025-01-01', 'holiday': 'New Year\'s Day', 'hours': '12:00 AM - 11:59 PM'},
    {'date': '2025-01-26', 'holiday': 'Republic Day', 'hours': '6:00 AM - 11:00 PM'},
    {'date': '2025-03-13', 'holiday': 'Holi', 'hours': '10:00 AM - 12:00 AM'},
    {'date': '2025-03-31', 'holiday': 'Eid ul-Fitr', 'hours': '6:00 AM - 11:00 PM'},
    {'date': '2025-04-14', 'holiday': 'Ram Navami', 'hours': '6:00 AM - 10:00 PM'},
    {'date': '2025-04-18', 'holiday': 'Good Friday', 'hours': '9:00 AM - 9:00 PM'},
    {'date': '2025-08-15', 'holiday': 'Independence Day', 'hours': '6:00 AM - 11:00 PM'},
    {'date': '2025-08-27', 'holiday': 'Janmashtami', 'hours': '8:00 AM - 12:00 AM'},
    {'date': '2025-10-02', 'holiday': 'Gandhi Jayanti', 'hours': '6:00 AM - 10:00 PM'},
    {'date': '2025-10-22', 'holiday': 'Dussehra', 'hours': '10:00 AM - 12:00 AM'},
    {'date': '2025-11-12', 'holiday': 'Diwali', 'hours': '6:00 PM - 2:00 AM'},
    {'date': '2025-12-25', 'holiday': 'Christmas Day', 'hours': '10:00 AM - 11:00 PM'}
]

def sync_global_state():
    """Sync global state with session state"""
    global_state.auto_mode_active = st.session_state.get('auto_mode_active', False)
    global_state.detection_phase = st.session_state.get('detection_phase', 'idle')
    global_state.auto_detection_results = st.session_state.get('auto_detection_results', {})
    global_state.plate_captured = st.session_state.get('plate_captured', False)
    global_state.gesture_confirmed = st.session_state.get('gesture_confirmed', False)
    global_state.current_finger_count = st.session_state.get('current_finger_count', 0)

def sync_session_state():
    """Sync session state with global state"""
    st.session_state.auto_mode_active = global_state.auto_mode_active
    st.session_state.detection_phase = global_state.detection_phase
    st.session_state.auto_detection_results = global_state.auto_detection_results
    st.session_state.plate_captured = global_state.plate_captured
    st.session_state.gesture_confirmed = global_state.gesture_confirmed
    st.session_state.current_finger_count = global_state.current_finger_count

def render_parking_grid():
    """Render the parking grid visualization"""
    cols = st.columns(4)
    
    for i in range(20):
        slot_id = f"slot_{i+1}"
        slot_data = parking_manager.get_slot_data(slot_id)
        col_idx = i % 4
        
        with cols[col_idx]:
            status = slot_data.status.value
            slot_number = i + 1
            
            if status == 'available':
                css_class = 'slot-available'
                info_text = 'Available'
            elif status == 'occupied':
                css_class = 'slot-occupied'
                vehicle_info = f"{slot_data.vehicle_type}"
                info_text = f"{vehicle_info}<br>{slot_data.vehicle_number}"
            else:  # reserved
                css_class = 'slot-reserved'
                info_text = f"Reserved<br>{slot_data.vehicle_number}"
            
            st.markdown(f"""
            <div class="parking-slot {css_class}">
                <div class="slot-number">{slot_number}</div>
                <div class="slot-info">{info_text}</div>
            </div>
            """, unsafe_allow_html=True)

def render_legend():
    """Render the parking status legend"""
    st.markdown("""
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color legend-available"></div>
            <span>Available</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-occupied"></div>
            <span>Occupied</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-reserved"></div>
            <span>Reserved</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_pricing_info():
    """Render pricing information cards"""
    cols = st.columns(4)
    
    pricing_data = [
        {"title": "Bikes", "amount": "₹200/hour", "note": "+₹50 rush hour surcharge"},
        {"title": "Cars", "amount": "₹150/hour", "note": "+₹30 rush hour surcharge"},
        {"title": "Trucks", "amount": "₹300/hour", "note": "+₹70 rush hour surcharge"},
        {"title": "Night Hours", "amount": "₹100/hour", "note": "11 PM - 5 AM (All vehicles)"}
    ]
    
    for i, data in enumerate(pricing_data):
        with cols[i]:
            st.markdown(f"""
            <div class="pricing-card">
                <div class="pricing-title">{data['title']}</div>
                <div class="pricing-amount">{data['amount']}</div>
                <div class="pricing-note">{data['note']}</div>
            </div>
            """, unsafe_allow_html=True)

def render_metrics():
    """Render parking statistics metrics"""
    stats = parking_manager.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['total_slots']}</div>
            <div class="metric-label">Total Slots</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['occupied_count']}</div>
            <div class="metric-label">Occupied</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['available_count']}</div>
            <div class="metric-label">Available</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['reserved_count']}</div>
            <div class="metric-label">Reserved</div>
        </div>
        """, unsafe_allow_html=True)

def render_auto_mode():
    """Render Auto Mode interface"""
    st.markdown("### 🤖 AI Auto Detection Mode")
    
    # Sync states
    sync_global_state()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if not st.session_state.auto_mode_active:
            if st.button("🚀 Start Auto Detection", type="primary", use_container_width=True):
                st.session_state.auto_mode_active = True
                st.session_state.detection_phase = 'vehicle_detection'
                st.session_state.auto_detection_results = {}
                st.session_state.plate_captured = False
                st.session_state.gesture_confirmed = False
                detection_engine.reset_detection()
                sync_global_state()
                st.rerun()
        else:
            if st.button("⏹️ Stop Detection", type="secondary", use_container_width=True):
                st.session_state.auto_mode_active = False
                st.session_state.detection_phase = 'idle'
                detection_engine.reset_detection()
                sync_global_state()
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset Detection", use_container_width=True):
            st.session_state.auto_mode_active = False
            st.session_state.detection_phase = 'idle'
            st.session_state.auto_detection_results = {}
            st.session_state.plate_captured = False
            st.session_state.gesture_confirmed = False
            detection_engine.reset_detection()
            sync_global_state()
            st.rerun()
    
    # Detection Status
    if st.session_state.auto_mode_active:
        st.markdown("""
        <div class="auto-mode-container">
            <h4 style="text-align: center; margin-bottom: 1rem;">🎯 AI Detection In Progress</h4>
        """, unsafe_allow_html=True)
        
        # Phase indicators
        phases = [
            ("vehicle_detection", "🚗 Vehicle Detection", "Point camera at vehicle"),
            ("license_plate_detection", "🔍 License Plate Detection", "Point camera at license plate"),
            ("gesture_detection", "✋ Hand Gesture Detection", "Show fingers (1-10) for hours, then OK gesture"),
            ("completed", "✅ Detection Complete", "All phases completed")
        ]
        
        current_phase = st.session_state.detection_phase
        
        for phase_id, phase_name, phase_desc in phases:
            if phase_id == current_phase:
                phase_class = "phase-active"
            elif phases.index((phase_id, phase_name, phase_desc)) < phases.index(next((p for p in phases if p[0] == current_phase), phases[-1])):
                phase_class = "phase-completed"
            else:
                phase_class = "phase-pending"
            
            st.markdown(f"""
            <div class="detection-phase {phase_class}">
                <strong>{phase_name}</strong><br>
                <small>{phase_desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Video Stream
    if st.session_state.auto_mode_active:
        st.markdown("#### 📹 Live Detection Feed")
        
        webrtc_ctx = webrtc_streamer(
            key="auto-detection",
            video_processor_factory=AutoDetectionProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Sync back from global state
        sync_session_state()
    
    # Detection Results
    if st.session_state.auto_detection_results:
        st.markdown("#### 🎯 Detection Results")
        
        results = st.session_state.auto_detection_results
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vehicle_type = results.get('vehicle_type', 'Not detected')
            st.metric("Vehicle Type", vehicle_type)
        
        with col2:
            license_plate = results.get('license_plate', 'Not detected')
            st.metric("License Plate", license_plate)
        
        with col3:
            parking_hours = results.get('parking_hours', 'Not detected')
            st.metric("Parking Hours", f"{parking_hours} hours" if parking_hours else 'Not detected')
        
        # Auto-park option
        if all(key in results for key in ['vehicle_type', 'license_plate', 'parking_hours']):
            st.markdown("#### 🚗 Auto-Park Vehicle")
            
            with st.form("auto_park_form"):
                customer_name = st.text_input("Customer Name", placeholder="Enter customer name")
                
                available_slots = parking_manager.get_available_slots()
                if available_slots:
                    slot_options = [f"Slot {slot.split('_')[1]}" for slot in available_slots]
                    selected_slot = st.selectbox("Select Parking Slot", slot_options)
                    
                    current_time = datetime.now()
                    arrival_dt = current_time
                    pickup_dt = current_time + timedelta(hours=results['parking_hours'])
                    
                    st.write(f"**Arrival Time:** {arrival_dt.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Expected Pickup:** {pickup_dt.strftime('%Y-%m-%d %H:%M')}")
                    
                    submitted = st.form_submit_button("🚗 Auto-Park Vehicle")
                    
                    if submitted:
                        if not customer_name:
                            st.error("Please enter customer name.")
                        else:
                            slot_id = f"slot_{selected_slot.split()[-1]}"
                            
                            success = parking_manager.park_vehicle(
                                slot_id, results['vehicle_type'], results['license_plate'],
                                customer_name, arrival_dt, pickup_dt
                            )
                            
                            if success:
                                st.success(f"Vehicle {results['license_plate']} auto-parked successfully in {selected_slot}!")
                                
                                # Reset auto mode
                                st.session_state.auto_mode_active = False
                                st.session_state.detection_phase = 'idle'
                                st.session_state.auto_detection_results = {}
                                detection_engine.reset_detection()
                                sync_global_state()
                                
                                st.rerun()
                            else:
                                st.error("Failed to park vehicle. Slot may not be available.")
                else:
                    st.warning("No available parking slots for auto-parking.")

def main():
    # Load CSS and initialize session state
    load_css()
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚗 Vehicle Vacancy Vault</h1>
        <p class="main-subtitle">Smart Parking Management with AI Auto Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current time display
    current_time = datetime.now()
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; color: rgba(255, 255, 255, 0.7);">
        <strong>Current Time:</strong> {current_time.strftime('%A, %B %d, %Y at %I:%M %p')}
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    render_metrics()
    
    # Pricing Information
    st.markdown("## 💰 Pricing Information")
    render_pricing_info()
    
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">
        <strong>Rush hours:</strong> Monday - Friday 5PM-12AM, Weekends 11AM-12AM, Holidays as scheduled
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("## 🏢 Parking Layout")
        render_parking_grid()
        render_legend()
    
    with col2:
        st.markdown("## ⚙️ Vehicle Management")
        
        # Tabs for different operations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Auto Mode", "🚗 Park Vehicle", "📅 Reserve Slot", "🚪 Remove Vehicle", "📊 Reports"])
        
        with tab1:
            render_auto_mode()
        
        with tab2:
            st.markdown("### Park New Vehicle")
            
            with st.form("park_vehicle_form"):
                vehicle_type = st.selectbox("Vehicle Type", ["Car", "Bike", "Truck"])
                vehicle_number = st.text_input("Vehicle Number", placeholder="e.g., WB01A1234")
                customer_name = st.text_input("Customer Name", placeholder="Enter customer name")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    arrival_date = st.date_input("Arrival Date", value=current_time.date())
                    arrival_time = st.time_input("Arrival Time", value=current_time.time())
                with col_b:
                    pickup_date = st.date_input("Expected Pickup Date", value=current_time.date())
                    pickup_time = st.time_input("Expected Pickup Time", 
                                               value=(current_time + timedelta(hours=2)).time())
                
                available_slots = parking_manager.get_available_slots()
                if available_slots:
                    slot_options = [f"Slot {slot.split('_')[1]}" for slot in available_slots]
                    selected_slot = st.selectbox("Select Parking Slot", slot_options)
                    
                    submitted = st.form_submit_button("🚗 Park Vehicle")
                    
                    if submitted:
                        if not vehicle_number or not customer_name:
                            st.error("Please fill in all required fields.")
                        else:
                            slot_id = f"slot_{selected_slot.split()[-1]}"
                            arrival_dt = datetime.combine(arrival_date, arrival_time)
                            pickup_dt = datetime.combine(pickup_date, pickup_time)
                            
                            if pickup_dt <= arrival_dt:
                                st.error("Pickup time must be after arrival time.")
                            else:
                                success = parking_manager.park_vehicle(
                                    slot_id, vehicle_type, vehicle_number, 
                                    customer_name, arrival_dt, pickup_dt
                                )
                                
                                if success:
                                    st.success(f"Vehicle {vehicle_number} parked successfully in {selected_slot}!")
                                    st.rerun()
                                else:
                                    st.error("Failed to park vehicle. Slot may not be available.")
                else:
                    st.warning("No available parking slots.")
        
        with tab3:
            st.markdown("### Reserve Parking Slot")
            
            with st.form("reserve_slot_form"):
                customer_name = st.text_input("Customer Name", placeholder="Enter customer name")
                vehicle_type = st.selectbox("Vehicle Type", ["Car", "Bike", "Truck"], key="reserve_type")
                vehicle_number = st.text_input("Vehicle Number", placeholder="e.g., WB01A1234", key="reserve_number")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    reserve_date = st.date_input("Reservation Date", value=current_time.date())
                    reserve_time = st.time_input("Reservation Time", value=current_time.time())
                with col_b:
                    duration = st.number_input("Duration (hours)", min_value=1, max_value=24, value=2)
                
                available_slots = parking_manager.get_available_slots()
                if available_slots:
                    slot_options = [f"Slot {slot.split('_')[1]}" for slot in available_slots]
                    selected_slot = st.selectbox("Select Parking Slot", slot_options, key="reserve_slot")
                    
                    submitted = st.form_submit_button("📅 Reserve Slot")
                    
                    if submitted:
                        if not vehicle_number or not customer_name:
                            st.error("Please fill in all required fields.")
                        else:
                            slot_id = f"slot_{selected_slot.split()[-1]}"
                            reserve_dt = datetime.combine(reserve_date, reserve_time)
                            
                            success = parking_manager.reserve_slot(
                                slot_id, vehicle_type, vehicle_number, 
                                customer_name, reserve_dt, duration
                            )
                            
                            if success:
                                st.success(f"Slot {selected_slot} reserved successfully for {customer_name}!")
                                st.rerun()
                            else:
                                st.error("Failed to reserve slot. Slot may not be available.")
                else:
                    st.warning("No available parking slots.")
        
        with tab4:
            st.markdown("### Remove Vehicle & Generate Bill")
            
            occupied_slots = parking_manager.get_occupied_slots()
            if occupied_slots:
                with st.form("remove_vehicle_form"):
                    slot_options = [f"Slot {slot.split('_')[1]}" for slot in occupied_slots]
                    selected_slot = st.selectbox("Select Occupied Slot", slot_options)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        departure_date = st.date_input("Current Date", value=current_time.date())
                        departure_time = st.time_input("Current Time", value=current_time.time())
                    
                    # Show slot information
                    if selected_slot:
                        slot_id = f"slot_{selected_slot.split()[-1]}"
                        slot_info = parking_manager.get_slot_data(slot_id)
                        
                        with col_b:
                            st.markdown("**Vehicle Information:**")
                            st.write(f"Vehicle: {slot_info.vehicle_type}")
                            st.write(f"Number: {slot_info.vehicle_number}")
                            st.write(f"Customer: {slot_info.customer_name}")
                            st.write(f"Arrival: {slot_info.arrival_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    submitted = st.form_submit_button("🚪 Generate Bill & Remove")
                    
                    if submitted:
                        slot_id = f"slot_{selected_slot.split()[-1]}"
                        departure_dt = datetime.combine(departure_date, departure_time)
                        
                        bill_info = parking_manager.remove_vehicle(slot_id, departure_dt)
                        
                        if bill_info:
                            st.success(f"Vehicle removed successfully from {selected_slot}!")
                            
                            # Display bill
                            st.markdown("### 🧾 Parking Bill")
                            st.markdown(f"""
                            <div style="background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
                                <h3 style="color: #6366f1; text-align: center; margin-bottom: 1rem;">Vehicle Vacancy Vault - Parking Bill</h3>
                                <hr style="border-color: rgba(255, 255, 255, 0.1);">
                                <p><strong>Transaction ID:</strong> {bill_info['id']}</p>
                                <p><strong>Vehicle Number:</strong> {bill_info['vehicle_number']}</p>
                                <p><strong>Vehicle Type:</strong> {bill_info['vehicle_type']}</p>
                                <p><strong>Customer Name:</strong> {bill_info['customer_name']}</p>
                                <p><strong>Parking Slot:</strong> {selected_slot}</p>
                                <p><strong>Arrival Time:</strong> {bill_info['arrival_time'].strftime('%Y-%m-%d %H:%M')}</p>
                                <p><strong>Departure Time:</strong> {bill_info['departure_time'].strftime('%Y-%m-%d %H:%M')}</p>
                                <hr style="border-color: rgba(255, 255, 255, 0.1);">
                                <p><strong>Duration:</strong> {bill_info['duration_hours']:.2f} hours</p>
                                <p><strong>Regular Hours:</strong> {bill_info['regular_hours']:.2f} × ₹{bill_info['base_rate']}</p>
                                <p><strong>Rush Hours:</strong> {bill_info['rush_hours']:.2f} × ₹{bill_info['base_rate'] + bill_info['rush_surcharge']}</p>
                                <p><strong>Night Hours:</strong> {bill_info['night_hours']:.2f} × ₹{bill_info['night_rate']}</p>
                                <hr style="border-color: rgba(255, 255, 255, 0.1);">
                                <h3 style="color: #10b981; text-align: center;"><strong>Total Amount: ₹{bill_info['total_cost']:.2f}</strong></h3>
                                <p style="text-align: center; margin-top: 1rem; font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">Thank you for using Vehicle Vacancy Vault!</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.rerun()
                        else:
                            st.error("Failed to remove vehicle.")
            else:
                st.info("No occupied slots to remove vehicles from.")
        
        with tab5:
            st.markdown("### 📊 Analytics & Reports")
            
            # Revenue metrics
            stats = parking_manager.get_statistics()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Revenue", f"₹{stats['total_revenue']:.2f}")
            with col2:
                st.metric("Total Transactions", stats['total_transactions'])
            with col3:
                st.metric("Current Occupancy", f"{stats['occupancy_rate']:.1f}%")
            
            # Transaction history
            recent_transactions = parking_manager.get_recent_transactions(10)
            if recent_transactions:
                st.markdown("#### Recent Transactions")
                
                # Create DataFrame for transactions
                transactions_df = pd.DataFrame([
                    {
                        'Date': t['departure_time'].strftime('%Y-%m-%d'),
                        'Time': t['departure_time'].strftime('%H:%M'),
                        'Vehicle': t['vehicle_number'],
                        'Type': t['vehicle_type'],
                        'Customer': t['customer_name'],
                        'Slot': t['slot_id'].replace('slot_', 'Slot '),
                        'Amount': f"₹{t['amount']:.2f}"
                    }
                    for t in recent_transactions
                ])
                
                st.dataframe(transactions_df, use_container_width=True)
                
                # Export data button
                if st.button("📥 Export Transaction Data"):
                    csv = transactions_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="parking_transactions.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No transaction history available.")
    
    # Sidebar with additional features
    with st.sidebar:
        st.markdown("## 🔧 Quick Actions")
        
        # Search vehicle
        st.markdown("### 🔍 Search Vehicle")
        search_query = st.text_input("Enter vehicle number", placeholder="e.g., WB01A1234")
        
        if search_query:
            found_slots = parking_manager.search_vehicle(search_query)
            
            if found_slots:
                for slot_id, slot_data in found_slots:
                    slot_num = slot_id.split('_')[1]
                    st.success(f"Found in Slot {slot_num}: {slot_data.vehicle_number} ({slot_data.status.value})")
            else:
                if len(search_query) > 2:
                    st.warning("Vehicle not found.")
        
        st.markdown("---")
        
        # Current slot status
        st.markdown("### 📊 Current Status")
        stats = parking_manager.get_statistics()
        status_data = {
            'Available': stats['available_count'],
            'Occupied': stats['occupied_count'],
            'Reserved': stats['reserved_count']
        }
        
        for status, count in status_data.items():
            st.metric(status, count)
        
        st.markdown("---")
        
        # Holiday calendar
        st.markdown("### 📅 Holiday Calendar 2025")
        
        with st.expander("View Holiday Rush Hours"):
            st.markdown("#### Upcoming Holidays with Rush Hours:")
            
            current_date = datetime.now().date()
            upcoming_holidays = [
                h for h in HOLIDAYS_2025 
                if datetime.strptime(h['date'], '%Y-%m-%d').date() >= current_date
            ][:5]  # Show next 5 holidays
            
            for holiday in upcoming_holidays:
                date_obj = datetime.strptime(holiday['date'], '%Y-%m-%d').date()
                st.write(f"**{holiday['holiday']}**")
                st.write(f"📅 {date_obj.strftime('%B %d, %Y')}")
                st.write(f"🕒 {holiday['hours']}")
                st.write("---")
        
        # Clear all data (admin function)
        st.markdown("---")
        st.markdown("### ⚠️ Admin Functions")
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I confirm to clear all parking data"):
                parking_manager.clear_all_data()
                st.success("All data cleared successfully!")
                st.rerun()

if __name__ == "__main__":
    main()
