import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import json
import uuid
from typing import Dict, List, Optional
import base64
import numpy as np
import threading
import time as time_module
import random

# Try importing computer vision libraries with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Some features will be disabled.")

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("WebRTC not available. Camera features will be disabled.")

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

# Sample parking data for simulation
class ParkingSlot:
    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.status = 'available'
        self.vehicle_type = None
        self.vehicle_number = None
        self.customer_name = None
        self.arrival_time = None
        self.expected_pickup = None
        
    def to_dict(self):
        return {
            'slot_id': self.slot_id,
            'status': self.status,
            'vehicle_type': self.vehicle_type,
            'vehicle_number': self.vehicle_number,
            'customer_name': self.customer_name,
            'arrival_time': self.arrival_time,
            'expected_pickup': self.expected_pickup
        }

class SimulatedParkingManager:
    def __init__(self):
        self.slots = {}
        self.transactions = []
        self.revenue = 0
        self.initialize_sample_data()
    
    def initialize_sample_data(self):
        # Initialize 20 slots
        for i in range(1, 21):
            self.slots[f"slot_{i}"] = ParkingSlot(f"slot_{i}")
        
        # Add some sample occupied slots
        sample_data = [
            {"slot": 2, "status": "occupied", "vehicle_type": "Bike", "vehicle_number": "WB02B5678", "customer_name": "John Doe"},
            {"slot": 4, "status": "occupied", "vehicle_type": "Car", "vehicle_number": "WB01A1234", "customer_name": "Jane Smith"},
            {"slot": 6, "status": "occupied", "vehicle_type": "Truck", "vehicle_number": "WB03C9101", "customer_name": "Mike Johnson"},
            {"slot": 8, "status": "reserved", "vehicle_type": "Car", "vehicle_number": "WB04D1121", "customer_name": "Alice Brown"},
        ]
        
        for data in sample_data:
            slot_id = f"slot_{data['slot']}"
            slot = self.slots[slot_id]
            slot.status = data['status']
            slot.vehicle_type = data['vehicle_type']
            slot.vehicle_number = data['vehicle_number']
            slot.customer_name = data['customer_name']
            slot.arrival_time = datetime.now() - timedelta(hours=random.randint(1, 5))
    
    def get_statistics(self):
        available = sum(1 for slot in self.slots.values() if slot.status == 'available')
        occupied = sum(1 for slot in self.slots.values() if slot.status == 'occupied')
        reserved = sum(1 for slot in self.slots.values() if slot.status == 'reserved')
        
        return {
            'total_slots': 20,
            'available_count': available,
            'occupied_count': occupied,
            'reserved_count': reserved,
            'occupancy_rate': (occupied + reserved) / 20 * 100,
            'total_revenue': self.revenue,
            'total_transactions': len(self.transactions)
        }
    
    def get_available_slots(self):
        return [slot_id for slot_id, slot in self.slots.items() if slot.status == 'available']
    
    def get_occupied_slots(self):
        return [slot_id for slot_id, slot in self.slots.items() if slot.status == 'occupied']
    
    def park_vehicle(self, slot_id, vehicle_type, vehicle_number, customer_name, arrival_dt, pickup_dt):
        if slot_id in self.slots and self.slots[slot_id].status == 'available':
            slot = self.slots[slot_id]
            slot.status = 'occupied'
            slot.vehicle_type = vehicle_type
            slot.vehicle_number = vehicle_number
            slot.customer_name = customer_name
            slot.arrival_time = arrival_dt
            slot.expected_pickup = pickup_dt
            return True
        return False
    
    def remove_vehicle(self, slot_id, departure_dt):
        if slot_id in self.slots and self.slots[slot_id].status == 'occupied':
            slot = self.slots[slot_id]
            
            # Calculate bill
            duration = (departure_dt - slot.arrival_time).total_seconds() / 3600
            base_rates = {'Car': 150, 'Bike': 200, 'Truck': 300}
            rate = base_rates.get(slot.vehicle_type, 150)
            total_cost = duration * rate
            
            bill_info = {
                'id': f"TXN{len(self.transactions) + 1:04d}",
                'vehicle_number': slot.vehicle_number,
                'vehicle_type': slot.vehicle_type,
                'customer_name': slot.customer_name,
                'arrival_time': slot.arrival_time,
                'departure_time': departure_dt,
                'duration_hours': duration,
                'regular_hours': duration,
                'rush_hours': 0,
                'night_hours': 0,
                'base_rate': rate,
                'rush_surcharge': 0,
                'night_rate': 100,
                'total_cost': total_cost,
                'amount': total_cost
            }
            
            self.transactions.append(bill_info)
            self.revenue += total_cost
            
            # Reset slot
            slot.status = 'available'
            slot.vehicle_type = None
            slot.vehicle_number = None
            slot.customer_name = None
            slot.arrival_time = None
            slot.expected_pickup = None
            
            return bill_info
        return None
    
    def get_slot_data(self, slot_id):
        return self.slots.get(slot_id, ParkingSlot(slot_id))
    
    def search_vehicle(self, vehicle_number):
        results = []
        for slot_id, slot in self.slots.items():
            if slot.vehicle_number and vehicle_number.upper() in slot.vehicle_number.upper():
                results.append((slot_id, slot))
        return results
    
    def get_recent_transactions(self, limit=10):
        return self.transactions[-limit:] if self.transactions else []

# Initialize parking manager
parking_mgr = SimulatedParkingManager()

# Custom CSS
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
    if 'simulation_mode' not in st.session_state:
        st.session_state.simulation_mode = True

# Updated WebRTC configuration with multiple STUN/TURN servers
def get_ice_servers():
    """Get ICE servers for WebRTC connection"""
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        # Free TURN servers for better connectivity
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject", 
            "credential": "openrelayproject"
        }
    ]

# Auto Detection Video Processor (Fixed for new API)
class AutoDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detection_start_time = None
        self.last_vehicle_type = None
        self.last_finger_count = 0
        self.finger_count_history = []
        self.ok_gesture_counter = 0
        
    def recv(self, frame):
        if not CV2_AVAILABLE:
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        
        # Use global state instead of st.session_state
        if not global_state.auto_mode_active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        current_time = time_module.time()
        
        # Add detection overlay text
        if CV2_AVAILABLE:
            cv2.putText(img, 'AI Detection Active', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Phase: {global_state.detection_phase}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
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

def sync_session_state():
    """Sync session state with global state"""
    st.session_state.auto_mode_active = global_state.auto_mode_active
    st.session_state.detection_phase = global_state.detection_phase
    st.session_state.auto_detection_results = global_state.auto_detection_results

def render_parking_grid():
    """Render the parking grid visualization"""
    cols = st.columns(4)
    
    for i in range(20):
        slot_id = f"slot_{i+1}"
        slot_data = parking_mgr.get_slot_data(slot_id)
        col_idx = i % 4
        
        with cols[col_idx]:
            status = slot_data.status
            slot_number = i + 1
            
            if status == 'available':
                css_class = 'slot-available'
                info_text = 'Available'
            elif status == 'occupied':
                css_class = 'slot-occupied'
                vehicle_info = f"{slot_data.vehicle_type or 'Vehicle'}"
                info_text = f"{vehicle_info}<br>{slot_data.vehicle_number or 'N/A'}"
            else:  # reserved
                css_class = 'slot-reserved'
                info_text = f"Reserved<br>{slot_data.vehicle_number or 'N/A'}"
            
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
    stats = parking_mgr.get_statistics()
    
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

def render_auto_mode_fallback():
    """Fallback Auto Mode without WebRTC"""
    st.markdown("### 🤖 AI Auto Detection Mode")
    
    # Check if we should use simulation mode
    use_simulation = st.session_state.get('simulation_mode', True)
    
    if use_simulation or not WEBRTC_AVAILABLE:
        st.info("📱 **Demo Mode**: Camera detection simulated for cloud deployment. Full AI detection available in local deployment.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎲 Simulate Auto Detection", type="primary", use_container_width=True):
                # Simulate detection process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Phase 1: Vehicle Detection
                status_text.text("🚗 Detecting vehicle...")
                for i in range(33):
                    progress_bar.progress(i + 1)
                    time_module.sleep(0.05)
                
                # Simulate results
                vehicle_types = ["Car", "Bike", "Truck"]
                states = ['MH', 'DL', 'KA', 'UP', 'WB', 'TN', 'GJ', 'RJ']
                
                simulated_results = {
                    'vehicle_type': random.choice(vehicle_types),
                    'license_plate': f"{random.choice(states)}{random.randint(1,99):02d}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000,9999)}",
                    'parking_hours': random.randint(2, 8)
                }
                
                # Phase 2: License Plate
                status_text.text("🔍 Reading license plate...")
                for i in range(33, 66):
                    progress_bar.progress(i + 1)
                    time_module.sleep(0.05)
                
                # Phase 3: Hand Gesture
                status_text.text("✋ Detecting parking duration...")
                for i in range(66, 100):
                    progress_bar.progress(i + 1)
                    time_module.sleep(0.05)
                
                status_text.text("✅ Detection completed!")
                st.session_state.auto_detection_results = simulated_results
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Simulation", use_container_width=True):
                st.session_state.auto_detection_results = {}
                st.rerun()
    
    else:
        # WebRTC Mode
        st.markdown("#### 📹 Live Detection Feed")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if not st.session_state.auto_mode_active:
                if st.button("🚀 Start Auto Detection", type="primary", use_container_width=True):
                    st.session_state.auto_mode_active = True
                    st.session_state.detection_phase = 'vehicle_detection'
                    st.session_state.auto_detection_results = {}
                    sync_global_state()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Detection", type="secondary", use_container_width=True):
                    st.session_state.auto_mode_active = False
                    st.session_state.detection_phase = 'idle'
                    sync_global_state()
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset Detection", use_container_width=True):
                st.session_state.auto_mode_active = False
                st.session_state.detection_phase = 'idle'
                st.session_state.auto_detection_results = {}
                sync_global_state()
                st.rerun()
        
        # Video Stream with improved configuration
        if st.session_state.auto_mode_active and WEBRTC_AVAILABLE:
            try:
                webrtc_ctx = webrtc_streamer(
                    key="auto-detection",
                    video_processor_factory=AutoDetectionProcessor,
                    rtc_configuration=RTCConfiguration({
                        "iceServers": get_ice_servers(),
                        "iceTransportPolicy": "all",
                        "bundlePolicy": "max-bundle",
                        "rtcpMuxPolicy": "require"
                    }),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 15, "max": 30}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # Show connection status
                if webrtc_ctx.state.playing:
                    st.success("🟢 Camera connected successfully!")
                elif webrtc_ctx.state.signalling:
                    st.info("🟡 Establishing camera connection...")
                else:
                    st.warning("🔴 Camera connection failed. Using simulation mode...")
                    st.session_state.simulation_mode = True
                    st.rerun()
                    
            except Exception as e:
                st.error(f"WebRTC Error: {str(e)}")
                st.info("💡 **Switching to simulation mode** due to connection issues.")
                st.session_state.simulation_mode = True
                st.rerun()
    
    # Show detection results
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
                
                available_slots = parking_mgr.get_available_slots()
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
                            
                            success = parking_mgr.park_vehicle(
                                slot_id, results['vehicle_type'], results['license_plate'],
                                customer_name, arrival_dt, pickup_dt
                            )
                            
                            if success:
                                st.success(f"Vehicle {results['license_plate']} auto-parked successfully in {selected_slot}!")
                                
                                # Reset auto mode
                                st.session_state.auto_mode_active = False
                                st.session_state.detection_phase = 'idle'
                                st.session_state.auto_detection_results = {}
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
            render_auto_mode_fallback()
        
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
                
                available_slots = parking_mgr.get_available_slots()
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
                                success = parking_mgr.park_vehicle(
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
            st.info("Reservation functionality available in full version.")
        
        with tab4:
            st.markdown("### Remove Vehicle & Generate Bill")
            
            occupied_slots = parking_mgr.get_occupied_slots()
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
                        slot_info = parking_mgr.get_slot_data(slot_id)
                        
                        with col_b:
                            st.markdown("**Vehicle Information:**")
                            st.write(f"Vehicle: {slot_info.vehicle_type or 'N/A'}")
                            st.write(f"Number: {slot_info.vehicle_number or 'N/A'}")
                            st.write(f"Customer: {slot_info.customer_name or 'N/A'}")
                            if slot_info.arrival_time:
                                st.write(f"Arrival: {slot_info.arrival_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    submitted = st.form_submit_button("🚪 Generate Bill & Remove")
                    
                    if submitted:
                        slot_id = f"slot_{selected_slot.split()[-1]}"
                        departure_dt = datetime.combine(departure_date, departure_time)
                        
                        bill_info = parking_mgr.remove_vehicle(slot_id, departure_dt)
                        
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
                                <p><strong>Rate:</strong> ₹{bill_info['base_rate']}/hour</p>
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
            stats = parking_mgr.get_statistics()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Revenue", f"₹{stats['total_revenue']:.2f}")
            with col2:
                st.metric("Total Transactions", stats['total_transactions'])
            with col3:
                st.metric("Current Occupancy", f"{stats['occupancy_rate']:.1f}%")
            
            # Transaction history
            recent_transactions = parking_mgr.get_recent_transactions(10)
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
                        'Amount': f"₹{t['amount']:.2f}"
                    }
                    for t in recent_transactions
                ])
                
                st.dataframe(transactions_df, use_container_width=True)
            else:
                st.info("No transaction history available.")
    
    # Sidebar with additional features
    with st.sidebar:
        st.markdown("## 🔧 Quick Actions")
        
        # Search vehicle
        st.markdown("### 🔍 Search Vehicle")
        search_query = st.text_input("Enter vehicle number", placeholder="e.g., WB01A1234")
        
        if search_query:
            found_slots = parking_mgr.search_vehicle(search_query)
            
            if found_slots:
                for slot_id, slot_data in found_slots:
                    slot_num = slot_id.split('_')[1]
                    st.success(f"Found in Slot {slot_num}: {slot_data.vehicle_number} ({slot_data.status})")
            else:
                if len(search_query) > 2:
                    st.warning("Vehicle not found.")
        
        st.markdown("---")
        
        # Current slot status
        st.markdown("### 📊 Current Status")
        stats = parking_mgr.get_statistics()
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

if __name__ == "__main__":
    main()
