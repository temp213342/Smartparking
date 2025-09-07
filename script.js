// Enhanced Vehicle Vacancy Vault - Smart Parking System JavaScript

// Vehicle data structure
let parkingData = [];
let holidayData = [];
let currentSelectedReservedSlot = null;

// Auto Mode variables
let autoModeActive = false;
let detectionInProgress = false;
let detectionStartTime = null;

// Theme management
function toggleTheme() {
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    
    if (body.getAttribute('data-theme') === 'dark') {
        body.setAttribute('data-theme', 'light');
        themeIcon.className = 'fas fa-sun';
        themeText.textContent = 'Light Mode';
        localStorage.setItem('theme', 'light');
        showNotification('☀️ Switched to Light Mode', 'success');
    } else {
        body.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-moon';
        themeText.textContent = 'Dark Mode';
        localStorage.setItem('theme', 'dark');
        showNotification('🌙 Switched to Dark Mode', 'success');
    }
}

// Load saved theme
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    
    body.setAttribute('data-theme', savedTheme);
    if (savedTheme === 'light') {
        themeIcon.className = 'fas fa-sun';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.className = 'fas fa-moon';
        themeText.textContent = 'Dark Mode';
    }
}

// Initialize parking data
function initializeParkingData() {
    for (let i = 1; i <= 20; i++) {
        parkingData.push({
            slot: i,
            vehicleType: null,
            vehicleNumber: null,
            arrivalDate: null,
            arrivalTime: null,
            expectedPickupDate: null,
            expectedPickupTime: null,
            weekday: null,
            charge: 0,
            isReserved: false,
            reservationData: null
        });
    }

    // Sample initial data with reservations
    const initialData = [
        {slot: 2, vehicleType: "Bike", vehicleNumber: "WB02B5678", arrivalDate: "01-02-25", arrivalTime: "09:30", expectedPickupDate: "01-02-25", expectedPickupTime: "12:00", weekday: "Mon", charge: 450.0},
        {slot: 4, vehicleType: "Car", vehicleNumber: "WB01A1234", arrivalDate: "01-02-25", arrivalTime: "10:00", expectedPickupDate: "01-02-25", expectedPickupTime: "14:00", weekday: "Mon", charge: 800.0},
        {slot: 6, vehicleType: "Truck", vehicleNumber: "WB03C9101", arrivalDate: "01-02-25", arrivalTime: "08:00", expectedPickupDate: "01-02-25", expectedPickupTime: "16:00", weekday: "Mon", charge: 2400.0},
        {slot: 8, vehicleType: "Car", vehicleNumber: "WB04D1121", arrivalDate: "31-01-25", arrivalTime: "22:00", expectedPickupDate: "01-02-25", expectedPickupTime: "06:00", weekday: "Tue", charge: 1600.0},
        {slot: 9, vehicleType: "Bike", vehicleNumber: "WB05E3141", arrivalDate: "01-02-25", arrivalTime: "15:00", expectedPickupDate: "01-02-25", expectedPickupTime: "18:00", weekday: "Mon", charge: 510.0},
        {slot: 11, vehicleType: "Truck", vehicleNumber: "WB06F7171", arrivalDate: "01-02-25", arrivalTime: "19:00", expectedPickupDate: "01-02-25", expectedPickupTime: "23:00", weekday: "Mon", charge: 1200.0},
        {slot: 13, vehicleType: "Car", vehicleNumber: "WB07G8181", arrivalDate: "01-02-25", arrivalTime: "07:00", expectedPickupDate: "01-02-25", expectedPickupTime: "11:00", weekday: "Mon", charge: 1000.0},
        {slot: 15, vehicleType: "Bike", vehicleNumber: "WB08H9191", arrivalDate: "01-02-25", arrivalTime: "21:00", expectedPickupDate: "01-02-25", expectedPickupTime: "23:00", weekday: "Mon", charge: 400.0},
        {slot: 17, vehicleType: "Truck", vehicleNumber: "WB09I2121", arrivalDate: "01-02-25", arrivalTime: "12:00", expectedPickupDate: "01-02-25", expectedPickupTime: "18:00", weekday: "Mon", charge: 1800.0},
        {slot: 19, vehicleType: "Car", vehicleNumber: "WB10J3131", arrivalDate: "01-02-25", arrivalTime: "20:00", expectedPickupDate: "01-02-25", expectedPickupTime: "22:00", weekday: "Mon", charge: 400.0}
    ];

    initialData.forEach(data => {
        parkingData[data.slot - 1] = { ...parkingData[data.slot - 1], ...data };
    });

    // Add sample reservation
    parkingData[2] = {
        ...parkingData[2],
        isReserved: true,
        reservationData: {
            customerName: "John Doe",
            vehicleType: "Car",
            vehicleNumber: "WB11X1234",
            date: "01-02-25",
            time: "14:00",
            duration: 3
        }
    };

    // Holiday data
    holidayData = [
        {date: "01-01-2025", name: "New Year's Day", rushFrom: "00:00", rushTo: "23:59"},
        {date: "26-01-2025", name: "Republic Day", rushFrom: "08:00", rushTo: "14:00"},
        {date: "02-02-2025", name: "Vasant Panchami", rushFrom: "09:00", rushTo: "17:00"},
        {date: "26-02-2025", name: "Maha Shivaratri", rushFrom: "09:00", rushTo: "17:00"},
        {date: "13-03-2025", name: "Holika Dahana", rushFrom: "09:00", rushTo: "22:00"},
        {date: "14-03-2025", name: "Holi", rushFrom: "09:00", rushTo: "20:00"},
        {date: "28-03-2025", name: "Jamat Ul-Vida", rushFrom: "09:00", rushTo: "17:00"},
        {date: "30-03-2025", name: "Chaitra Sukhladi / Ugadi / Gudi Padwa", rushFrom: "09:00", rushTo: "17:00"},
        {date: "31-03-2025", name: "Eid-ul-Fitr", rushFrom: "08:00", rushTo: "21:00"},
        {date: "06-04-2025", name: "Rama Navami", rushFrom: "09:00", rushTo: "17:00"},
        {date: "10-04-2025", name: "Mahavir Jayanti", rushFrom: "09:00", rushTo: "17:00"},
        {date: "18-04-2025", name: "Good Friday", rushFrom: "08:00", rushTo: "16:00"},
        {date: "12-05-2025", name: "Buddha Purnima", rushFrom: "09:00", rushTo: "18:00"},
        {date: "07-06-2025", name: "Eid ul-Adha (Bakrid)", rushFrom: "08:00", rushTo: "21:00"},
        {date: "06-07-2025", name: "Muharram", rushFrom: "07:00", rushTo: "19:00"},
        {date: "09-08-2025", name: "Raksha Bandhan", rushFrom: "10:00", rushTo: "18:00"},
        {date: "15-08-2025", name: "Independence Day", rushFrom: "08:00", rushTo: "14:00"},
        {date: "16-08-2025", name: "Janmashtami", rushFrom: "08:00", rushTo: "23:00"},
        {date: "27-08-2025", name: "Ganesh Chaturthi", rushFrom: "08:00", rushTo: "21:00"},
        {date: "05-09-2025", name: "Milad-un-Nabi / Onam", rushFrom: "09:00", rushTo: "17:00"},
        {date: "29-09-2025", name: "Maha Saptami", rushFrom: "06:00", rushTo: "23:59"},
        {date: "30-09-2025", name: "Maha Ashtami", rushFrom: "06:00", rushTo: "23:59"},
        {date: "01-10-2025", name: "Maha Navami", rushFrom: "06:00", rushTo: "23:59"},
        {date: "02-10-2025", name: "Mahatma Gandhi Jayanti / Dussehra", rushFrom: "08:00", rushTo: "17:00"},
        {date: "07-10-2025", name: "Maharishi Valmiki Jayanti", rushFrom: "09:00", rushTo: "17:00"},
        {date: "20-10-2025", name: "Diwali", rushFrom: "10:00", rushTo: "23:59"},
        {date: "22-10-2025", name: "Govardhan Puja", rushFrom: "09:00", rushTo: "18:00"},
        {date: "23-10-2025", name: "Bhai Duj", rushFrom: "10:00", rushTo: "18:00"},
        {date: "05-11-2025", name: "Guru Nanak Jayanti", rushFrom: "09:00", rushTo: "19:00"},
        {date: "24-11-2025", name: "Guru Tegh Bahadur's Martyrdom Day", rushFrom: "09:00", rushTo: "17:00"},
        {date: "25-12-2025", name: "Christmas Day", rushFrom: "09:00", rushTo: "22:00"},
        {date: "31-12-2025", name: "New Year's Eve", rushFrom: "00:00", rushTo: "23:59"}
    ];
}

// Auto Mode Functions
function showAutoModeModal() {
    document.getElementById('autoModeModal').classList.add('show');
    updateAutoModeStatus();
}

function getPhaseProgress(currentPhase) {
    const phase = currentPhase ? currentPhase.toLowerCase() : '';
    
    let vehicle = 'var(--text-muted)'; // Default: not started
    let license = 'var(--text-muted)'; // Default: not started  
    let gesture = 'var(--text-muted)'; // Default: not started
    
    if (phase.includes('vehicle') || phase.includes('detected')) {
        if (phase.includes('complete') || phase.includes('confirmed')) {
            vehicle = 'var(--accent-green)'; // Completed
            license = 'var(--accent-orange)'; // In progress
        } else {
            vehicle = 'var(--accent-orange)'; // In progress
        }
    } else if (phase.includes('license') || phase.includes('plate')) {
        vehicle = 'var(--accent-green)'; // Completed
        if (phase.includes('complete') || phase.includes('processed') || phase.includes('successfully')) {
            license = 'var(--accent-green)'; // Completed
            gesture = 'var(--accent-orange)'; // In progress
        } else {
            license = 'var(--accent-orange)'; // In progress
        }
    } else if (phase.includes('hand') || phase.includes('gesture') || phase.includes('finger') || phase.includes('confirm')) {
        vehicle = 'var(--accent-green)'; // Completed
        license = 'var(--accent-green)'; // Completed
        if (phase.includes('complete') || phase.includes('confirmed')) {
            gesture = 'var(--accent-green)'; // Completed
        } else {
            gesture = 'var(--accent-orange)'; // In progress
        }
    }
    
    return { vehicle, license, gesture };
}

function getPhaseTip(currentPhase) {
    const phase = currentPhase ? currentPhase.toLowerCase() : '';
    
    if (phase.includes('vehicle')) {
        return "💡 Point your camera at the vehicle clearly. Make sure the entire vehicle is visible in the frame.";
    } else if (phase.includes('license') || phase.includes('plate')) {
        if (phase.includes('processing') || phase.includes('ocr')) {
            return "💡 Analyzing license plate image... This may take a few seconds.";
        }
        return "💡 Point your camera directly at the license plate. Ensure good lighting and clear visibility.";
    } else if (phase.includes('hand') || phase.includes('gesture') || phase.includes('finger')) {
        if (phase.includes('confirm')) {
            return "💡 Make an OK gesture (thumb and index finger in a circle) to confirm the parking duration.";
        }
        return "💡 Show 1-10 fingers to indicate parking hours. Hold the same number for 3 seconds.";
    }
    
    return "💡 Follow the on-screen instructions for each detection phase.";
}

function updateAutoModeStatus() {
    const statusDiv = document.getElementById('detectionStatus');
    
    if (detectionInProgress) {
        statusDiv.innerHTML = `
            <div style="text-align: center;">
                <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: var(--accent-primary); margin-bottom: 1rem;"></i>
                <p style="color: var(--accent-primary); font-weight: 600;">AI Detection in Progress...</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Please follow the on-screen instructions</p>
                <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-glass); border-radius: 8px; border: 1px solid var(--border-primary);">
                    <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">Detection Phases:</p>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                        <span style="color: var(--accent-green);">✓ Vehicle Detection</span>
                        <span style="color: var(--accent-orange);">⏳ License Plate</span>
                        <span style="color: var(--text-muted);">⏸ Hand Gesture</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        statusDiv.innerHTML = `
            <div style="text-align: center; padding: 1rem;">
                <i class="fas fa-camera" style="font-size: 2rem; color: var(--text-secondary); margin-bottom: 1rem;"></i>
                <p style="color: var(--text-primary);">Ready to start AI detection</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Click "Start Auto Detection" to begin the 3-phase process</p>
            </div>
        `;
    }
}

function startAutoDetection() {
    if (detectionInProgress) {
        showNotification('Detection already in progress!', 'warning');
        return;
    }

    // Check server health first
    checkServerHealth()
    .then(() => {
    detectionInProgress = true;
        detectionStartTime = Date.now();
    updateAutoModeStatus();
    showNotification('Starting AI detection process...', 'info');

        // Disable modal close during detection
        const modal = document.getElementById('autoModeModal');
        const closeBtn = modal.querySelector('.close-btn');
        closeBtn.style.display = 'none';
        
        // Start the actual detection
        startDetectionProcess();
    })
    .catch(error => {
        showNotification('Server not available. Please start the Python server first.', 'error');
        console.error('Server health check failed:', error);
    });
}

function checkServerHealth() {
    return fetch('http://localhost:8000/health')
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status !== 'healthy') {
            throw new Error('Server is not healthy');
        }
        return data;
    });
}

function startDetectionProcess() {
    // Call the Python server to start detection
    fetch('http://localhost:8000/start_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'started') {
            showNotification('AI detection started successfully', 'success');
            pollDetectionResults();
        } else {
            throw new Error(data.message || 'Failed to start detection');
        }
    })
    .catch(error => {
        console.error('Error starting detection:', error);
        showNotification('Failed to start AI detection. Make sure the server is running.', 'error');
        detectionInProgress = false;
        updateAutoModeStatus();
        
        // Re-enable modal close button
        const modal = document.getElementById('autoModeModal');
        const closeBtn = modal.querySelector('.close-btn');
        closeBtn.style.display = 'block';
    });
}

function pollDetectionResults() {
    if (!detectionInProgress) return;

    fetch('http://localhost:8000/get_results')
    .then(response => response.json())
    .then(data => {
        const statusDiv = document.getElementById('detectionStatus');
        const resultsDiv = document.getElementById('detectionResults');

        if (data.status === 'running') {
            // Determine current phase for progress indicator
            let phaseProgress = getPhaseProgress(data.current_phase);
            let elapsedTime = detectionStartTime ? Math.floor((Date.now() - detectionStartTime) / 1000) : 0;
            let phaseTip = getPhaseTip(data.current_phase);
            
            statusDiv.innerHTML = `
                <div style="text-align: center;">
                    <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: var(--accent-primary); margin-bottom: 1rem;"></i>
                    <p style="color: var(--accent-primary); font-weight: 600;">${data.current_phase || 'Processing...'}</p>
                    ${data.message ? `<p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">${data.message}</p>` : ''}
                    <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-glass); border-radius: 8px; border: 1px solid var(--border-primary);">
                        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">Detection Progress:</p>
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                            <span style="color: ${phaseProgress.vehicle};">${phaseProgress.vehicle === 'var(--accent-green)' ? '✓' : phaseProgress.vehicle === 'var(--accent-orange)' ? '⏳' : '⏸'} Vehicle Detection</span>
                            <span style="color: ${phaseProgress.license};">${phaseProgress.license === 'var(--accent-green)' ? '✓' : phaseProgress.license === 'var(--accent-orange)' ? '⏳' : '⏸'} License Plate</span>
                            <span style="color: ${phaseProgress.gesture};">${phaseProgress.gesture === 'var(--accent-green)' ? '✓' : phaseProgress.gesture === 'var(--accent-orange)' ? '⏳' : '⏸'} Hand Gesture</span>
                        </div>
                        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border-primary);">
                            <p style="font-size: 0.7rem; color: var(--text-muted);">Elapsed: ${elapsedTime}s</p>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3);">
                        <p style="font-size: 0.8rem; color: var(--accent-blue); font-weight: 500;">${phaseTip}</p>
                    </div>
                </div>
            `;
            
            // Continue polling
            setTimeout(pollDetectionResults, 2000);
        } else if (data.status === 'completed') {
            detectionInProgress = false;
            const results = data.results || {};
            
            // Re-enable modal close button
            const modal = document.getElementById('autoModeModal');
            const closeBtn = modal.querySelector('.close-btn');
            closeBtn.style.display = 'block';
            
            statusDiv.innerHTML = `
                <div style="text-align: center;">
                    <i class="fas fa-check-circle" style="font-size: 2rem; color: var(--accent-green); margin-bottom: 1rem;"></i>
                    <p style="color: var(--accent-green); font-weight: 600;">Detection Completed!</p>
                </div>
            `;
            
            resultsDiv.innerHTML = `
                <div style="background: var(--bg-glass); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; border: 1px solid var(--border-primary);">
                    <h4 style="color: var(--accent-primary); margin-bottom: 1rem;">Detection Results:</h4>
                    <div style="display: grid; gap: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span><strong>Vehicle Type:</strong></span>
                            <span style="color: ${results.vehicle_type ? 'var(--accent-green)' : 'var(--accent-red)'};">${results.vehicle_type || 'Not detected'}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span><strong>License Plate:</strong></span>
                            <span style="color: ${results.license_plate ? 'var(--accent-green)' : 'var(--accent-red)'};">${results.license_plate || 'Not detected'}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span><strong>Parking Hours:</strong></span>
                            <span style="color: ${results.parking_hours ? 'var(--accent-green)' : 'var(--accent-red)'};">${results.parking_hours || 'Not detected'}</span>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-primary);">
                        <p style="font-size: 0.8rem; color: var(--text-muted); text-align: center; margin-bottom: 1rem;">
                            ${results.vehicle_type && results.license_plate && results.parking_hours ? 
                                '🎉 All detections successful! Ready to park vehicle.' : 
                                '⚠️ Some detections failed. You can manually enter missing information.'}
                        </p>
                        ${!(results.vehicle_type && results.license_plate && results.parking_hours) ? `
                            <div style="display: flex; gap: 0.5rem; justify-content: center;">
                                <button class="btn btn-secondary" onclick="useDetectionResults()" style="width: auto; padding: 0.5rem 1rem; font-size: 0.9rem;">
                                    <i class="fas fa-check"></i>
                                    Use Partial Results
                                </button>
                                <button class="btn btn-warning" onclick="startAutoDetection()" style="width: auto; padding: 0.5rem 1rem; font-size: 0.9rem;">
                                    <i class="fas fa-redo"></i>
                                    Retry Detection
                                </button>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            
            showNotification('AI detection completed successfully!', 'success');
        } else if (data.status === 'error') {
            detectionInProgress = false;
            
            // Re-enable modal close button
            const modal = document.getElementById('autoModeModal');
            const closeBtn = modal.querySelector('.close-btn');
            closeBtn.style.display = 'block';
            
            statusDiv.innerHTML = `
                <div style="text-align: center;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 2rem; color: var(--accent-red); margin-bottom: 1rem;"></i>
                    <p style="color: var(--accent-red); font-weight: 600;">Detection Error</p>
                    <p style="font-size: 0.9rem; color: var(--text-secondary);">${data.message || 'Unknown error occurred'}</p>
                </div>
            `;
            
            resultsDiv.innerHTML = `
                <div style="background: var(--bg-glass); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; border: 1px solid var(--accent-red);">
                    <div style="text-align: center;">
                        <p style="color: var(--accent-red); margin-bottom: 1rem;">No detection results available</p>
                        <button class="btn btn-secondary" onclick="startAutoDetection()" style="width: auto; padding: 0.5rem 1rem; font-size: 0.9rem;">
                            <i class="fas fa-redo"></i>
                            Retry Detection
                        </button>
                    </div>
                </div>
            `;
            
            showNotification('AI detection failed: ' + (data.message || 'Unknown error'), 'error');
        } else {
            // Continue polling for other statuses
            setTimeout(pollDetectionResults, 2000);
        }
    })
    .catch(error => {
        console.error('Error polling results:', error);
        detectionInProgress = false;
        updateAutoModeStatus();
        showNotification('Error communicating with AI detection server', 'error');
    });
}

function resetDetection() {
    fetch('http://localhost:8000/reset', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        detectionInProgress = false;
        
        // Re-enable modal close button
        const modal = document.getElementById('autoModeModal');
        const closeBtn = modal.querySelector('.close-btn');
        closeBtn.style.display = 'block';
        
        updateAutoModeStatus();
        document.getElementById('detectionResults').innerHTML = '';
        showNotification('Detection system reset', 'success');
    })
    .catch(error => {
        console.error('Error resetting detection:', error);
        showNotification('Error resetting detection system', 'error');
    });
}

function useDetectionResults() {
    fetch('http://localhost:8000/get_results')
    .then(response => response.json())
    .then(data => {
        if (data.status === 'completed' && data.results) {
            const results = data.results;
            let appliedCount = 0;
            
            // Map AI results to form
            if (results.vehicle_type) {
                const vehicleTypeMap = {
                    '4 Wheeler (Car)': 'Car',
                    'Two Wheeler (Bike)': 'Bike', 
                    'Heavy Vehicle (Bus/Truck)': 'Truck'
                };
                
                const mappedType = vehicleTypeMap[results.vehicle_type] || 'Car';
                document.getElementById('vehicleType').value = mappedType;
                appliedCount++;
            }
            
            if (results.license_plate) {
                document.getElementById('vehicleNumber').value = results.license_plate;
                appliedCount++;
            }
            
            // Set arrival time to now
            const now = new Date();
            document.getElementById('arrivalDate').value = now.toISOString().split('T')[0];
            document.getElementById('arrivalTime').value = now.toTimeString().split(':').slice(0, 2).join(':');
            
            // Set pickup time based on detected hours
            if (results.parking_hours) {
                const pickup = new Date(now.getTime() + results.parking_hours * 60 * 60 * 1000);
                document.getElementById('pickupDate').value = pickup.toISOString().split('T')[0];
                document.getElementById('pickupTime').value = pickup.toTimeString().split(':').slice(0, 2).join(':');
                appliedCount++;
            } else {
                // Set default 2 hours if no parking hours detected
                const pickup = new Date(now.getTime() + 2 * 60 * 60 * 1000);
                document.getElementById('pickupDate').value = pickup.toISOString().split('T')[0];
                document.getElementById('pickupTime').value = pickup.toTimeString().split(':').slice(0, 2).join(':');
            }
            
            // Close modal
            closeModal('autoModeModal');
            
            if (appliedCount > 0) {
                showNotification(`Applied ${appliedCount} detection result(s)! Please review and complete missing information.`, 'success');
            
            // Auto-park if all required fields are filled
            setTimeout(() => {
                if (document.getElementById('vehicleNumber').value && 
                    document.getElementById('arrivalDate').value &&
                    document.getElementById('pickupDate').value) {
                    parkVehicle();
                    } else {
                        showNotification('Please complete the missing vehicle information before parking.', 'warning');
                }
            }, 1000);
            } else {
                showNotification('No detection results to apply. Please try again or enter information manually.', 'warning');
            }
        } else {
            showNotification('No valid detection results to use', 'error');
        }
    })
    .catch(error => {
        console.error('Error getting detection results:', error);
        showNotification('Error retrieving detection results', 'error');
    });
}

// Reservation Functions
function showReserveModal() {
    const modal = document.getElementById('reserveModal');
    
    // Set current date and time as defaults
    const now = new Date();
    document.getElementById('reserveDate').value = now.toISOString().split('T')[0];
    document.getElementById('reserveTime').value = now.toTimeString().split(':').slice(0, 2).join(':');
    
    modal.classList.add('show');
}

function reserveSlot() {
    const customerName = document.getElementById('reserveCustomerName').value.trim();
    const vehicleType = document.getElementById('reserveVehicleType').value;
    const vehicleNumber = document.getElementById('reserveVehicleNumber').value.trim();
    const reserveDate = document.getElementById('reserveDate').value;
    const reserveTime = document.getElementById('reserveTime').value;
    const duration = parseInt(document.getElementById('reserveDuration').value);

    if (!customerName || !vehicleNumber || !reserveDate || !reserveTime || !duration) {
        showNotification('Please fill in all reservation fields', 'error');
        return;
    }

    // Check if vehicle is already parked or reserved
    const existingVehicle = parkingData.find(slot => 
        (slot.vehicleNumber && slot.vehicleNumber.toUpperCase() === vehicleNumber.toUpperCase()) ||
        (slot.reservationData && slot.reservationData.vehicleNumber.toUpperCase() === vehicleNumber.toUpperCase())
    );
    
    if (existingVehicle) {
        showNotification(`Vehicle ${vehicleNumber} already exists in the system`, 'error');
        return;
    }

    // Find available slot
    let availableSlot = null;
    for (let i = 0; i < parkingData.length; i++) {
        if (!parkingData[i].vehicleType && !parkingData[i].isReserved) {
            availableSlot = i;
            break;
        }
    }

    if (availableSlot === null) {
        showNotification('No available slots for reservation!', 'error');
        return;
    }

    // Reserve the slot
    parkingData[availableSlot].isReserved = true;
    parkingData[availableSlot].reservationData = {
        customerName: customerName,
        vehicleType: vehicleType,
        vehicleNumber: vehicleNumber.toUpperCase(),
        date: formatDate(reserveDate),
        time: reserveTime,
        duration: duration
    };

    // Clear form
    document.getElementById('reserveCustomerName').value = '';
    document.getElementById('reserveVehicleNumber').value = '';
    document.getElementById('reserveDuration').value = '2';

    closeModal('reserveModal');
    generateParkingGrid();
    showNotification(`Slot ${availableSlot + 1} reserved for ${customerName} (${vehicleNumber})`, 'success');
}

// Generate parking grid
function generateParkingGrid() {
    const grid = document.getElementById('parkingGrid');
    grid.innerHTML = '';

    parkingData.forEach(slot => {
        const slotElement = document.createElement('div');
        slotElement.className = 'parking-slot';
        slotElement.id = `slot-${slot.slot}`;
        
        if (slot.vehicleType) {
            // Occupied slot
            slotElement.classList.add('occupied');
            slotElement.innerHTML = `
                <div class="slot-number">${slot.slot}</div>
                <div class="slot-info">${slot.vehicleType}</div>
                <div class="slot-info">${slot.vehicleNumber}</div>
            `;
        } else if (slot.isReserved) {
            // Reserved slot
            slotElement.classList.add('reserved');
            slotElement.innerHTML = `
                <div class="slot-number">${slot.slot}</div>
                <div class="slot-info">Reserved</div>
                <div class="slot-info">${slot.reservationData.vehicleNumber}</div>
            `;
        } else {
            // Available slot
            slotElement.classList.add('empty');
            slotElement.innerHTML = `
                <div class="slot-number">${slot.slot}</div>
                <div class="slot-info">Available</div>
            `;
        }

        slotElement.addEventListener('click', () => selectSlot(slot.slot));
        grid.appendChild(slotElement);
    });

    updateStats();
}

// Update statistics
function updateStats() {
    const occupied = parkingData.filter(slot => slot.vehicleType).length;
    const reserved = parkingData.filter(slot => slot.isReserved && !slot.vehicleType).length;
    const available = 20 - occupied - reserved;

    document.getElementById('occupiedSlots').textContent = occupied;
    document.getElementById('reservedSlots').textContent = reserved;
    document.getElementById('availableSlots').textContent = available;
}

// Select slot
// Enhanced selectSlot function
function selectSlot(slotNumber) {
    const slot = parkingData[slotNumber - 1];
    
    if (slot.vehicleType) {
        // Show vehicle details
        showNotification(`Slot ${slotNumber}: ${slot.vehicleType} ${slot.vehicleNumber} (${slot.arrivalDate} ${slot.arrivalTime})`, 'success');
    } else if (slot.isReserved) {
        // Show reservation details with conversion option
        const reservation = slot.reservationData;
        
        if (confirm(`Slot ${slotNumber} is reserved for ${reservation.customerName} - ${reservation.vehicleNumber}\n\nConvert this reservation to parked status?`)) {
            convertReservedToParked(slotNumber);
        } else {
            showNotification(`Slot ${slotNumber} reserved for ${reservation.customerName} - ${reservation.vehicleNumber}`, 'info');
        }
    } else {
        // Auto-fill slot for parking
        showNotification(`Selected slot ${slotNumber} for parking`, 'success');
    }
}


// Park vehicle with improved logic
function parkVehicle() {
    const vehicleType = document.getElementById('vehicleType').value;
    const vehicleNumber = document.getElementById('vehicleNumber').value.trim();
    const arrivalDate = document.getElementById('arrivalDate').value;
    const arrivalTime = document.getElementById('arrivalTime').value;
    const pickupDate = document.getElementById('pickupDate').value;
    const pickupTime = document.getElementById('pickupTime').value;

    // Validation
    const errors = validateForm();
    if (errors.length > 0) {
        showNotification(errors[0], 'error');
        return;
    }

    // Check if vehicle is already parked
    const existingVehicle = parkingData.find(slot => 
        slot.vehicleNumber && slot.vehicleNumber.toUpperCase() === vehicleNumber.toUpperCase()
    );
    
    if (existingVehicle) {
        showNotification(`Vehicle ${vehicleNumber} already parked in slot ${existingVehicle.slot}`, 'error');
        return;
    }

    // Check if there's a reservation for this vehicle
    let reservedSlotIndex = null;
    for (let i = 0; i < parkingData.length; i++) {
        if (parkingData[i].isReserved && 
            parkingData[i].reservationData.vehicleNumber.toUpperCase() === vehicleNumber.toUpperCase()) {
            reservedSlotIndex = i;
            break;
        }
    }

    let emptySlot = null;
    
    if (reservedSlotIndex !== null) {
        // Use reserved slot
        emptySlot = reservedSlotIndex;
        showNotification(`Using reserved slot ${emptySlot + 1} for ${vehicleNumber}`, 'info');
    } else {
        // Find empty slot based on vehicle type
        if (vehicleType === 'Truck') {
            // Trucks get higher numbered slots
            for (let i = parkingData.length - 1; i >= 0; i--) {
                if (!parkingData[i].vehicleType && !parkingData[i].isReserved) {
                    emptySlot = i;
                    break;
                }
            }
        } else {
            // Cars and bikes get lower numbered slots
            for (let i = 0; i < parkingData.length; i++) {
                if (!parkingData[i].vehicleType && !parkingData[i].isReserved) {
                    emptySlot = i;
                    break;
                }
            }
        }
    }

    if (emptySlot === null) {
        showNotification('No available slots!', 'error');
        return;
    }

    // Convert date to get weekday
    const arrivalDateTime = new Date(arrivalDate + 'T' + arrivalTime);
    const weekday = arrivalDateTime.toLocaleDateString('en-US', { weekday: 'short' });

    // Park vehicle
    parkingData[emptySlot] = {
        slot: emptySlot + 1,
        vehicleType: vehicleType,
        vehicleNumber: vehicleNumber.toUpperCase(),
        arrivalDate: formatDate(arrivalDate),
        arrivalTime: arrivalTime,
        expectedPickupDate: formatDate(pickupDate),
        expectedPickupTime: pickupTime,
        weekday: weekday,
        charge: 0,
        isReserved: false,
        reservationData: null
    };

    // Clear form
    document.getElementById('vehicleNumber').value = '';
    setCurrentDateTime();
    
    generateParkingGrid();
    showNotification(`Vehicle ${vehicleNumber.toUpperCase()} parked at slot ${emptySlot + 1}`, 'success');
}

// Modified showRemoveModal function
function showRemoveModal() {
    const modal = document.getElementById('removeModal');
    const select = document.getElementById('removeSlotSelect');
    const slotInfo = document.getElementById('selectedSlotInfo');
    
    // Clear existing options and info
    select.innerHTML = '<option value="">Select a slot</option>';
    slotInfo.innerHTML = '';
    
    // Get occupied AND reserved slots
    const occupiedSlots = parkingData.filter(slot => slot.vehicleType);
    const reservedSlots = parkingData.filter(slot => slot.isReserved && !slot.vehicleType);
    
    if (occupiedSlots.length === 0 && reservedSlots.length === 0) {
        showNotification('No vehicles to remove or reserved slots!', 'error');
        return;
    }
    
    // Populate occupied slots
    occupiedSlots.forEach(slot => {
        const option = document.createElement('option');
        option.value = slot.slot;
        option.textContent = `Slot ${slot.slot} - ${slot.vehicleType} ${slot.vehicleNumber} (PARKED)`;
        select.appendChild(option);
    });

    // Populate reserved slots
    reservedSlots.forEach(slot => {
        const option = document.createElement('option');
        option.value = slot.slot;
        option.textContent = `Slot ${slot.slot} - ${slot.reservationData.vehicleNumber} (RESERVED)`;
        select.appendChild(option);
    });

    // Enhanced slot info display
    select.addEventListener('change', function() {
        const slotNumber = parseInt(this.value);
        if (slotNumber) {
            const slot = parkingData[slotNumber - 1];
            
            if (slot.vehicleType) {
                // Occupied slot
                slotInfo.innerHTML = `
                    <div style="background: rgba(239, 68, 68, 0.1); border-radius: 12px; padding: 1rem; margin: 1rem 0; border: 1px solid rgba(239, 68, 68, 0.3);">
                        <h4 style="color: #ef4444; margin-bottom: 0.5rem;">🚗 Parked Vehicle Details:</h4>
                        <div><strong>Vehicle Type:</strong> ${slot.vehicleType}</div>
                        <div><strong>Vehicle Number:</strong> ${slot.vehicleNumber}</div>
                        <div><strong>Arrival:</strong> ${slot.arrivalDate} ${slot.arrivalTime}</div>
                        <div><strong>Expected Pickup:</strong> ${slot.expectedPickupDate} ${slot.expectedPickupTime}</div>
                    </div>
                `;
            } else if (slot.isReserved) {
                // Reserved slot
                slotInfo.innerHTML = `
                    <div style="background: rgba(245, 158, 11, 0.1); border-radius: 12px; padding: 1rem; margin: 1rem 0; border: 1px solid rgba(245, 158, 11, 0.3);">
                        <h4 style="color: #f59e0b; margin-bottom: 0.5rem;">📅 Reservation Details:</h4>
                        <div><strong>Customer:</strong> ${slot.reservationData.customerName}</div>
                        <div><strong>Vehicle Type:</strong> ${slot.reservationData.vehicleType}</div>
                        <div><strong>Vehicle Number:</strong> ${slot.reservationData.vehicleNumber}</div>
                        <div><strong>Reserved Date:</strong> ${slot.reservationData.date}</div>
                        <div><strong>Reserved Time:</strong> ${slot.reservationData.time}</div>
                        <div><strong>Duration:</strong> ${slot.reservationData.duration} hours</div>
                        
                        <button class="btn btn-primary" onclick="convertReservedToParked(${slotNumber})" style="margin-top: 1rem; width: 100%;">
                            <i class="fas fa-car"></i>
                            Convert to Parked Status
                        </button>
                    </div>
                `;
            }
        } else {
            slotInfo.innerHTML = '';
        }
    });

    // Set current date and time
    const now = new Date();
    document.getElementById('currentDate').value = now.toISOString().split('T')[0];
    document.getElementById('currentTime').value = now.toTimeString().split(':').slice(0, 2).join(':');

    modal.classList.add('show');
}

// Convert reserved slot to parked status
function convertReservedToParked(slotNumber) {
    const slot = parkingData[slotNumber - 1];
    
    if (!slot.isReserved) {
        showNotification('This slot is not reserved!', 'error');
        return;
    }
    
    const reservation = slot.reservationData;
    
    // Convert reservation to parked vehicle
    const now = new Date();
    const arrivalDateTime = now;
    const pickupDateTime = new Date(now.getTime() + reservation.duration * 60 * 60 * 1000);
    
    parkingData[slotNumber - 1] = {
        slot: slotNumber,
        vehicleType: reservation.vehicleType,
        vehicleNumber: reservation.vehicleNumber,
        arrivalDate: formatDateFromJS(arrivalDateTime),
        arrivalTime: arrivalDateTime.toTimeString().split(':').slice(0, 2).join(':'),
        expectedPickupDate: formatDateFromJS(pickupDateTime),
        expectedPickupTime: pickupDateTime.toTimeString().split(':').slice(0, 2).join(':'),
        weekday: arrivalDateTime.toLocaleDateString('en-US', { weekday: 'short' }),
        charge: 0,
        isReserved: false,
        reservationData: null,
        // Store original reservation info for reference
        originalReservation: {
            customerName: reservation.customerName,
            reservedDate: reservation.date,
            reservedTime: reservation.time
        }
    };
    
    // Close modal and refresh grid
    closeModal('removeModal');
    generateParkingGrid();
    
    showNotification(
        `✅ Reservation converted! ${reservation.vehicleNumber} is now parked in slot ${slotNumber}`, 
        'success'
    );
}


// Remove vehicle with enhanced billing
function removeVehicle() {
    const slotNumber = parseInt(document.getElementById('removeSlotSelect').value);
    const currentDate = document.getElementById('currentDate').value;
    const currentTime = document.getElementById('currentTime').value;

    if (!slotNumber || !currentDate || !currentTime) {
        showNotification('Please fill in all fields', 'error');
        return;
    }

    const slot = parkingData[slotNumber - 1];
    if (!slot.vehicleType) {
        showNotification('Slot is empty!', 'error');
        return;
    }

    // Calculate charges
    const arrivalDateTime = parseDateTime(slot.arrivalDate, slot.arrivalTime);
    const currentDateTime = parseDateTime(formatDate(currentDate), currentTime);
    
    if (currentDateTime < arrivalDateTime) {
        showNotification('Departure time cannot be before arrival time!', 'error');
        return;
    }
    
    const chargeInfo = calculateCharge(arrivalDateTime, currentDateTime, slot.vehicleType);

    // Generate bill
    generateBill(slot, chargeInfo, currentDateTime);

    // Clear slot
    parkingData[slotNumber - 1] = {
        slot: slotNumber,
        vehicleType: null,
        vehicleNumber: null,
        arrivalDate: null,
        arrivalTime: null,
        expectedPickupDate: null,
        expectedPickupTime: null,
        weekday: null,
        charge: chargeInfo.total,
        isReserved: false,
        reservationData: null
    };

    closeModal('removeModal');
    generateParkingGrid();
    showNotification('Vehicle removed successfully', 'success');
}

// Enhanced charge calculation with proper holiday logic
function calculateCharge(arrival, current, vehicleType) {
    const standardRate = {Car: 150, Bike: 200, Truck: 300};
    const rushExtra = {Car: 30, Bike: 50, Truck: 70};
    const nightRate = 100;

    const totalMs = current - arrival;
    const totalHours = Math.ceil(totalMs / (1000 * 60 * 60));

    let standardHours = 0;
    let rushHours = 0;
    let nightHours = 0;

    // Calculate hour by hour
    for (let i = 0; i < totalHours; i++) {
        const currentTime = new Date(arrival.getTime() + i * 60 * 60 * 1000);
        const hour = currentTime.getHours();
        const dayOfWeek = currentTime.getDay(); // 0 = Sunday, 5 = Friday, 6 = Saturday
        const dateStr = formatDateForComparison(currentTime);

        // Check for holiday
        const holiday = holidayData.find(h => h.date === dateStr);
        let isRushHour = false;
        let isNightHour = (hour >= 23 || hour < 5);

        if (holiday) {
            const rushStart = parseInt(holiday.rushFrom.split(':')[0]);
            let rushEnd = parseInt(holiday.rushTo.split(':')[0]);
            
            // Handle full day holidays (00:00 to 23:59)
            if (holiday.rushFrom === "00:00" && holiday.rushTo === "23:59") {
                if (!isNightHour) {
                    isRushHour = true;
                }
            } else {
                // Handle crossing midnight
                if (rushEnd === 23 && holiday.rushTo.includes("59")) {
                    rushEnd = 24; // Treat as end of day
                }
                
                if (rushStart <= rushEnd) {
                    if (hour >= rushStart && hour < rushEnd && !isNightHour) {
                        isRushHour = true;
                    }
                } else {
                    // Crosses midnight
                    if ((hour >= rushStart || hour < rushEnd) && !isNightHour) {
                        isRushHour = true;
                    }
                }
            }
        } else {
            // Regular rush hours
            if (!isNightHour) {
                if ((dayOfWeek === 5 && hour >= 17) || // Friday 5PM+
                    ((dayOfWeek === 0 || dayOfWeek === 6) && hour >= 11)) { // Weekend 11AM+
                    isRushHour = true;
                }
            }
        }

        if (isNightHour) {
            nightHours++;
        } else if (isRushHour) {
            rushHours++;
        } else {
            standardHours++;
        }
    }

    const standardCharge = standardHours * standardRate[vehicleType];
    const rushCharge = rushHours * (standardRate[vehicleType] + rushExtra[vehicleType]);
    const nightCharge = nightHours * nightRate;
    const total = standardCharge + rushCharge + nightCharge;

    return {
        total,
        standardHours,
        rushHours,
        nightHours,
        standardCharge,
        rushCharge,
        nightCharge,
        totalHours
    };
}

// Enhanced bill generation
function generateBill(slot, chargeInfo, currentDateTime) {
    const billContent = document.getElementById('billContent');
    const currentDate = formatDateFromJS(currentDateTime);
    const currentTime = currentDateTime.toTimeString().substring(0, 5);

    billContent.innerHTML = `
        <div class="bill-header">
            <h3>🅿️ PARKING BILL</h3>
            <p><strong>Vehicle Vacancy Vault</strong></p>
            <p>Smart Parking Management System</p>
            <small>${currentDate} ${currentTime}</small>
        </div>
        
        <div class="bill-row">
            <span>Vehicle Type:</span>
            <span><strong>${slot.vehicleType}</strong></span>
        </div>
        <div class="bill-row">
            <span>Vehicle Number:</span>
            <span><strong>${slot.vehicleNumber}</strong></span>
        </div>
        <div class="bill-row">
            <span>Slot Number:</span>
            <span><strong>${slot.slot}</strong></span>
        </div>
        <div class="bill-row">
            <span>Entry:</span>
            <span>${slot.arrivalDate} ${slot.arrivalTime}</span>
        </div>
        <div class="bill-row">
            <span>Exit:</span>
            <span>${currentDate} ${currentTime}</span>
        </div>
        <div class="bill-row">
            <span>Total Duration:</span>
            <span><strong>${chargeInfo.totalHours} hour${chargeInfo.totalHours > 1 ? 's' : ''}</strong></span>
        </div>
        
        <hr style="margin: 1.5rem 0; border: 1px solid var(--border-primary);">
        
        <div style="font-size: 0.95rem;">
            ${chargeInfo.standardHours > 0 ? `
            <div class="bill-row">
                <span>Standard Hours (${chargeInfo.standardHours}h @ ₹${getStandardRate(slot.vehicleType)}/h):</span>
                <span>₹${chargeInfo.standardCharge.toFixed(2)}</span>
            </div>` : ''}
            ${chargeInfo.rushHours > 0 ? `
            <div class="bill-row">
                <span>Rush Hours (${chargeInfo.rushHours}h @ ₹${getStandardRate(slot.vehicleType) + getRushExtra(slot.vehicleType)}/h):</span>
                <span>₹${chargeInfo.rushCharge.toFixed(2)}</span>
            </div>` : ''}
            ${chargeInfo.nightHours > 0 ? `
            <div class="bill-row">
                <span>Night Hours (${chargeInfo.nightHours}h @ ₹100/h):</span>
                <span>₹${chargeInfo.nightCharge.toFixed(2)}</span>
            </div>` : ''}
        </div>
        
        <div class="bill-total">
            <div class="bill-row">
                <span><strong>TOTAL AMOUNT:</strong></span>
                <span><strong>₹${chargeInfo.total.toFixed(2)}</strong></span>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 2rem; font-size: 0.9rem; color: var(--text-secondary);">
            <p>🚗 Thank you for using Vehicle Vacancy Vault! 🚗</p>
            <p>Have a safe journey!</p>
            <hr style="margin: 1rem 0; border: 1px solid var(--border-primary);">
            <small>For support: support@vehiclevacancyvault.com | Ph: +91-XXXX-XXXXXX</small>
        </div>
    `;

    document.getElementById('billModal').classList.add('show');
}

// Helper functions for bill generation
function getStandardRate(vehicleType) {
    const rates = {Car: 150, Bike: 200, Truck: 300};
    return rates[vehicleType];
}

function getRushExtra(vehicleType) {
    const extras = {Car: 30, Bike: 50, Truck: 70};
    return extras[vehicleType];
}

// Show holiday calendar modal
function showHolidayCalendar() {
    const modal = document.getElementById('holidayModal');
    const tbody = document.getElementById('holidayTableBody');
    
    // Clear existing content
    tbody.innerHTML = '';
    
    // Populate holiday table
    holidayData.forEach(holiday => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${holiday.date}</strong></td>
            <td>${holiday.name}</td>
            <td>${holiday.rushFrom} - ${holiday.rushTo}</td>
        `;
        tbody.appendChild(row);
    });
    
    modal.classList.add('show');
}

// Enhanced generate report function
function generateReport() {
    const occupiedSlots = parkingData.filter(slot => slot.vehicleType);
    const reservedSlots = parkingData.filter(slot => slot.isReserved && !slot.vehicleType);
    const availableSlots = parkingData.filter(slot => !slot.vehicleType && !slot.isReserved);
    const totalRevenue = parkingData.reduce((sum, slot) => sum + (slot.charge || 0), 0);
    
    const vehicleTypes = {};
    const revenueByType = {};
    
    occupiedSlots.forEach(slot => {
        vehicleTypes[slot.vehicleType] = (vehicleTypes[slot.vehicleType] || 0) + 1;
    });
    
    parkingData.forEach(slot => {
        if (slot.vehicleType && slot.charge > 0) {
            revenueByType[slot.vehicleType] = (revenueByType[slot.vehicleType] || 0) + slot.charge;
        }
    });

    const reportContent = document.getElementById('reportContent');
    
    reportContent.innerHTML = `
        <div class="report-section">
            <h3><i class="fas fa-chart-pie"></i> Occupancy Statistics</h3>
            <div class="report-grid">
                <div class="report-card">
                    <h4>Total Slots</h4>
                    <div class="stat-number" style="font-size: 2rem; color: var(--accent-primary);">20</div>
                </div>
                <div class="report-card">
                    <h4>Occupied</h4>
                    <div class="stat-number" style="font-size: 2rem; color: var(--accent-red);">${occupiedSlots.length}</div>
                </div>
                <div class="report-card">
                    <h4>Reserved</h4>
                    <div class="stat-number" style="font-size: 2rem; color: var(--accent-orange);">${reservedSlots.length}</div>
                </div>
                <div class="report-card">
                    <h4>Available</h4>
                    <div class="stat-number" style="font-size: 2rem; color: var(--accent-green);">${availableSlots.length}</div>
                </div>
                <div class="report-card">
                    <h4>Occupancy Rate</h4>
                    <div class="stat-number" style="font-size: 2rem; color: var(--accent-secondary);">${((occupiedSlots.length / 20) * 100).toFixed(1)}%</div>
                </div>
            </div>
        </div>

        <div class="report-section">
            <h3><i class="fas fa-car"></i> Vehicle Breakdown</h3>
            <div class="report-grid">
                ${Object.entries(vehicleTypes).map(([type, count]) => `
                    <div class="report-card">
                        <h4>${type}s</h4>
                        <div class="stat-number" style="font-size: 1.8rem; color: var(--accent-blue);">${count}</div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="report-section">
            <h3><i class="fas fa-rupee-sign"></i> Revenue Analysis</h3>
            <div class="report-grid">
                <div class="report-card">
                    <h4>Total Revenue</h4>
                    <div class="stat-number" style="font-size: 1.8rem; color: var(--accent-green);">₹${totalRevenue.toFixed(2)}</div>
                </div>
                ${Object.entries(revenueByType).map(([type, revenue]) => `
                    <div class="report-card">
                        <h4>${type} Revenue</h4>
                        <div class="stat-number" style="font-size: 1.5rem; color: var(--accent-secondary);">₹${revenue.toFixed(2)}</div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="report-section">
            <h3><i class="fas fa-info-circle"></i> System Status</h3>
            <div class="report-grid">
                <div class="report-card">
                    <h4>AI Detection</h4>
                    <div class="stat-number" style="font-size: 1.2rem; color: var(--accent-blue);">${detectionInProgress ? 'Active' : 'Ready'}</div>
                </div>
                <div class="report-card">
                    <h4>Active Reservations</h4>
                    <div class="stat-number" style="font-size: 1.5rem; color: var(--accent-orange);">${reservedSlots.length}</div>
                </div>
                <div class="report-card">
                    <h4>Average Revenue/Slot</h4>
                    <div class="stat-number" style="font-size: 1.2rem; color: var(--accent-green);">₹${(totalRevenue / (occupiedSlots.length || 1)).toFixed(2)}</div>
                </div>
            </div>
        </div>

        <div class="report-section" style="text-align: center; background: var(--bg-glass);">
            <small style="color: var(--text-muted);">Report generated on: ${new Date().toLocaleString()}</small>
        </div>
    `;

    document.getElementById('reportModal').classList.add('show');
}

// Enhanced search functionality
function searchVehicle() {
    const searchTerm = prompt('Enter vehicle number or slot number to search:');
    if (!searchTerm) return;

    const term = searchTerm.trim().toLowerCase();
    
    // Clear any existing highlights
    document.querySelectorAll('.parking-slot').forEach(slot => {
        slot.classList.remove('highlighted');
    });

    let found = null;
    
    // Search by vehicle number
    if (isNaN(term)) {
        found = parkingData.find(slot => 
            (slot.vehicleNumber && slot.vehicleNumber.toLowerCase().includes(term)) ||
            (slot.reservationData && slot.reservationData.vehicleNumber.toLowerCase().includes(term))
        );
    } else {
        // Search by slot number
        const slotNum = parseInt(term);
        if (slotNum >= 1 && slotNum <= 20) {
            found = parkingData[slotNum - 1];
        }
    }
    
    if (found) {
        let message = '';
        if (found.vehicleType) {
            message = `Found: ${found.vehicleType} ${found.vehicleNumber} in slot ${found.slot} (Parked: ${found.arrivalDate} ${found.arrivalTime})`;
        } else if (found.isReserved) {
            message = `Slot ${found.slot} reserved for ${found.reservationData.customerName} - ${found.reservationData.vehicleNumber}`;
        } else {
            message = `Slot ${found.slot} is available`;
        }
        
        showNotification(message, 'success');
        
        // Highlight the slot
        const slotElement = document.getElementById(`slot-${found.slot}`);
        if (slotElement) {
            slotElement.classList.add('highlighted');
            slotElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            setTimeout(() => {
                slotElement.classList.remove('highlighted');
            }, 5000);
        }
    } else {
        showNotification('No results found', 'error');
    }
}

// Enhanced export data function
function exportData() {
    try {
        const exportData = {
            timestamp: new Date().toISOString(),
            parkingData: parkingData,
            holidayData: holidayData,
            stats: {
                totalSlots: 20,
                occupiedSlots: parkingData.filter(slot => slot.vehicleType).length,
                reservedSlots: parkingData.filter(slot => slot.isReserved).length,
                availableSlots: parkingData.filter(slot => !slot.vehicleType && !slot.isReserved).length,
                totalRevenue: parkingData.reduce((sum, slot) => sum + (slot.charge || 0), 0)
            }
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `parking_data_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showNotification('Data exported successfully', 'success');
    } catch (error) {
        showNotification('Error exporting data', 'error');
        console.error('Export error:', error);
    }
}

// Import data functionality
function importData() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const data = JSON.parse(e.target.result);
                    if (data.parkingData && Array.isArray(data.parkingData)) {
                        if (confirm('This will replace all current data. Are you sure?')) {
                            parkingData = data.parkingData;
                            if (data.holidayData) {
                                holidayData = data.holidayData;
                            }
                            generateParkingGrid();
                            showNotification('Data imported successfully', 'success');
                        }
                    } else {
                        showNotification('Invalid data format', 'error');
                    }
                } catch (error) {
                    showNotification('Error reading file', 'error');
                }
            };
            reader.readAsText(file);
        }
    };
    
    input.click();
}

// Print bill function
function printBill() {
    const printBtn = document.querySelector('.btn-print');
    const originalDisplay = printBtn.style.display;
    printBtn.style.display = 'none';
    
    window.print();
    
    setTimeout(() => {
        printBtn.style.display = originalDisplay;
    }, 1000);
}

// Clear all data
function clearAllData() {
    if (confirm('Are you sure you want to clear all parking data? This action cannot be undone.')) {
        parkingData.forEach((slot, index) => {
            parkingData[index] = {
                slot: index + 1,
                vehicleType: null,
                vehicleNumber: null,
                arrivalDate: null,
                arrivalTime: null,
                expectedPickupDate: null,
                expectedPickupTime: null,
                weekday: null,
                charge: 0,
                isReserved: false,
                reservationData: null
            };
        });
        generateParkingGrid();
        showNotification('All parking data cleared successfully', 'success');
    }
}

// Auto-fill current date and time
function setCurrentDateTime() {
    const now = new Date();
    const today = now.toISOString().split('T')[0];
    const currentTime = now.toTimeString().split(':').slice(0, 2).join(':');
    
    document.getElementById('arrivalDate').value = today;
    document.getElementById('arrivalTime').value = currentTime;
    
    // Set pickup time to 2 hours later
    const pickup = new Date(now.getTime() + 2 * 60 * 60 * 1000);
    document.getElementById('pickupDate').value = pickup.toISOString().split('T')[0];
    document.getElementById('pickupTime').value = pickup.toTimeString().split(':').slice(0, 2).join(':');
}

// Real-time clock with enhanced display
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { 
        hour12: true, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    });
    const dateStr = now.toLocaleDateString('en-IN', { 
        weekday: 'short',
        day: '2-digit',
        month: 'short',
        year: 'numeric'
    });
    
    const clockElement = document.getElementById('currentClock');
    if (clockElement) {
        clockElement.innerHTML = `${dateStr}<br>${timeStr}`;
    }
}

// Enhanced form validation
function validateForm() {
    const vehicleNumber = document.getElementById('vehicleNumber').value.trim();
    const arrivalDate = document.getElementById('arrivalDate').value;
    const arrivalTime = document.getElementById('arrivalTime').value;
    const pickupDate = document.getElementById('pickupDate').value;
    const pickupTime = document.getElementById('pickupTime').value;

    const errors = [];

    if (!vehicleNumber) {
        errors.push('Vehicle number is required');
    } else if (vehicleNumber.length < 4) {
        errors.push('Vehicle number must be at least 4 characters');
    } else if (!/^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$/i.test(vehicleNumber)) {
        // Check for Indian vehicle number format (flexible)
        if (!/^[A-Z0-9]+$/i.test(vehicleNumber)) {
            errors.push('Vehicle number should contain only letters and numbers');
        }
    }

    if (!arrivalDate) errors.push('Arrival date is required');
    if (!arrivalTime) errors.push('Arrival time is required');
    if (!pickupDate) errors.push('Pickup date is required');
    if (!pickupTime) errors.push('Pickup time is required');

    if (arrivalDate && pickupDate && arrivalTime && pickupTime) {
        const arrival = new Date(arrivalDate + 'T' + arrivalTime);
        const pickup = new Date(pickupDate + 'T' + pickupTime);
        
        if (pickup <= arrival) {
            errors.push('Pickup time must be after arrival time');
        }
        
        // Check if parking duration is reasonable (max 72 hours)
        const diffHours = (pickup - arrival) / (1000 * 60 * 60);
        if (diffHours > 72) {
            errors.push('Maximum parking duration is 72 hours');
        }
    }

    return errors;
}

// Utility functions
function formatDate(dateStr) {
    const date = new Date(dateStr);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = String(date.getFullYear()).substring(2);
    return `${day}-${month}-${year}`;
}

function formatDateFromJS(dateObj) {
    const day = String(dateObj.getDate()).padStart(2, '0');
    const month = String(dateObj.getMonth() + 1).padStart(2, '0');
    const year = String(dateObj.getFullYear()).substring(2);
    return `${day}-${month}-${year}`;
}

function formatDateForComparison(dateObj) {
    const day = String(dateObj.getDate()).padStart(2, '0');
    const month = String(dateObj.getMonth() + 1).padStart(2, '0');
    const year = dateObj.getFullYear();
    return `${day}-${month}-${year}`;
}

function parseDateTime(dateStr, timeStr) {
    const [day, month, year] = dateStr.split('-');
    const fullYear = '20' + year;
    return new Date(`${fullYear}-${month}-${day}T${timeStr}`);
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('show');
}

function showNotification(message, type) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    notification.classList.add('show');

    setTimeout(() => {
        notification.classList.remove('show');
    }, 4000);
}

// Enhanced keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Only trigger shortcuts if not typing in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    if (e.ctrlKey) {
        switch(e.key.toLowerCase()) {
            case 't':
                e.preventDefault();
                toggleTheme();
                break;
            case 'p':
                e.preventDefault();
                document.getElementById('vehicleNumber').focus();
                showNotification('Ready to park vehicle - Enter vehicle details', 'success');
                break;
            case 'r':
                e.preventDefault();
                showRemoveModal();
                break;
            case 'f':
                e.preventDefault();
                searchVehicle();
                break;
            case 'h':
                e.preventDefault();
                showHolidayCalendar();
                break;
            case 'g':
                e.preventDefault();
                generateReport();
                break;
            case 'e':
                e.preventDefault();
                exportData();
                break;
            case 'a':
                e.preventDefault();
                showAutoModeModal();
                break;
        }
    }
    
    // ESC to close modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => modal.classList.remove('show'));
    }
});

// Initialize everything
document.addEventListener('DOMContentLoaded', function() {
    try {
        loadTheme();
        initializeParkingData();
        generateParkingGrid();
        setCurrentDateTime();
        
        // Update clock every second
        updateClock();
        setInterval(updateClock, 1000);
        
        // Auto-update current date/time every minute for new entries
        setInterval(() => {
            if (!document.getElementById('vehicleNumber').value) {
                setCurrentDateTime();
            }
        }, 60000);
        
        // Enhanced vehicle number input formatting
        const vehicleNumberInput = document.getElementById('vehicleNumber');
        vehicleNumberInput.addEventListener('input', function(e) {
            let value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
            e.target.value = value;
        });
        
        // Show welcome message
        setTimeout(() => {
            showNotification('🚗 Vehicle Vacancy Vault loaded! Press Ctrl+H for help', 'success');
        }, 1000);
        
        console.log('Parking system initialized with', parkingData.length, 'slots and', holidayData.length, 'holidays');
        
    } catch (error) {
        console.error('Initialization error:', error);
        showNotification('Error initializing system', 'error');
    }
});

// Close modals when clicking outside
window.addEventListener('click', function(e) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });
});

// Auto-save to localStorage periodically
setInterval(function() {
    try {
        localStorage.setItem('parkingBackup', JSON.stringify({
            timestamp: new Date().toISOString(),
            data: parkingData
        }));
    } catch (e) {
        // Silently handle localStorage errors
    }
}, 300000); // Every 5 minutes

// Load from localStorage on startup
function loadBackup() {
    try {
        const backup = localStorage.getItem('parkingBackup');
        if (backup) {
            const parsedBackup = JSON.parse(backup);
            const backupDate = new Date(parsedBackup.timestamp);
            const now = new Date();
            const hoursDiff = (now - backupDate) / (1000 * 60 * 60);
            
            if (hoursDiff < 24 && parsedBackup.data.length > 0) {
                const hasData = parsedBackup.data.some(slot => slot.vehicleType || slot.isReserved);
                if (hasData && confirm('Found recent parking data backup. Would you like to restore it?')) {
                    parkingData = parsedBackup.data;
                    generateParkingGrid();
                    showNotification('Backup data restored', 'success');
                }
            }
        }
    } catch (e) {
        // Silently handle errors
    }
}

// Load backup after initialization
setTimeout(loadBackup, 2000);

// Context menu functionality
let contextMenu = null;

// Add to your generateParkingGrid function, inside the forEach loop:
slotElement.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    
    if (slot.isReserved && !slot.vehicleType) {
        showContextMenu(e, slot);
    }
});

function showContextMenu(event, slot) {
    // Remove existing context menu
    if (contextMenu) {
        contextMenu.remove();
    }
    
    contextMenu = document.createElement('div');
    contextMenu.className = 'context-menu';
    contextMenu.style.left = event.pageX + 'px';
    contextMenu.style.top = event.pageY + 'px';
    
    contextMenu.innerHTML = `
        <div class="context-menu-item" onclick="convertReservedToParked(${slot.slot})">
            <i class="fas fa-car"></i>
            Convert to Parked
        </div>
        <div class="context-menu-item" onclick="cancelReservation(${slot.slot})">
            <i class="fas fa-times"></i>
            Cancel Reservation
        </div>
        <div class="context-menu-item" onclick="viewReservationDetails(${slot.slot})">
            <i class="fas fa-info-circle"></i>
            View Details
        </div>
    `;
    
    document.body.appendChild(contextMenu);
    
    // Remove context menu when clicking elsewhere
    setTimeout(() => {
        document.addEventListener('click', () => {
            if (contextMenu) {
                contextMenu.remove();
                contextMenu = null;
            }
        }, { once: true });
    }, 100);
}
// Cancel reservation
function cancelReservation(slotNumber) {
    if (confirm('Are you sure you want to cancel this reservation?')) {
        parkingData[slotNumber - 1] = {
            slot: slotNumber,
            vehicleType: null,
            vehicleNumber: null,
            arrivalDate: null,
            arrivalTime: null,
            expectedPickupDate: null,
            expectedPickupTime: null,
            weekday: null,
            charge: 0,
            isReserved: false,
            reservationData: null
        };
        
        generateParkingGrid();
        showNotification(`Reservation for slot ${slotNumber} cancelled`, 'success');
    }
}

// View reservation details
function viewReservationDetails(slotNumber) {
    const slot = parkingData[slotNumber - 1];
    const reservation = slot.reservationData;
    
    if (reservation) {
        alert(`Reservation Details:
        
Customer: ${reservation.customerName}
Vehicle: ${reservation.vehicleType} - ${reservation.vehicleNumber}
Date: ${reservation.date}
Time: ${reservation.time}
Duration: ${reservation.duration} hours`);
    }
}

// Enhanced format date function
function formatDateFromJS(dateObj) {
    const day = String(dateObj.getDate()).padStart(2, '0');
    const month = String(dateObj.getMonth() + 1).padStart(2, '0');
    const year = String(dateObj.getFullYear()).substring(2);
    return `${day}-${month}-${year}`;
}
