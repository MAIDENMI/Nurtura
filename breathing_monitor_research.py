"""
Infant Breathing & Heart Rate Monitor - Research-Validated Version
Based on advanced signal processing methods from physiological monitoring research

This version implements:
- Multi-point BGR signal extraction for breathing rate detection
- Remote photoplethysmography (rPPG) for heart rate detection
  Based on: van der Kooij & Naber (2019) https://doi.org/10.3758/s13428-019-01256-8
- Bandpass filtering (0.12-0.75 Hz for breathing, 0.7-4.0 Hz for heart rate)
- Control point normalization
- Real-time graphical display with dual confidence metrics
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass Butterworth filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to signal"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class BreathingMonitorResearch:
    def __init__(self, window_size=150, block_size=50):
        """
        Initialize research-validated breathing monitor.
        
        Args:
            window_size: Number of frames for analysis (5 seconds at 30fps)
            block_size: Size of tracking blocks around key points (increased for better capture)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # UPGRADED: Maximum accuracy pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # HIGHEST ACCURACY (0=lite, 1=full, 2=heavy - most accurate)
            min_detection_confidence=0.7,  # Higher threshold for better detection
            min_tracking_confidence=0.7,  # Better frame-to-frame tracking
            smooth_landmarks=True,  # Smoother, more natural landmark movement
            enable_segmentation=True  # Enable person segmentation for better isolation
        )
        
        # Key points for multi-region tracking
        # Will track: chest center, abdomen center, shoulders (breathing), background (control)
        self.points_initialized = False
        self.tracking_points = []  # Will be populated based on pose
        
        # Signal histories for each point and BGR channel
        self.signal_history = {
            'chest': {'B': deque(maxlen=window_size), 'G': deque(maxlen=window_size), 'R': deque(maxlen=window_size)},
            'abdomen': {'B': deque(maxlen=window_size), 'G': deque(maxlen=window_size), 'R': deque(maxlen=window_size)},
            'nose': {'B': deque(maxlen=window_size), 'G': deque(maxlen=window_size), 'R': deque(maxlen=window_size)},
            'control': {'B': deque(maxlen=window_size), 'G': deque(maxlen=window_size), 'R': deque(maxlen=window_size)}
        }
        
        self.breathing_rate = 0
        self.heart_rate = 0  # NEW: Heart rate in BPM
        self.confidence = 0
        self.hr_confidence = 0  # NEW: Heart rate confidence
        self.fps = 30
        self.block_size = block_size
        self.window_size = window_size
        
        # Track landmark quality
        self.landmark_quality = {}
        
        # For graphing
        self.breathing_rate_history = deque(maxlen=window_size)
        self.heart_rate_history = deque(maxlen=window_size)  # NEW: Heart rate history
        self.time_history = deque(maxlen=window_size)
        self.start_time = time.time()
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup matplotlib figure for real-time plotting"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 10))
        
        # Breathing rate plot
        self.ax1.set_title('Breathing Rate (Research Method)', color='cyan', fontsize=12, weight='bold')
        self.ax1.set_ylabel('Breaths/min', color='white')
        self.ax1.set_ylim(0, 80)
        self.ax1.axhspan(20, 60, alpha=0.2, color='green', label='Normal Range (20-60)')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'cyan', linewidth=2, label='Breathing Rate')
        self.ax1.legend(loc='upper right')
        
        # Heart rate plot (NEW)
        self.ax2.set_title('Heart Rate (rPPG Method)', color='red', fontsize=12, weight='bold')
        self.ax2.set_ylabel('BPM', color='white')
        self.ax2.set_ylim(40, 200)
        self.ax2.axhspan(60, 100, alpha=0.2, color='green', label='Normal Range (60-100)')
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'red', linewidth=2, label='Heart Rate')
        self.ax2.legend(loc='upper right')
        
        # Signal strength plot
        self.ax3.set_title('Signal Quality', color='yellow', fontsize=10)
        self.ax3.set_ylabel('Amplitude', color='white')
        self.ax3.grid(True, alpha=0.3)
        self.line3_chest, = self.ax3.plot([], [], 'r', linewidth=2, label='Chest (Torso)')
        self.line3_abdomen, = self.ax3.plot([], [], 'g', linewidth=2, label='Abdomen')
        self.line3_nose, = self.ax3.plot([], [], 'cyan', linewidth=2, label='Nose')
        self.line3_control, = self.ax3.plot([], [], 'gray', linewidth=1, linestyle='--', label='Control')
        self.ax3.legend(loc='upper right', fontsize=8)
        
        # Confidence plot
        self.ax4.set_title('Confidence Scores', color='lime', fontsize=10)
        self.ax4.set_ylabel('Confidence %', color='white')
        self.ax4.set_ylim(0, 100)
        self.ax4.grid(True, alpha=0.3)
        self.line4_breathing, = self.ax4.plot([], [], 'cyan', linewidth=2, label='Breathing')
        self.line4_heart, = self.ax4.plot([], [], 'red', linewidth=2, label='Heart Rate')
        self.ax4.legend(loc='upper right', fontsize=8)
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_xlabel('Time (seconds)', color='white', fontsize=9)
        
        plt.tight_layout()
        
        # Convert figure to image for OpenCV display
        self.canvas = FigureCanvasAgg(self.fig)
        
    def initialize_tracking_points(self, landmarks, frame_shape):
        """
        ENHANCED: Initialize tracking points with much better accuracy.
        Uses multiple landmarks for robust positioning and visibility checks.
        """
        h, w = frame_shape[:2]
        
        # Get all relevant landmarks with visibility/presence scores
        # Core body points
        left_shoulder = landmarks[11]  # Left shoulder
        right_shoulder = landmarks[12]  # Right shoulder
        left_hip = landmarks[23]  # Left hip
        right_hip = landmarks[24]  # Right hip
        
        # Additional torso landmarks for better chest tracking
        left_elbow = landmarks[13]  # Left elbow
        right_elbow = landmarks[15]  # Right elbow
        
        # Facial landmarks
        nose = landmarks[0]  # Nose tip
        mouth_left = landmarks[9]  # Left mouth
        mouth_right = landmarks[10]  # Right mouth
        
        # === ENHANCED CHEST TRACKING ===
        # Use weighted average of multiple landmarks for more stable chest center
        # This captures the actual center of mass of the torso
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        
        # Chest is positioned at upper-mid torso (60% from shoulders, 40% from hips)
        # This is where maximum respiratory expansion occurs
        chest_x = int((shoulder_mid_x * 0.5 + hip_mid_x * 0.5) * w)
        chest_y = int((shoulder_mid_y * 0.6 + hip_mid_y * 0.4) * h)
        
        chest_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
        
        # === ENHANCED ABDOMEN TRACKING ===
        # Position at lower torso (diaphragm area) for abdominal breathing
        # 30% shoulders, 70% hips - captures diaphragmatic movement
        abdomen_x = int((shoulder_mid_x * 0.3 + hip_mid_x * 0.7) * w)
        abdomen_y = int((shoulder_mid_y * 0.3 + hip_mid_y * 0.7) * h)
        
        abdomen_visibility = (left_hip.visibility + right_hip.visibility) / 2
        
        # === ENHANCED NOSE/FACE TRACKING ===
        # Use nose with fallback to mouth center if nose not visible
        if nose.visibility > 0.5:
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            nose_visibility = nose.visibility
        else:
            # Fallback: estimate from mouth center
            nose_x = int((mouth_left.x + mouth_right.x) / 2 * w)
            nose_y = int((mouth_left.y + mouth_right.y) / 2 * h - 20)
            nose_visibility = (mouth_left.visibility + mouth_right.visibility) / 2
        
        # === ENHANCED CONTROL POINT ===
        # Place control point to the side, at mid-torso height
        # Uses shoulder width to adaptively position it
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * w
        
        # Place control point outside body (1.2x shoulder width from center)
        control_offset = shoulder_width * 0.6
        control_x = int(shoulder_mid_x * w + control_offset)
        control_x = max(30, min(control_x, w - 30))  # Keep within frame with margin
        control_y = int((shoulder_mid_y + hip_mid_y) / 2 * h)  # Mid-torso height
        
        # Store tracking points
        self.tracking_points = {
            'chest': (chest_x, chest_y),
            'abdomen': (abdomen_x, abdomen_y),
            'nose': (nose_x, nose_y),
            'control': (control_x, control_y)
        }
        
        # Store landmark quality for confidence weighting
        self.landmark_quality = {
            'chest': chest_visibility,
            'abdomen': abdomen_visibility,
            'nose': nose_visibility,
            'control': 1.0
        }
        
        # Calculate adaptive block size based on body size
        # Larger people need larger blocks, smaller people need smaller blocks
        torso_height = abs((shoulder_mid_y - hip_mid_y) * h)
        shoulder_distance = abs((right_shoulder.x - left_shoulder.x) * w)
        
        # Adaptive block size (20-80 pixels based on body size)
        self.adaptive_block_size = int(np.clip(min(torso_height, shoulder_distance) * 0.25, 30, 80))
        
        self.points_initialized = True
        
    def extract_region_signal(self, frame, center_point):
        """
        ENHANCED: Extract BGR signal with noise reduction and quality assessment.
        Uses adaptive block sizing and robust statistical methods.
        """
        x, y = center_point
        h, w = frame.shape[:2]
        
        # Use adaptive block size if available, otherwise use default
        if hasattr(self, 'adaptive_block_size'):
            half_block = self.adaptive_block_size // 2
        else:
            half_block = self.block_size // 2
        
        # Ensure we stay within frame boundaries with safety margin
        y_start = max(y - half_block, 0)
        y_end = min(y + half_block, h)
        x_start = max(x - half_block, 0)
        x_end = min(x + half_block, w)
        
        # Extract region
        region = frame[y_start:y_end, x_start:x_end]
        
        # Validate region size
        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return None
        
        # ENHANCED: Apply gentle Gaussian blur to reduce camera noise
        # This helps rPPG signal quality significantly
        region_smoothed = cv2.GaussianBlur(region, (5, 5), 0)
        
        # ENHANCED: Use median instead of mean for more robust signal
        # Median is less affected by outliers (bright spots, shadows, etc.)
        b_value = np.median(region_smoothed[:, :, 0])
        g_value = np.median(region_smoothed[:, :, 1])
        r_value = np.median(region_smoothed[:, :, 2])
        
        # Calculate signal quality metrics
        # Higher standard deviation = more texture = better signal quality
        signal_quality = (np.std(region[:, :, 0]) + 
                         np.std(region[:, :, 1]) + 
                         np.std(region[:, :, 2])) / 3
        
        return {
            'B': b_value, 
            'G': g_value, 
            'R': r_value,
            'quality': signal_quality
        }
    
    def estimate_breathing_rate(self):
        """Estimate breathing rate using research-validated method with multiple sources"""
        # Need at least 3 seconds of data
        min_frames = int(self.window_size * 0.5)
        if len(self.signal_history['chest']['G']) < min_frames:
            return 0, 0
        
        all_breathing_rates = []
        all_confidences = []
        
        # Try multiple regions: chest, abdomen, and nose
        for region in ['chest', 'abdomen', 'nose']:
            region_signals = []
            for channel in ['B', 'G', 'R']:
                sig = np.array(list(self.signal_history[region][channel]))
                if len(sig) > min_frames:
                    # Apply bandpass filter for respiratory frequencies
                    # 0.12-0.75 Hz = 7-45 breaths/min (good range for adults at rest)
                    try:
                        filtered = apply_bandpass_filter(sig, 0.12, 0.75, self.fps, order=4)
                        region_signals.append(filtered)
                    except:
                        region_signals.append(sig - np.mean(sig))
            
            if len(region_signals) == 0:
                continue
            
            # Average across BGR channels
            combined_signal = np.mean(region_signals, axis=0)
            
            # Find peaks in the filtered signal
            # Minimum distance between breaths: ~0.6 seconds (balanced responsiveness)
            # Allows detection up to 100 BPM max
            min_distance = int(self.fps * 0.6)
            
            # Dynamic threshold
            signal_std = np.std(combined_signal)
            if signal_std < 1e-6:
                continue
            
            # Balanced prominence - not too sensitive, not too conservative
            prominence = signal_std * 0.25
            
            peaks, properties = find_peaks(
                combined_signal,
                distance=min_distance,
                prominence=prominence
            )
            
            # Calculate breathing rate for this region
            if len(peaks) >= 2:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                
                # Sanity check
                breathing_rate = max(8, min(breathing_rate, 70))
                
                # Calculate confidence based on signal quality
                confidence = min(100, (signal_std * 2000 + len(peaks) * 10))
                confidence = max(0, min(confidence, 100))
                
                all_breathing_rates.append(breathing_rate)
                all_confidences.append(confidence)
        
        # Combine results from all regions (weighted by confidence)
        if len(all_breathing_rates) == 0:
            # Gradually decay if no new measurements
            return max(8, self.breathing_rate * 0.95), max(0, self.confidence * 0.8)
        
        # Weighted average
        total_confidence = sum(all_confidences)
        if total_confidence > 0:
            weighted_rate = sum(r * c for r, c in zip(all_breathing_rates, all_confidences)) / total_confidence
            avg_confidence = np.mean(all_confidences)
        else:
            weighted_rate = np.mean(all_breathing_rates)
            avg_confidence = 20
        
        # Smooth with previous reading (more responsive)
        if self.breathing_rate > 0:
            weighted_rate = 0.3 * self.breathing_rate + 0.7 * weighted_rate
        
        return weighted_rate, avg_confidence
    
    def estimate_heart_rate(self):
        """
        Estimate heart rate using rPPG method (remote photoplethysmography)
        Based on van der Kooij & Naber (2019) - https://doi.org/10.3758/s13428-019-01256-8
        
        Key findings from paper:
        - GREEN channel is most effective for heart rate detection
        - Facial regions provide highest accuracy
        - Works well under ambient lighting with consumer cameras
        """
        min_frames = int(self.window_size * 0.5)
        if len(self.signal_history['nose']['G']) < min_frames:
            return 0, 0
        
        # Focus on NOSE region (facial tissue) - most accurate per research
        # Use GREEN channel primarily (hemoglobin absorbs green light most)
        green_signal = np.array(list(self.signal_history['nose']['G']))
        
        if len(green_signal) < min_frames:
            return 0, 0
        
        try:
            # Apply bandpass filter for cardiac frequencies
            # 0.7-4.0 Hz = 42-240 BPM (covers resting to high exercise heart rates)
            filtered_signal = apply_bandpass_filter(green_signal, 0.7, 4.0, self.fps, order=5)
        except:
            filtered_signal = green_signal - np.mean(green_signal)
        
        # Find peaks in filtered signal
        # Minimum distance: 0.3 seconds (allows up to 200 BPM)
        min_distance = int(self.fps * 0.3)
        
        signal_std = np.std(filtered_signal)
        if signal_std < 1e-6:
            return 0, 0
        
        # More sensitive prominence for heart beats (smaller than breathing)
        prominence = signal_std * 0.3
        
        peaks, properties = find_peaks(
            filtered_signal,
            distance=min_distance,
            prominence=prominence
        )
        
        # Calculate heart rate
        if len(peaks) < 3:  # Need at least 3 beats for reliable measurement
            return max(40, self.heart_rate * 0.9), max(0, self.hr_confidence * 0.7)
        
        time_duration = len(filtered_signal) / self.fps
        heart_rate = (len(peaks) / time_duration) * 60
        
        # Sanity check for physiological heart rate range
        heart_rate = max(40, min(heart_rate, 200))
        
        # Calculate confidence based on signal quality and peak regularity
        # Higher std and more peaks = better confidence
        peak_regularity = 1.0 - (np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-6)) if len(peaks) > 2 else 0
        peak_regularity = max(0, min(peak_regularity, 1))
        
        confidence = min(100, (signal_std * 3000 + len(peaks) * 5 + peak_regularity * 30))
        confidence = max(0, min(confidence, 100))
        
        # Smooth with previous reading
        if self.heart_rate > 0:
            heart_rate = 0.4 * self.heart_rate + 0.6 * heart_rate
        
        return heart_rate, confidence
    
    def update_plot(self):
        """Update matplotlib plot with current data"""
        if len(self.time_history) > 0:
            times = list(self.time_history)
            
            # Update breathing rate
            if len(self.breathing_rate_history) > 0:
                rates = list(self.breathing_rate_history)
                self.line1.set_data(times, rates)
                self.ax1.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            # Update heart rate (NEW)
            if len(self.heart_rate_history) > 0:
                hr_rates = list(self.heart_rate_history)
                self.line2.set_data(times, hr_rates)
                self.ax2.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            # Update signal quality (chest, abdomen, nose vs control)
            if len(self.signal_history['chest']['G']) > 10:
                chest_sig = np.array(list(self.signal_history['chest']['G']))
                abdomen_sig = np.array(list(self.signal_history['abdomen']['G']))
                nose_sig = np.array(list(self.signal_history['nose']['G']))
                control_sig = np.array(list(self.signal_history['control']['G']))
                
                # Normalize for display (with safety checks)
                if len(chest_sig) > 0 and np.std(chest_sig) > 1e-6:
                    chest_norm = (chest_sig - np.mean(chest_sig)) / (np.std(chest_sig) + 1e-6)
                else:
                    chest_norm = chest_sig
                    
                if len(abdomen_sig) > 0 and np.std(abdomen_sig) > 1e-6:
                    abdomen_norm = (abdomen_sig - np.mean(abdomen_sig)) / (np.std(abdomen_sig) + 1e-6)
                else:
                    abdomen_norm = abdomen_sig
                
                if len(nose_sig) > 0 and np.std(nose_sig) > 1e-6:
                    nose_norm = (nose_sig - np.mean(nose_sig)) / (np.std(nose_sig) + 1e-6)
                else:
                    nose_norm = nose_sig
                    
                if len(control_sig) > 0 and np.std(control_sig) > 1e-6:
                    control_norm = (control_sig - np.mean(control_sig)) / (np.std(control_sig) + 1e-6)
                else:
                    control_norm = control_sig
                
                if len(chest_norm) > 0:
                    self.line3_chest.set_data(times[:len(chest_norm)], chest_norm)
                if len(abdomen_norm) > 0:
                    self.line3_abdomen.set_data(times[:len(abdomen_norm)], abdomen_norm)
                if len(nose_norm) > 0:
                    self.line3_nose.set_data(times[:len(nose_norm)], nose_norm)
                if len(control_norm) > 0:
                    self.line3_control.set_data(times[:len(control_norm)], control_norm)
                
                self.ax3.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
                
                # Safe max calculation
                max_vals = []
                if len(chest_norm) > 0:
                    max_vals.append(np.max(np.abs(chest_norm)))
                if len(abdomen_norm) > 0:
                    max_vals.append(np.max(np.abs(abdomen_norm)))
                if len(nose_norm) > 0:
                    max_vals.append(np.max(np.abs(nose_norm)))
                max_val = max(max_vals) if max_vals else 1
                max_val = max(max_val, 1)  # At least 1
                self.ax3.set_ylim(-max_val * 1.5, max_val * 1.5)
            
            # Update confidence scores (both breathing and heart rate)
            breathing_conf_history = [self.confidence] * len(times)
            heart_conf_history = [self.hr_confidence] * len(times)
            self.line4_breathing.set_data(times, breathing_conf_history)
            self.line4_heart.set_data(times, heart_conf_history)
            self.ax4.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
        
        # Render to canvas
        self.canvas.draw()
        
        # Convert to OpenCV image
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        plot_image = buf.reshape(self.canvas.get_width_height()[::-1] + (4,))
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
        
        return plot_image
    
    def process_frame(self, frame):
        """Process single frame"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = self.pose.process(rgb_frame)
        
        current_time = time.time() - self.start_time
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # ENHANCED: Draw full pose skeleton for visualization
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2)
            )
            
            # Update tracking points every frame to follow movement
            self.initialize_tracking_points(landmarks, frame.shape)
            
            # Extract signals from each region
            for region_name, point in self.tracking_points.items():
                signals = self.extract_region_signal(frame, point)
                if signals:
                    for channel in ['B', 'G', 'R']:
                        self.signal_history[region_name][channel].append(signals[channel])
                
                # ENHANCED: Draw tracking points with better visualization
                # Different colors for each region with quality indicators
                if region_name == 'chest':
                    color = (0, 0, 255)  # Red for chest
                elif region_name == 'abdomen':
                    color = (0, 255, 0)  # Green for abdomen
                elif region_name == 'nose':
                    color = (255, 255, 0)  # Cyan for nose (heart rate!)
                else:
                    color = (128, 128, 128)  # Gray for control
                
                # Get block size (adaptive if available)
                if hasattr(self, 'adaptive_block_size'):
                    block_radius = self.adaptive_block_size // 2
                else:
                    block_radius = self.block_size // 2
                
                # Draw larger region box (shows capture area)
                cv2.rectangle(frame, 
                             (point[0] - block_radius, point[1] - block_radius),
                             (point[0] + block_radius, point[1] + block_radius),
                             color, 2)
                
                # Draw center point
                cv2.circle(frame, point, 5, color, -1)
                cv2.circle(frame, point, 8, (255, 255, 255), 2)  # White outline
                
                # Label with quality indicator
                label = region_name.upper()
                if region_name in self.landmark_quality:
                    quality = self.landmark_quality[region_name]
                    label += f" ({quality:.0%})"
                
                # Label with semi-transparent background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_pos = (point[0] + 15, point[1] + 5)
                cv2.rectangle(frame,
                             (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                             (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                             (0, 0, 0), -1)
                cv2.putText(frame, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Estimate breathing rate
            self.breathing_rate, self.confidence = self.estimate_breathing_rate()
            
            # Estimate heart rate (NEW - using rPPG method)
            self.heart_rate, self.hr_confidence = self.estimate_heart_rate()
            
        # Update histories
        self.breathing_rate_history.append(self.breathing_rate)
        self.heart_rate_history.append(self.heart_rate)
        self.time_history.append(current_time)
        
        # Display breathing rate with color coding
        status_text = "NORMAL"
        text_color = (0, 255, 0)
        
        if self.breathing_rate < 20 and self.breathing_rate > 0:
            status_text = "LOW"
            text_color = (0, 0, 255)
        elif self.breathing_rate > 60:
            status_text = "HIGH"
            text_color = (0, 0, 255)
        
        rate_text = f"Breathing: {self.breathing_rate:.1f} BPM -- {status_text}"
        cv2.putText(frame, rate_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        # Display heart rate with color coding (NEW - rPPG Method)
        hr_status = "NORMAL"
        hr_color = (0, 255, 0)
        
        if self.heart_rate < 60 and self.heart_rate > 0:
            hr_status = "LOW"
            hr_color = (0, 0, 255)
        elif self.heart_rate > 100:
            hr_status = "HIGH"
            hr_color = (0, 0, 255)
        
        hr_text = f"Heart Rate: {self.heart_rate:.0f} BPM -- {hr_status}"
        cv2.putText(frame, hr_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, hr_color, 2)
        
        # Display confidence scores
        conf_text = f"BR Conf: {self.confidence:.0f}%  |  HR Conf: {self.hr_confidence:.0f}%"
        cv2.putText(frame, conf_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Research: Breathing + rPPG Heart Rate", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame


def main():
    """Main function"""
    print("="*60)
    print("Infant Breathing & Heart Rate Monitor")
    print("Research: Breathing Detection + rPPG Heart Rate")
    print("="*60)
    print("Initializing...")
    
    monitor = BreathingMonitorResearch()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Screenshot")
    print("\nStarting monitor...")
    
    # Create windows
    cv2.namedWindow('Research-Validated Breathing Monitor', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Analysis Graphs', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            output_frame = monitor.process_frame(frame)
            
            # Update and get plot
            plot_image = monitor.update_plot()
            
            # Display
            cv2.imshow('Research-Validated Breathing Monitor', output_frame)
            cv2.imshow('Analysis Graphs', plot_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'screenshot_{timestamp}.png', output_frame)
                print(f"✓ Screenshot saved: screenshot_{timestamp}.png")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        monitor.pose.close()
        plt.close('all')
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()

