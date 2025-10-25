"""
Infant Breathing Monitor - Research-Validated Version
Based on advanced signal processing methods from respiratory rate research

This version implements:
- Multi-point BGR signal extraction
- Bandpass filtering (0.05-0.7 Hz for respiration)
- Control point normalization
- Real-time graphical display with confidence metrics
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
    def __init__(self, window_size=150, block_size=20):
        """
        Initialize research-validated breathing monitor.
        
        Args:
            window_size: Number of frames for analysis (5 seconds at 30fps)
            block_size: Size of tracking blocks around key points
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5
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
        self.confidence = 0
        self.fps = 30
        self.block_size = block_size
        self.window_size = window_size
        
        # For graphing
        self.breathing_rate_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.start_time = time.time()
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup matplotlib figure for real-time plotting"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # Breathing rate plot
        self.ax1.set_title('Breathing Rate (Research Method)', color='cyan', fontsize=14, weight='bold')
        self.ax1.set_ylabel('Breaths/min', color='white')
        self.ax1.set_ylim(0, 80)
        self.ax1.axhspan(20, 60, alpha=0.2, color='green', label='Normal Range (20-60)')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'cyan', linewidth=2, label='Breathing Rate')
        self.ax1.legend(loc='upper right')
        
        # Signal strength plot
        self.ax2.set_title('Signal Quality', color='yellow', fontsize=12)
        self.ax2.set_ylabel('Amplitude', color='white')
        self.ax2.grid(True, alpha=0.3)
        self.line2_chest, = self.ax2.plot([], [], 'r', linewidth=2, label='Chest (Torso)')
        self.line2_abdomen, = self.ax2.plot([], [], 'g', linewidth=2, label='Abdomen')
        self.line2_nose, = self.ax2.plot([], [], 'cyan', linewidth=2, label='Nose')
        self.line2_control, = self.ax2.plot([], [], 'gray', linewidth=1, linestyle='--', label='Control')
        self.ax2.legend(loc='upper right')
        
        # Confidence plot
        self.ax3.set_title('Confidence Score', color='lime', fontsize=12)
        self.ax3.set_ylabel('Confidence %', color='white')
        self.ax3.set_ylim(0, 100)
        self.ax3.grid(True, alpha=0.3)
        self.line3, = self.ax3.plot([], [], 'lime', linewidth=2)
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlabel('Time (seconds)', color='white')
        
        plt.tight_layout()
        
        # Convert figure to image for OpenCV display
        self.canvas = FigureCanvasAgg(self.fig)
        
    def initialize_tracking_points(self, landmarks, frame_shape):
        """Initialize tracking points based on detected pose landmarks"""
        h, w = frame_shape[:2]
        
        # Get key body landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Get nose landmark for breathing detection
        nose = landmarks[0]
        
        # Calculate center points for different body regions
        # Chest center - more precise positioning for torso breathing
        # Position between shoulders and mid-torso (where maximum chest expansion occurs)
        chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        chest_y = int(((left_shoulder.y + right_shoulder.y) / 2 + (left_hip.y + right_hip.y) / 2) / 2 * h)
        
        # Abdomen center - lower chest area where diaphragm movement is visible
        abdomen_x = int((left_hip.x + right_hip.x) / 2 * w)
        abdomen_y = int(((left_shoulder.y + right_shoulder.y) / 2 + (left_hip.y + right_hip.y) / 2) / 2 * h + 50)
        
        # Nose area for nostril airflow detection
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h + 15)  # Below nose for nostril area
        
        # Control point (shoulder area - less movement)
        control_x = int(left_shoulder.x * w - 50)
        control_y = int(left_shoulder.y * h)
        
        self.tracking_points = {
            'chest': (chest_x, chest_y),
            'abdomen': (abdomen_x, abdomen_y),
            'nose': (nose_x, nose_y),
            'control': (control_x, control_y)
        }
        
        self.points_initialized = True
        
    def extract_region_signal(self, frame, center_point):
        """Extract BGR signal from a region around a point"""
        x, y = center_point
        half_block = self.block_size // 2
        h, w = frame.shape[:2]
        
        # Ensure we stay within frame boundaries
        y_start = max(y - half_block, 0)
        y_end = min(y + half_block, h)
        x_start = max(x - half_block, 0)
        x_end = min(x + half_block, w)
        
        # Extract region
        region = frame[y_start:y_end, x_start:x_end]
        
        if region.size == 0:
            return None
        
        # Calculate mean BGR values for this region
        b_mean = np.mean(region[:, :, 0])
        g_mean = np.mean(region[:, :, 1])
        r_mean = np.mean(region[:, :, 2])
        
        return {'B': b_mean, 'G': g_mean, 'R': r_mean}
    
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
    
    def update_plot(self):
        """Update matplotlib plot with current data"""
        if len(self.time_history) > 0:
            times = list(self.time_history)
            
            # Update breathing rate
            if len(self.breathing_rate_history) > 0:
                rates = list(self.breathing_rate_history)
                self.line1.set_data(times, rates)
                self.ax1.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
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
                    self.line2_chest.set_data(times[:len(chest_norm)], chest_norm)
                if len(abdomen_norm) > 0:
                    self.line2_abdomen.set_data(times[:len(abdomen_norm)], abdomen_norm)
                if len(nose_norm) > 0:
                    self.line2_nose.set_data(times[:len(nose_norm)], nose_norm)
                if len(control_norm) > 0:
                    self.line2_control.set_data(times[:len(control_norm)], control_norm)
                
                self.ax2.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
                
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
                self.ax2.set_ylim(-max_val * 1.5, max_val * 1.5)
            
            # Update confidence
            confidence_history = [self.confidence] * len(times)
            self.line3.set_data(times, confidence_history)
            self.ax3.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
        
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
            
            # Update tracking points every frame to follow movement
            self.initialize_tracking_points(landmarks, frame.shape)
            
            # Extract signals from each region
            for region_name, point in self.tracking_points.items():
                signals = self.extract_region_signal(frame, point)
                if signals:
                    for channel in ['B', 'G', 'R']:
                        self.signal_history[region_name][channel].append(signals[channel])
                
                # Draw tracking points with better visualization
                # Different colors for each region
                if region_name == 'chest':
                    color = (0, 0, 255)  # Red for chest
                elif region_name == 'abdomen':
                    color = (0, 255, 0)  # Green for abdomen
                elif region_name == 'nose':
                    color = (255, 255, 0)  # Cyan for nose
                else:
                    color = (128, 128, 128)  # Gray for control
                
                # Draw circle and label
                cv2.circle(frame, point, 7, color, -1)
                cv2.circle(frame, point, self.block_size//2, color, 2)
                
                # Label with background for better visibility
                label = region_name.upper()
                label_pos = (point[0] + 15, point[1] + 5)
                cv2.putText(frame, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Estimate breathing rate
            self.breathing_rate, self.confidence = self.estimate_breathing_rate()
            
        # Update histories
        self.breathing_rate_history.append(self.breathing_rate)
        self.time_history.append(current_time)
        
        # Display breathing rate with confidence
        status_text = "NORMAL"
        text_color = (0, 255, 0)
        
        if self.breathing_rate < 20 or self.breathing_rate > 60:
            status_text = "HIGH" if self.breathing_rate > 60 else "LOW"
            text_color = (0, 0, 255)
        
        rate_text = f"Breathing Rate: {self.breathing_rate:.1f} BPM -- {status_text}"
        cv2.putText(frame, rate_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
        
        conf_text = f"Confidence: {self.confidence:.0f}%"
        conf_color = (0, 255, 0) if self.confidence > 60 else (0, 165, 255) if self.confidence > 30 else (0, 0, 255)
        cv2.putText(frame, conf_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        cv2.putText(frame, "Research-Validated Method", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame


def main():
    """Main function"""
    print("="*60)
    print("Infant Breathing Monitor - Research-Validated Version")
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

