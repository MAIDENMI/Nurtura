"""
Infant Breathing Monitor with Real-Time Graphical Display
Shows breathing rate and motion signal graphs similar to heart rate monitors
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
from scipy import signal
from scipy.signal import find_peaks


class BreathingMonitorGraphical:
    def __init__(self, window_size=75, breathing_threshold=0.03):
        """
        Initialize breathing monitor with graphical display.
        
        Args:
            window_size: Number of frames for moving average (increased for better accuracy)
            breathing_threshold: Sensitivity for motion detection (increased to reduce false positives)
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lightweight for performance
            min_detection_confidence=0.5
        )
        
        self.motion_history = deque(maxlen=window_size)
        self.breathing_rate_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        
        self.breathing_threshold = breathing_threshold
        self.fps = 30
        self.breathing_rate = 0
        self.torso_prev = None
        
        self.start_time = time.time()
        
        # Setup matplotlib figure
        self.setup_plot()
        
    def setup_plot(self):
        """Setup matplotlib figure for real-time plotting"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.patch.set_facecolor('#1a1a1a')
        
        # Breathing Rate plot
        self.ax1.set_title('Breathing Rate (breaths/min)', color='#00ff00', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Time (seconds)', color='white')
        self.ax1.set_ylabel('Breaths/min', color='white')
        self.ax1.grid(True, alpha=0.3, color='#444444')
        self.ax1.set_facecolor('#0a0a0a')
        self.line1, = self.ax1.plot([], [], color='#00ff00', linewidth=2, label='Breathing Rate')
        
        # Normal range bands
        self.ax1.axhspan(20, 60, alpha=0.1, color='green', label='Normal Range (20-60)')
        self.ax1.axhspan(0, 20, alpha=0.1, color='red', label='Low')
        self.ax1.axhspan(60, 100, alpha=0.1, color='red', label='High')
        
        self.ax1.legend(loc='upper left', fontsize=8)
        self.ax1.set_ylim(0, 80)
        
        # Motion Signal plot (waveform)
        self.ax2.set_title('Motion Signal (Breathing Waveform)', color='#00ffff', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Time (seconds)', color='white')
        self.ax2.set_ylabel('Motion Intensity', color='white')
        self.ax2.grid(True, alpha=0.3, color='#444444')
        self.ax2.set_facecolor('#0a0a0a')
        self.line2, = self.ax2.plot([], [], color='#00ffff', linewidth=1.5)
        self.ax2.axhline(y=self.breathing_threshold, color='red', linestyle='--', 
                        linewidth=1, alpha=0.5, label='Threshold')
        self.ax2.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
    def extract_torso(self, frame, landmarks):
        """Extract torso region from frame using pose landmarks"""
        h, w = frame.shape[:2]
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
        y_max = int(max(left_hip.y, right_hip.y) * h)
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        if x_max > x_min and y_max > y_min:
            torso = frame[y_min:y_max, x_min:x_max]
            return torso, (x_min, y_min, x_max, y_max)
        return None, None
    
    def measure_motion(self, torso_prev, torso_curr):
        """Measure motion between consecutive torso frames"""
        if torso_prev is None or torso_curr is None:
            return 0
        
        try:
            torso_prev_resized = cv2.resize(torso_prev, (100, 100))
            torso_curr_resized = cv2.resize(torso_curr, (100, 100))
            
            gray_prev = cv2.cvtColor(torso_prev_resized, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(torso_curr_resized, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_curr,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion = np.mean(mag)
            
            return motion
        except:
            return 0
    
    def estimate_breathing_rate(self):
        """Estimate breathing rate from motion history with improved signal processing"""
        if len(self.motion_history) < 30:
            return 0
        
        motion_array = np.array(list(self.motion_history))
        
        # Step 1: Smooth the signal with moving average to remove noise
        window_size = 5
        if len(motion_array) >= window_size:
            motion_smoothed = np.convolve(motion_array, np.ones(window_size)/window_size, mode='valid')
        else:
            motion_smoothed = motion_array
        
        # Step 2: Apply bandpass filter for typical breathing frequencies
        # Normal breathing: 0.2-0.8 Hz (12-48 breaths/min)
        # Infant breathing: 0.33-1.0 Hz (20-60 breaths/min)
        try:
            nyquist = self.fps / 2
            low_freq = 0.15 / nyquist  # 9 breaths/min
            high_freq = 1.2 / nyquist  # 72 breaths/min
            
            if len(motion_smoothed) > 20:
                sos = signal.butter(2, [low_freq, high_freq], btype='band', output='sos')
                motion_filtered = signal.sosfilt(sos, motion_smoothed)
            else:
                motion_filtered = motion_smoothed
        except:
            motion_filtered = motion_smoothed
        
        # Step 3: Find peaks with minimum distance and prominence
        # Minimum distance between breaths: ~1 second (fps frames)
        min_distance = int(self.fps * 0.6)  # At least 0.6 seconds between breaths (more responsive)
        
        # Dynamic threshold based on signal strength
        signal_std = np.std(motion_filtered)
        signal_mean = np.mean(motion_filtered)
        
        # Lower prominence for more sensitivity
        prominence = max(signal_std * 0.3, self.breathing_threshold * 0.5)
        
        peaks, properties = find_peaks(
            motion_filtered, 
            distance=min_distance,
            prominence=prominence
        )
        
        # Step 4: Calculate breathing rate
        if len(peaks) < 1:
            # If no peaks, return a small value instead of keeping old rate
            return max(5, self.breathing_rate * 0.9) if hasattr(self, 'breathing_rate') else 0
        
        # Use the number of peaks over the time window
        frames_duration = len(self.motion_history) / self.fps
        breaths_per_second = len(peaks) / frames_duration
        breathing_rate = breaths_per_second * 60
        
        # Step 5: Sanity check - cap at reasonable values
        breathing_rate = max(5, min(breathing_rate, 80))
        
        # Step 6: Smooth the breathing rate itself (more responsive now)
        if hasattr(self, 'breathing_rate') and self.breathing_rate > 0:
            # More responsive: 50/50 blend instead of 70/30
            breathing_rate = 0.5 * self.breathing_rate + 0.5 * breathing_rate
        
        return breathing_rate
    
    def update_plot(self):
        """Update the matplotlib plot with current data"""
        if len(self.time_history) > 0:
            times = list(self.time_history)
            
            # Update breathing rate plot
            if len(self.breathing_rate_history) > 0:
                rates = list(self.breathing_rate_history)
                self.line1.set_data(times, rates)
                self.ax1.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            # Update motion signal plot
            if len(self.motion_history) > 0:
                motions = list(self.motion_history)
                self.line2.set_data(times, motions)
                self.ax2.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
                
                # Auto-scale y-axis for motion
                max_motion = max(motions) if motions else 1
                self.ax2.set_ylim(0, max(max_motion * 1.2, self.breathing_threshold * 2))
        
        # Convert plot to image (macOS Retina compatible)
        self.fig.canvas.draw()
        
        # Get the buffer and its actual dimensions
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        
        # Calculate actual dimensions from buffer size (handles Retina displays)
        w, h = self.fig.canvas.get_width_height()
        # On Retina displays, buffer might be 2x the reported size
        actual_pixels = len(buf) // 4  # RGBA = 4 bytes per pixel
        actual_h = int(np.sqrt(actual_pixels * h / w))
        actual_w = actual_pixels // actual_h
        
        buf = buf.reshape((actual_h, actual_w, 4))
        
        # Convert RGBA to BGR for OpenCV
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    
    def process_frame(self, frame):
        """Process single frame and return results"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = self.pose.process(rgb_frame)
        
        current_time = time.time() - self.start_time
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract torso
            torso, bbox = self.extract_torso(frame, landmarks)
            
            if torso is not None:
                # Measure motion
                motion = self.measure_motion(self.torso_prev, torso)
                self.motion_history.append(motion)
                self.torso_prev = torso
                
                # Estimate breathing rate
                self.breathing_rate = self.estimate_breathing_rate()
                
                # Update histories
                self.breathing_rate_history.append(self.breathing_rate)
                self.time_history.append(current_time)
                
                # Draw torso bounding box
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        else:
            # No pose detected
            self.motion_history.append(0)
            self.breathing_rate_history.append(self.breathing_rate)
            self.time_history.append(current_time)
        
        # Display breathing rate with color coding
        if self.breathing_rate < 20 and self.breathing_rate > 0:
            color = (0, 0, 255)  # Red - LOW
            status = "LOW"
        elif self.breathing_rate > 60:
            color = (0, 0, 255)  # Red - HIGH
            status = "HIGH"
        elif self.breathing_rate > 0:
            color = (0, 255, 0)  # Green - NORMAL
            status = "NORMAL"
        else:
            color = (255, 255, 255)  # White - DETECTING
            status = "DETECTING"
        
        text = f"Breathing Rate: {self.breathing_rate:.1f} BPM - {status}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add motion indicator
        motion_val = self.motion_history[-1] if self.motion_history else 0
        cv2.putText(frame, f"Motion: {motion_val:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, self.breathing_rate


def main():
    """Main loop for real-time breathing monitoring with graphs"""
    print("=" * 60)
    print("Infant Breathing Monitor - Graphical Version")
    print("=" * 60)
    print("\nInitializing...")
    
    monitor = BreathingMonitorGraphical(window_size=60)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Screenshot")
    print("\nStarting monitor...\n")
    
    screenshot_count = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for selfie-view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            output_frame, breathing_rate = monitor.process_frame(frame)
            
            # Update plot every 5 frames (for performance)
            if frame_count % 5 == 0:
                plot_image = monitor.update_plot()
                
                # Resize plot to fit nicely
                plot_height = 400
                plot_aspect = plot_image.shape[1] / plot_image.shape[0]
                plot_width = int(plot_height * plot_aspect)
                plot_resized = cv2.resize(plot_image, (plot_width, plot_height))
                
                # Display graphs
                cv2.imshow('Breathing Graphs', plot_resized)
            
            # Display video
            cv2.imshow('Infant Breathing Monitor', output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                cv2.imwrite(f"screenshot_{screenshot_count}.jpg", output_frame)
                cv2.imwrite(f"graph_{screenshot_count}.jpg", plot_resized)
                print(f"✓ Screenshots saved: screenshot_{screenshot_count}.jpg & graph_{screenshot_count}.jpg")
            
            frame_count += 1
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.pose.close()
        plt.close('all')
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()

