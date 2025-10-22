"""
Advanced Infant Breathing Monitor with Configuration Support
Uses config.py for easy parameter adjustment
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import csv
from datetime import datetime
import config


class BreathingMonitor:
    def __init__(self):
        """
        Initialize breathing monitor using settings from config.py
        """
        # Validate configuration
        config.validate_config()
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Initialize motion tracking
        self.motion_history = deque(maxlen=config.WINDOW_SIZE)
        self.breathing_threshold = config.BREATHING_THRESHOLD
        self.fps = config.CAMERA_FPS
        self.breathing_rate = 0
        self.torso_prev = None
        
        # FPS calculation
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Logging setup
        if config.ENABLE_LOGGING:
            self.init_logging()
        
        print(f"✓ Breathing Monitor initialized")
        print(f"  Window size: {config.WINDOW_SIZE} frames")
        print(f"  Threshold: {config.BREATHING_THRESHOLD}")
        
    def init_logging(self):
        """Initialize CSV logging"""
        self.log_file = open(config.LOG_FILE, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp', 'breathing_rate', 'motion_avg', 
            'pose_detected', 'fps'
        ])
        self.last_log_time = time.time()
        print(f"✓ Logging enabled: {config.LOG_FILE}")
        
    def extract_torso(self, frame, landmarks):
        """
        Extract torso region from frame using pose landmarks.
        Returns cropped torso region and bounding box.
        """
        h, w = frame.shape[:2]
        
        # Key landmarks for torso (in MediaPipe Pose)
        # 11: left shoulder, 12: right shoulder
        # 23: left hip, 24: right hip
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Get bounding box
        x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
        y_max = int(max(left_hip.y, right_hip.y) * h)
        
        # Add padding
        padding = config.TORSO_PADDING
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract torso region
        if x_max > x_min and y_max > y_min:
            torso = frame[y_min:y_max, x_min:x_max]
            return torso, (x_min, y_min, x_max, y_max)
        else:
            return None, None
    
    def measure_motion(self, torso_prev, torso_curr):
        """
        Measure motion between consecutive torso frames.
        Uses optical flow magnitude as breathing indicator.
        """
        if torso_prev is None or torso_curr is None:
            return 0
        
        # Resize for consistency
        w, h = config.TORSO_RESIZE
        try:
            torso_prev_resized = cv2.resize(torso_prev, (w, h))
            torso_curr_resized = cv2.resize(torso_curr, (w, h))
        except:
            return 0
        
        # Convert to grayscale
        gray_prev = cv2.cvtColor(torso_prev_resized, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(torso_curr_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr,
            None, 
            config.FLOW_PYR_SCALE,
            config.FLOW_LEVELS,
            config.FLOW_WINSIZE,
            config.FLOW_ITERATIONS,
            config.FLOW_POLY_N,
            config.FLOW_POLY_SIGMA,
            0
        )
        
        # Magnitude of motion
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = np.mean(mag)
        
        return motion
    
    def estimate_breathing_rate(self):
        """
        Estimate breathing rate from motion history.
        Uses simple peak detection on smoothed motion signal.
        """
        if len(self.motion_history) < 10:
            return 0
        
        motion_array = np.array(list(self.motion_history))
        
        # Simple peak detection: find local maxima
        peaks = 0
        for i in range(1, len(motion_array) - 1):
            if (motion_array[i] > motion_array[i-1] and 
                motion_array[i] > motion_array[i+1] and
                motion_array[i] > self.breathing_threshold):
                peaks += 1
        
        # Convert peaks to breaths per minute
        # Each breath = 2 peaks (inhale + exhale)
        frames_duration = len(self.motion_history) / self.fps
        breaths = peaks / 2
        breathing_rate = (breaths / frames_duration) * 60
        
        return max(0, breathing_rate)
    
    def check_alert(self, breathing_rate):
        """Check if breathing rate is outside normal range"""
        if not config.ENABLE_ALERTS:
            return None
            
        if breathing_rate < config.MIN_NORMAL_BREATHING_RATE and breathing_rate > 0:
            return "LOW"
        elif breathing_rate > config.MAX_NORMAL_BREATHING_RATE:
            return "HIGH"
        return None
    
    def update_fps(self):
        """Calculate actual FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def log_data(self, breathing_rate, motion_avg, pose_detected, fps):
        """Log data to CSV file"""
        if not config.ENABLE_LOGGING:
            return
            
        current_time = time.time()
        if current_time - self.last_log_time >= config.LOG_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.csv_writer.writerow([
                timestamp, breathing_rate, motion_avg,
                pose_detected, fps
            ])
            self.log_file.flush()
            self.last_log_time = current_time
    
    def process_frame(self, frame):
        """
        Process single frame and return results.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = self.pose.process(rgb_frame)
        
        pose_detected = False
        if results.pose_landmarks:
            pose_detected = True
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
                
                # Draw torso bounding box
                if config.SHOW_TORSO_BOX and bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                config.BOX_COLOR, config.BOX_THICKNESS)
        
        # Display breathing rate
        text = f"Breathing Rate: {self.breathing_rate:.1f} breaths/min"
        cv2.putText(frame, text, (10, 30),
                    config.TEXT_FONT, config.TEXT_SCALE, 
                    config.TEXT_COLOR, config.TEXT_THICKNESS)
        
        # Check for alerts
        alert = self.check_alert(self.breathing_rate)
        if alert:
            alert_text = f"ALERT: {alert} BREATHING RATE"
            color = (0, 0, 255)  # Red
            cv2.putText(frame, alert_text, (10, 70),
                       config.TEXT_FONT, config.TEXT_SCALE, 
                       color, config.TEXT_THICKNESS)
        
        # Display FPS
        if config.SHOW_FPS:
            fps = self.update_fps()
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, h - 20),
                       config.TEXT_FONT, 0.5, 
                       config.TEXT_COLOR, 1)
        
        # Log data
        motion_avg = np.mean(self.motion_history) if self.motion_history else 0
        self.log_data(self.breathing_rate, motion_avg, pose_detected, fps)
        
        return frame, self.breathing_rate
    
    def cleanup(self):
        """Clean up resources"""
        self.pose.close()
        if config.ENABLE_LOGGING:
            self.log_file.close()
            print(f"✓ Log file saved: {config.LOG_FILE}")


def main():
    """Main loop for real-time breathing monitoring."""
    print("=" * 60)
    print("Infant Breathing Monitor - Advanced Version")
    print("=" * 60)
    
    monitor = BreathingMonitor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open camera {config.CAMERA_INDEX}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    
    print(f"✓ Camera opened: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Screenshot")
    print("\nStarting monitor...\n")
    
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Error reading frame")
                break
            
            # Flip for selfie-view
            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            
            # Process frame
            output_frame, breathing_rate = monitor.process_frame(frame)
            
            # Display
            cv2.imshow(config.WINDOW_NAME, output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"✓ Screenshot saved: {filename}")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.cleanup()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()

