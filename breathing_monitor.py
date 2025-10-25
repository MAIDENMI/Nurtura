import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time

class BreathingMonitor:
    def __init__(self, window_size=30, breathing_threshold=0.02):
        """
        Initialize breathing monitor.
        
        Args:
            window_size: Number of frames for moving average
            breathing_threshold: Sensitivity for motion detection
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lightweight for Raspberry Pi
            min_detection_confidence=0.5
        )
        
        self.motion_history = deque(maxlen=window_size)
        self.breathing_threshold = breathing_threshold
        self.fps = 30
        self.breathing_rate = 0
        
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
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        torso = frame[y_min:y_max, x_min:x_max]
        
        # Validate torso is not empty
        if torso.size == 0 or torso.shape[0] == 0 or torso.shape[1] == 0:
            return None, None
            
        return torso, (x_min, y_min, x_max, y_max)
    
    def measure_motion(self, torso_prev, torso_curr):
        """
        Measure motion between consecutive torso frames.
        Uses optical flow magnitude as breathing indicator.
        """
        if torso_prev is None or torso_curr is None:
            return 0
        
        # Check if torso regions are valid (not empty)
        if torso_prev.size == 0 or torso_curr.size == 0:
            return 0
        
        # Resize for consistency
        h, w = 100, 100
        torso_prev_resized = cv2.resize(torso_prev, (w, h))
        torso_curr_resized = cv2.resize(torso_curr, (w, h))
        
        # Convert to grayscale
        gray_prev = cv2.cvtColor(torso_prev_resized, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(torso_curr_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
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
    
    def process_frame(self, frame):
        """
        Process single frame and return results.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return frame, None
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract torso
        torso, bbox = self.extract_torso(frame, landmarks)
        
        # Skip if torso extraction failed
        if torso is None or bbox is None:
            return frame, None
        
        # Measure motion
        if not hasattr(self, 'torso_prev'):
            self.torso_prev = torso
        
        motion = self.measure_motion(self.torso_prev, torso)
        self.motion_history.append(motion)
        self.torso_prev = torso
        
        # Estimate breathing rate
        self.breathing_rate = self.estimate_breathing_rate()
        
        # Draw torso bounding box
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display breathing rate
        text = f"Breathing Rate: {self.breathing_rate:.1f} breaths/min"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, self.breathing_rate


def main():
    """Main loop for real-time breathing monitoring."""
    monitor = BreathingMonitor()
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for Raspberry Pi optimization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting breathing monitor. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for selfie-view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            output_frame, breathing_rate = monitor.process_frame(frame)
            
            # Display
            cv2.imshow('Infant Breathing Monitor', output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.pose.close()


if __name__ == "__main__":
    main()

