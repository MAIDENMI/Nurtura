"""
Infant Breathing Monitor - Video File Version
Test the breathing monitor with pre-recorded video files
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import sys
import os


class BreathingMonitorVideo:
    def __init__(self, window_size=60, breathing_threshold=0.02):
        """Initialize breathing monitor for video files"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        
        self.motion_history = deque(maxlen=window_size)
        self.breathing_rate_history = deque(maxlen=100)
        self.breathing_threshold = breathing_threshold
        self.fps = 30
        self.breathing_rate = 0
        self.torso_prev = None
        
        # Statistics
        self.frame_count = 0
        self.pose_detected_count = 0
        
    def extract_torso(self, frame, landmarks):
        """Extract torso region from frame using pose landmarks"""
        h, w = frame.shape[:2]
        
        # Key landmarks for torso
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
        """Estimate breathing rate from motion history"""
        if len(self.motion_history) < 10:
            return 0
        
        motion_array = np.array(list(self.motion_history))
        
        # Simple peak detection
        peaks = 0
        for i in range(1, len(motion_array) - 1):
            if (motion_array[i] > motion_array[i-1] and 
                motion_array[i] > motion_array[i+1] and
                motion_array[i] > self.breathing_threshold):
                peaks += 1
        
        frames_duration = len(self.motion_history) / self.fps
        breaths = peaks / 2
        breathing_rate = (breaths / frames_duration) * 60
        
        return max(0, min(breathing_rate, 100))
    
    def process_frame(self, frame):
        """Process single frame and return results"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = self.pose.process(rgb_frame)
        
        pose_detected = False
        if results.pose_landmarks:
            pose_detected = True
            self.pose_detected_count += 1
            landmarks = results.pose_landmarks.landmark
            
            # Draw pose landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            
            # Extract torso
            torso, bbox = self.extract_torso(frame, landmarks)
            
            if torso is not None:
                # Measure motion
                motion = self.measure_motion(self.torso_prev, torso)
                self.motion_history.append(motion)
                self.torso_prev = torso
                
                # Estimate breathing rate
                self.breathing_rate = self.estimate_breathing_rate()
                self.breathing_rate_history.append(self.breathing_rate)
                
                # Draw torso bounding box
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                    cv2.putText(frame, "TORSO", (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            self.motion_history.append(0)
        
        # Display info
        if self.breathing_rate < 20 and self.breathing_rate > 0:
            color = (0, 0, 255)  # Red
            status = "LOW - ALERT!"
        elif self.breathing_rate > 60:
            color = (0, 0, 255)  # Red
            status = "HIGH - ALERT!"
        elif self.breathing_rate > 0:
            color = (0, 255, 0)  # Green
            status = "NORMAL"
        else:
            color = (255, 255, 255)  # White
            status = "DETECTING..."
        
        # Add semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Display breathing rate
        text = f"Breathing Rate: {self.breathing_rate:.1f} breaths/min"
        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"Status: {status}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show detection stats
        detection_rate = (self.pose_detected_count / self.frame_count * 100) if self.frame_count > 0 else 0
        cv2.putText(frame, f"Pose Detection: {detection_rate:.1f}%", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show motion value
        motion_val = self.motion_history[-1] if self.motion_history else 0
        cv2.putText(frame, f"Motion: {motion_val:.4f}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show average breathing rate if we have history
        if len(self.breathing_rate_history) > 10:
            valid_rates = [r for r in self.breathing_rate_history if r > 0]
            if valid_rates:
                avg_rate = np.mean(valid_rates)
                cv2.putText(frame, f"Average: {avg_rate:.1f} BPM", (10, h - 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame, pose_detected


def process_video(video_path):
    """Process a video file and analyze breathing"""
    print("=" * 60)
    print("Infant Breathing Monitor - Video Analysis")
    print("=" * 60)
    print()
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        print()
        print("Please provide a valid video file path.")
        return
    
    print(f"📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"✓ Video loaded successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f} seconds")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  'q' - Quit")
    print("  'r' - Restart video")
    print("  's' - Screenshot")
    print()
    print("Processing video...")
    print()
    
    monitor = BreathingMonitorVideo(window_size=60)
    monitor.fps = fps if fps > 0 else 30
    
    paused = False
    screenshot_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✓ Reached end of video")
                    print(f"  Total frames processed: {monitor.frame_count}")
                    print(f"  Pose detected in: {monitor.pose_detected_count} frames ({monitor.pose_detected_count/monitor.frame_count*100:.1f}%)")
                    if len(monitor.breathing_rate_history) > 0:
                        valid_rates = [r for r in monitor.breathing_rate_history if r > 0]
                        if valid_rates:
                            print(f"  Average breathing rate: {np.mean(valid_rates):.1f} BPM")
                            print(f"  Min: {np.min(valid_rates):.1f}, Max: {np.max(valid_rates):.1f}")
                    break
                
                # Process frame
                output_frame, pose_detected = monitor.process_frame(frame)
                
                # Add progress bar
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress = current_pos / frame_count
                bar_width = width - 20
                bar_height = 20
                cv2.rectangle(output_frame, (10, height - 40), (10 + int(bar_width * progress), height - 20), (0, 255, 0), -1)
                cv2.rectangle(output_frame, (10, height - 40), (10 + bar_width, height - 20), (255, 255, 255), 2)
                
                cv2.imshow('Infant Breathing Analysis', output_frame)
            else:
                # Still display when paused
                cv2.imshow('Infant Breathing Analysis', output_frame)
            
            # Handle key presses
            key = cv2.waitKey(int(1000/fps) if not paused else 1) & 0xFF
            
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord(' '):  # Space bar
                paused = not paused
                status = "PAUSED" if paused else "PLAYING"
                print(f"▶️  {status}")
            elif key == ord('r'):  # Restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                monitor = BreathingMonitorVideo(window_size=60)
                monitor.fps = fps if fps > 0 else 30
                print("🔄 Video restarted")
            elif key == ord('s'):  # Screenshot
                screenshot_count += 1
                filename = f"video_analysis_{screenshot_count}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"📸 Screenshot saved: {filename}")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.pose.close()
        print("\n✓ Analysis complete")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("=" * 60)
        print("Infant Breathing Monitor - Video Analysis")
        print("=" * 60)
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <video_file>")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} baby_sleeping.mp4")
        print()
        print("Supported formats: .mp4, .avi, .mov, .mkv, etc.")
        print()
        print("💡 Tip: You can find test baby videos on YouTube or use your own!")
        print()
        return
    
    process_video(video_path)


if __name__ == "__main__":
    main()

