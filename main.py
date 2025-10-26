#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import find_peaks, butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class BreathingMonitorResearch:
    def __init__(self, window_size=150, block_size=50, hr_window_size=210, processing_size=(640, 480)):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            smooth_landmarks=True,
            enable_segmentation=True
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True
        )
        
        self.window_size = window_size
        self.hr_window_size = hr_window_size
        self.block_size = block_size
        self.processing_size = processing_size  # Fixed processing resolution (width, height)
        
        self.reset_data()
        self.setup_plot()
    
    def reset_data(self):
        """Reset only the data buffers and state, keeping MediaPipe models intact."""
        self.frames_processed = 0
        self.primary_person_size = 0
        
        self.points_initialized = False
        self.tracking_points = []
        self.current_person_bbox = None
        
        self.signal_history = {
            'chest': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)},
            'abdomen': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)},
            'nose': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)},
            'control': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)}
        }
        
        self.forehead_signal = {'B': deque(maxlen=self.hr_window_size), 'G': deque(maxlen=self.hr_window_size), 'R': deque(maxlen=self.hr_window_size)}
        self.left_cheek_signal = {'B': deque(maxlen=self.hr_window_size), 'G': deque(maxlen=self.hr_window_size), 'R': deque(maxlen=self.hr_window_size)}
        self.right_cheek_signal = {'B': deque(maxlen=self.hr_window_size), 'G': deque(maxlen=self.hr_window_size), 'R': deque(maxlen=self.hr_window_size)}
        self.motion_history = deque(maxlen=30)
        
        self.breathing_rate = 0
        self.heart_rate = 0
        self.confidence = 0
        self.hr_confidence = 0
        self.fps = 30
        
        self.landmark_quality = {}
        
        self.age_category = "adult"
        self.body_size_cm = 0
        self.detection_confidence = 0
        
        self.breathing_rate_history = deque(maxlen=self.window_size)
        self.heart_rate_history = deque(maxlen=self.window_size)
        self.time_history = deque(maxlen=self.window_size)
        self.start_time = time.time()
        
    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 10))
        
        self.ax1.set_title('Breathing Rate (Research Method)', color='cyan', fontsize=12, weight='bold')
        self.ax1.set_ylabel('Breaths/min', color='white')
        self.ax1.set_ylim(0, 80)
        self.ax1.axhspan(20, 60, alpha=0.2, color='green', label='Normal Range (20-60)')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'cyan', linewidth=2, label='Breathing Rate')
        self.ax1.legend(loc='upper right')
        
        self.ax2.set_title('Heart Rate (rPPG Method)', color='red', fontsize=12, weight='bold')
        self.ax2.set_ylabel('BPM', color='white')
        self.ax2.set_ylim(40, 200)
        self.ax2.axhspan(60, 100, alpha=0.2, color='green', label='Normal Range (60-100)')
        self.ax2.grid(True, alpha=0.3)
        self.line2, = self.ax2.plot([], [], 'red', linewidth=2, label='Heart Rate')
        self.ax2.legend(loc='upper right')
        
        self.ax3.set_title('Signal Quality', color='yellow', fontsize=10)
        self.ax3.set_ylabel('Amplitude', color='white')
        self.ax3.grid(True, alpha=0.3)
        self.line3_chest, = self.ax3.plot([], [], 'r', linewidth=2, label='Chest (Torso)')
        self.line3_abdomen, = self.ax3.plot([], [], 'g', linewidth=2, label='Abdomen')
        self.line3_nose, = self.ax3.plot([], [], 'cyan', linewidth=2, label='Nose')
        self.line3_control, = self.ax3.plot([], [], 'gray', linewidth=1, linestyle='--', label='Control')
        self.ax3.legend(loc='upper right', fontsize=8)
        
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
        
        self.canvas = FigureCanvasAgg(self.fig)
        
    def initialize_tracking_points(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        left_elbow = landmarks[13]
        right_elbow = landmarks[15]
        
        nose = landmarks[0]
        mouth_left = landmarks[9]
        mouth_right = landmarks[10]
        
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        
        chest_x = int((shoulder_mid_x * 0.5 + hip_mid_x * 0.5) * w)
        chest_y = int((shoulder_mid_y * 0.6 + hip_mid_y * 0.4) * h)
        
        chest_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
        
        abdomen_x = int((shoulder_mid_x * 0.3 + hip_mid_x * 0.7) * w)
        abdomen_y = int((shoulder_mid_y * 0.3 + hip_mid_y * 0.7) * h)
        
        abdomen_visibility = (left_hip.visibility + right_hip.visibility) / 2
        
        if nose.visibility > 0.5:
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            nose_visibility = nose.visibility
        else:
            nose_x = int((mouth_left.x + mouth_right.x) / 2 * w)
            nose_y = int((mouth_left.y + mouth_right.y) / 2 * h - 20)
            nose_visibility = (mouth_left.visibility + mouth_right.visibility) / 2
        
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * w
        
        control_offset = shoulder_width * 0.6
        control_x = int(shoulder_mid_x * w + control_offset)
        control_x = max(30, min(control_x, w - 30))
        control_y = int((shoulder_mid_y + hip_mid_y) / 2 * h)
        
        self.tracking_points = {
            'chest': (chest_x, chest_y),
            'abdomen': (abdomen_x, abdomen_y),
            'nose': (nose_x, nose_y),
            'control': (control_x, control_y)
        }
        
        self.landmark_quality = {
            'chest': chest_visibility,
            'abdomen': abdomen_visibility,
            'nose': nose_visibility,
            'control': 1.0
        }
        
        torso_height = abs((shoulder_mid_y - hip_mid_y) * h)
        shoulder_distance = abs((right_shoulder.x - left_shoulder.x) * w)
        
        self.adaptive_block_size = int(np.clip(min(torso_height, shoulder_distance) * 0.25, 30, 80))
        
        self.detect_age_category(landmarks, frame_shape)
        
        self.points_initialized = True
    
    def detect_age_category(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            if left_ankle.visibility > 0.5 or right_ankle.visibility > 0.5:
                ankle_y = min(left_ankle.y, right_ankle.y) if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5 else (left_ankle.y if left_ankle.visibility > 0.5 else right_ankle.y)
                total_height_px = abs((nose.y - ankle_y) * h)
            else:
                torso_height_px = abs((nose.y - (left_hip.y + right_hip.y) / 2) * h)
                total_height_px = torso_height_px * 2.5
            
            shoulder_width_px = abs((right_shoulder.x - left_shoulder.x) * w)
            
            torso_height_px = abs(((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2) * h)
            
            head_size_px = abs((nose.y - (left_shoulder.y + right_shoulder.y) / 2) * h)
            
            pixels_per_cm = w / 100
            estimated_height_cm = total_height_px / pixels_per_cm
            
            head_to_body_ratio = head_size_px / torso_height_px if torso_height_px > 0 else 0
            
            confidence_scores = {}
            
            infant_score = 0
            if estimated_height_cm < 90:
                infant_score += 3
            elif estimated_height_cm < 110:
                infant_score += 1
            
            if head_to_body_ratio > 0.4:
                infant_score += 2
            
            if torso_height_px < 150:
                infant_score += 2
            
            confidence_scores['infant'] = infant_score
            
            child_score = 0
            if 80 < estimated_height_cm < 150:
                child_score += 3
            
            if 0.25 < head_to_body_ratio < 0.45:
                child_score += 2
            
            if 120 < torso_height_px < 250:
                child_score += 2
            
            confidence_scores['child'] = child_score
            
            adult_score = 0
            if estimated_height_cm > 140:
                adult_score += 3
            
            if head_to_body_ratio < 0.35:
                adult_score += 2
            
            if torso_height_px > 200:
                adult_score += 2
            
            confidence_scores['adult'] = adult_score
            
            best_category = max(confidence_scores, key=confidence_scores.get)
            best_score = confidence_scores[best_category]
            
            if best_score > 3:
                if not hasattr(self, 'age_category') or self.age_category == "adult":
                    self.age_category = best_category
                else:
                    if not hasattr(self, 'category_votes'):
                        self.category_votes = {best_category: 1}
                    else:
                        self.category_votes[best_category] = self.category_votes.get(best_category, 0) + 1
                        if self.category_votes[best_category] > 10:
                            self.age_category = best_category
                            self.category_votes = {}
            
            self.body_size_cm = estimated_height_cm
            self.detection_confidence = best_score / 7.0
            
        except Exception as e:
            self.age_category = "adult"
            self.body_size_cm = 170
            self.detection_confidence = 0.5
    
    def get_normal_ranges(self):
        ranges = {
            'infant': {
                'hr_low': 100,
                'hr_high': 180,
                'hr_normal': (100, 160),
                'br_low': 25,
                'br_high': 70,
                'br_normal': (30, 60),
                'label': 'Infant'
            },
            'child': {
                'hr_low': 70,
                'hr_high': 140,
                'hr_normal': (70, 120),
                'br_low': 18,
                'br_high': 45,
                'br_normal': (20, 35),
                'label': 'Child'
            },
            'adult': {
                'hr_low': 50,
                'hr_high': 110,
                'hr_normal': (60, 100),
                'br_low': 10,
                'br_high': 25,
                'br_normal': (12, 20),
                'label': 'Adult'
            }
        }
        
        return ranges.get(self.age_category, ranges['adult'])
        
    def extract_region_signal(self, frame, center_point):
        x, y = center_point
        h, w = frame.shape[:2]
        
        if hasattr(self, 'adaptive_block_size'):
            half_block = self.adaptive_block_size // 2
        else:
            half_block = self.block_size // 2
        
        y_start = max(y - half_block, 0)
        y_end = min(y + half_block, h)
        x_start = max(x - half_block, 0)
        x_end = min(x + half_block, w)
        
        region = frame[y_start:y_end, x_start:x_end]
        
        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return None
        
        region_smoothed = cv2.GaussianBlur(region, (5, 5), 0)
        
        b_value = np.median(region_smoothed[:, :, 0])
        g_value = np.median(region_smoothed[:, :, 1])
        r_value = np.median(region_smoothed[:, :, 2])
        
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
        min_frames = int(self.window_size * 0.4)
        if len(self.signal_history['chest']['G']) < min_frames:
            return 0, 0
        
        all_breathing_rates = []
        all_confidences = []
        
        for region in ['chest', 'abdomen', 'nose']:
            region_signals = []
            for channel in ['B', 'G', 'R']:
                sig = np.array(list(self.signal_history[region][channel]))
                if len(sig) > min_frames:
                    try:
                        filtered = apply_bandpass_filter(sig, 0.1, 1.0, self.fps, order=3)
                        region_signals.append(filtered)
                    except:
                        detrended = sig - np.mean(sig)
                        region_signals.append(detrended)
            
            if len(region_signals) == 0:
                continue
            
            combined_signal = np.mean(region_signals, axis=0)
            
            min_distance = int(self.fps * 0.5)
            
            signal_std = np.std(combined_signal)
            signal_mean_abs = np.mean(np.abs(combined_signal))
            
            if signal_std < 1e-6:
                continue
            
            if self.age_category == 'infant':
                prominence = signal_std * 0.10
                min_height = signal_mean_abs * 0.1
            elif self.age_category == 'child':
                prominence = signal_std * 0.15
                min_height = signal_mean_abs * 0.15
            else:
                prominence = signal_std * 0.20
                min_height = signal_mean_abs * 0.2
            
            peaks, properties = find_peaks(
                combined_signal,
                distance=min_distance,
                prominence=prominence,
                height=min_height
            )
            
            if len(peaks) < 3:
                peaks_retry, _ = find_peaks(
                    combined_signal,
                    distance=int(min_distance * 0.8),
                    prominence=prominence * 0.7,
                    height=min_height * 0.5
                )
                if len(peaks_retry) > len(peaks):
                    peaks = peaks_retry
            
            confidence_multiplier = 1.0
            peak_quality = 1.0
            
            if len(peaks) >= 4:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                breathing_rate = max(8, min(breathing_rate, 80))
                confidence_multiplier = 1.2
                peak_quality = 1.0
                
            elif len(peaks) == 3:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                breathing_rate = max(8, min(breathing_rate, 80))
                confidence_multiplier = 1.0
                peak_quality = 0.9
                
            elif len(peaks) == 2:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                breathing_rate = max(8, min(breathing_rate, 80))
                confidence_multiplier = 0.6
                peak_quality = 0.7
                
            else:
                continue
            
            peak_regularity = 1.0
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks)
                peak_regularity = 1.0 - min(np.std(peak_intervals) / (np.mean(peak_intervals) + 1), 0.5)
            
            base_confidence = (signal_std * 1500 + len(peaks) * 15 + peak_regularity * 20)
            confidence = min(100, base_confidence * confidence_multiplier * peak_quality)
            confidence = max(0, min(confidence, 100))
            
            region_weight = self.landmark_quality.get(region, 0.5)
            weighted_confidence = confidence * region_weight
            
            all_breathing_rates.append(breathing_rate)
            all_confidences.append(weighted_confidence)
        
        if len(all_breathing_rates) == 0:
            return max(8, self.breathing_rate * 0.95), max(0, self.confidence * 0.8)
        
        total_confidence = sum(all_confidences)
        if total_confidence > 0:
            weighted_rate = sum(r * c for r, c in zip(all_breathing_rates, all_confidences)) / total_confidence
            avg_confidence = np.mean(all_confidences)
        else:
            weighted_rate = np.mean(all_breathing_rates)
            avg_confidence = 20
        
        if self.breathing_rate > 0:
            if avg_confidence > 70:
                weighted_rate = 0.2 * self.breathing_rate + 0.8 * weighted_rate
            elif avg_confidence > 40:
                weighted_rate = 0.4 * self.breathing_rate + 0.6 * weighted_rate
            else:
                weighted_rate = 0.7 * self.breathing_rate + 0.3 * weighted_rate
        
        return weighted_rate, avg_confidence
    
    def estimate_heart_rate(self):
        min_frames = int(self.window_size * 0.5)
        if len(self.signal_history['nose']['G']) < min_frames:
            return 0, 0
        
        green_signal = np.array(list(self.signal_history['nose']['G']))
        
        if len(green_signal) < min_frames:
            return 0, 0
        
        try:
            filtered_signal = apply_bandpass_filter(green_signal, 0.7, 4.0, self.fps, order=5)
        except:
            filtered_signal = green_signal - np.mean(green_signal)
        
        min_distance = int(self.fps * 0.3)
        
        signal_std = np.std(filtered_signal)
        if signal_std < 1e-6:
            return 0, 0
        
        prominence = signal_std * 0.3
        
        peaks, properties = find_peaks(
            filtered_signal,
            distance=min_distance,
            prominence=prominence
        )
        
        if len(peaks) < 3:
            return max(40, self.heart_rate * 0.9), max(0, self.hr_confidence * 0.7)
        
        time_duration = len(filtered_signal) / self.fps
        heart_rate = (len(peaks) / time_duration) * 60
        
        heart_rate = max(40, min(heart_rate, 200))
        
        peak_regularity = 1.0 - (np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-6)) if len(peaks) > 2 else 0
        peak_regularity = max(0, min(peak_regularity, 1))
        
        confidence = min(100, (signal_std * 3000 + len(peaks) * 5 + peak_regularity * 30))
        confidence = max(0, min(confidence, 100))
        
        if self.heart_rate > 0:
            heart_rate = 0.4 * self.heart_rate + 0.6 * heart_rate
        
        return heart_rate, confidence
    
    def update_plot(self):
        if len(self.time_history) > 0:
            times = list(self.time_history)
            
            if len(self.breathing_rate_history) > 0:
                rates = list(self.breathing_rate_history)
                self.line1.set_data(times, rates)
                self.ax1.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            if len(self.heart_rate_history) > 0:
                hr_rates = list(self.heart_rate_history)
                self.line2.set_data(times, hr_rates)
                self.ax2.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            if len(self.signal_history['chest']['G']) > 10:
                chest_sig = np.array(list(self.signal_history['chest']['G']))
                abdomen_sig = np.array(list(self.signal_history['abdomen']['G']))
                nose_sig = np.array(list(self.signal_history['nose']['G']))
                control_sig = np.array(list(self.signal_history['control']['G']))
                
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
                
                max_vals = []
                if len(chest_norm) > 0:
                    max_vals.append(np.max(np.abs(chest_norm)))
                if len(abdomen_norm) > 0:
                    max_vals.append(np.max(np.abs(abdomen_norm)))
                if len(nose_norm) > 0:
                    max_vals.append(np.max(np.abs(nose_norm)))
                max_val = max(max_vals) if max_vals else 1
                max_val = max(max_val, 1)
                self.ax3.set_ylim(-max_val * 1.5, max_val * 1.5)
            
            breathing_conf_history = [self.confidence] * len(times)
            heart_conf_history = [self.hr_confidence] * len(times)
            self.line4_breathing.set_data(times, breathing_conf_history)
            self.line4_heart.set_data(times, heart_conf_history)
            self.ax4.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
        
        self.canvas.draw()
        
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        plot_image = buf.reshape(self.canvas.get_width_height()[::-1] + (4,))
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
        
        return plot_image
    
    def process_frame(self, frame):
        # Store original frame dimensions for drawing and landmark scaling
        h, w = frame.shape[:2]
        original_h, original_w = h, w
        
        # Normalize frame to FIXED processing resolution (width, height)
        # This ensures MediaPipe always sees the exact same dimensions regardless of source
        processing_frame = cv2.resize(frame, self.processing_size)
        
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        current_time = time.time() - self.start_time
        
        self.frames_processed += 1
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            xs = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
            ys = [lm.y * h for lm in landmarks if lm.visibility > 0.5]
            
            if xs and ys:
                person_width = max(xs) - min(xs)
                person_height = max(ys) - min(ys)
                person_size = person_width * person_height
                
                if person_size > self.primary_person_size * 0.8:
                    self.primary_person_size = max(person_size, self.primary_person_size)
                elif person_size < self.primary_person_size * 0.5:
                    return frame
            
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2)
            )
            
            self.initialize_tracking_points(landmarks, frame.shape)
            
            for region_name, point in self.tracking_points.items():
                signals = self.extract_region_signal(frame, point)
                if signals:
                    for channel in ['B', 'G', 'R']:
                        self.signal_history[region_name][channel].append(signals[channel])
                
                if region_name == 'chest':
                    color = (0, 0, 255)
                elif region_name == 'abdomen':
                    color = (0, 255, 0)
                elif region_name == 'nose':
                    color = (255, 255, 0)
                else:
                    color = (128, 128, 128)
                
                if hasattr(self, 'adaptive_block_size'):
                    block_radius = self.adaptive_block_size // 2
                else:
                    block_radius = self.block_size // 2
                
                cv2.rectangle(frame, 
                             (point[0] - block_radius, point[1] - block_radius),
                             (point[0] + block_radius, point[1] + block_radius),
                             color, 2)
                
                cv2.circle(frame, point, 5, color, -1)
                cv2.circle(frame, point, 8, (255, 255, 255), 2)
                
                label = region_name.upper()
                if region_name in self.landmark_quality:
                    quality = self.landmark_quality[region_name]
                    label += f" ({quality:.0%})"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_pos = (point[0] + 15, point[1] + 5)
                cv2.rectangle(frame,
                             (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                             (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                             (0, 0, 0), -1)
                cv2.putText(frame, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Always estimate rates (no stabilization delay)
            self.breathing_rate, self.confidence = self.estimate_breathing_rate()
            self.heart_rate, self.hr_confidence = self.estimate_heart_rate()
            
        self.breathing_rate_history.append(self.breathing_rate)
        self.heart_rate_history.append(self.heart_rate)
        self.time_history.append(current_time)
        
        ranges = self.get_normal_ranges()
        
        # Determine breathing rate status
        status_text = "NORMAL"
        text_color = (0, 255, 0)
        
        if self.breathing_rate < ranges['br_low'] and self.breathing_rate > 0:
            status_text = "LOW"
            text_color = (0, 0, 255)
        elif self.breathing_rate > ranges['br_high']:
            status_text = "HIGH"
            text_color = (0, 0, 255)
        
        rate_text = f"Breathing: {self.breathing_rate:.1f} BPM -- {status_text}"
        normal_range = f"(Normal: {ranges['br_normal'][0]}-{ranges['br_normal'][1]})"
        cv2.putText(frame, rate_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(frame, normal_range, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Determine heart rate status
        hr_status = "NORMAL"
        hr_color = (0, 255, 0)
        
        if self.heart_rate < ranges['hr_low'] and self.heart_rate > 0:
            hr_status = "LOW"
            hr_color = (0, 0, 255)
        elif self.heart_rate > ranges['hr_high']:
            hr_status = "HIGH"
            hr_color = (0, 0, 255)
        
        hr_text = f"Heart Rate: {self.heart_rate:.0f} BPM -- {hr_status}"
        hr_normal_range = f"(Normal: {ranges['hr_normal'][0]}-{ranges['hr_normal'][1]})"
        cv2.putText(frame, hr_text, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
        cv2.putText(frame, hr_normal_range, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        conf_text = f"BR Conf: {self.confidence:.0f}%  |  HR Conf: {self.hr_confidence:.0f}%"
        cv2.putText(frame, conf_text, (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Research: Breathing + rPPG Heart Rate | Auto Age Detection", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame

def resize_with_aspect_ratio(image, target_width, target_height, bg_color=(0, 0, 0)):
    """Resize image to fit within target dimensions while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image with aspect ratio preserved
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas with target dimensions
    canvas = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    
    # Calculate centering position
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def main():
    monitor = BreathingMonitorResearch()
    
    use_webcam = True
    video_path = '/Users/aidenm/Testch/test_videos/002.mp4'
    
    def get_capture(use_webcam):
        if use_webcam:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(video_path)
        return cap
    
    cap = get_capture(use_webcam)
    if not cap.isOpened():
        return
    
    cv2.namedWindow('Vital Signs Monitor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vital Signs Monitor', 1920, 1080)
    
    paused = False
    combined = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not use_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            if use_webcam:
                frame = cv2.flip(frame, 1)
            
            processed_frame = monitor.process_frame(frame)
            plot_image = monitor.update_plot()
            
            # Resize with aspect ratio preserved and centered
            display_frame = resize_with_aspect_ratio(processed_frame, 960, 1080)
            display_plot = resize_with_aspect_ratio(plot_image, 960, 1080)
            combined = cv2.hconcat([display_frame, display_plot])
            
            source_text = "SOURCE: WEBCAM" if use_webcam else "SOURCE: VIDEO FILE"
            source_color = (0, 255, 255) if use_webcam else (255, 0, 255)
            cv2.putText(combined, source_text, (20, combined.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, source_color, 3)
            
            controls_text = "Controls: [T] Toggle Source | [SPACE] Pause | [S] Screenshot | [Q] Quit"
            cv2.putText(combined, controls_text, (20, combined.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Always update the display (even when paused)
        if combined is not None:
            # Add PAUSED indicator when paused
            display_combined = combined.copy()
            if paused:
                cv2.putText(display_combined, "PAUSED", (display_combined.shape[1]//2 - 100, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            cv2.imshow('Vital Signs Monitor', display_combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_webcam = not use_webcam
            cap.release()
            cap = get_capture(use_webcam)
            monitor.reset_data()  # Reset data buffers, models stay initialized
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            if combined is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'screenshot_{timestamp}.png', combined)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
