#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            smooth_landmarks=True,
            enable_segmentation=True
        )
        self.window_size = window_size
        self.hr_window_size = hr_window_size
        self.block_size = block_size
        self.processing_size = processing_size
        self.reset_data()
    
    def reset_data(self):
        self.frames_processed = 0
        self.primary_person_size = 0
        self.points_initialized = False
        self.tracking_points = []
        self.current_person_bbox = None
        self.last_valid_landmarks = None
        self.frames_since_detection = 0
        self.recent_frames = deque(maxlen=150)
        
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
        self.avg_breathing_rate = 0
        self.avg_heart_rate = 0
        self.confidence = 0
        self.hr_confidence = 0
        self.fps = 30
        self.landmark_quality = {}
        
        if hasattr(self, 'adaptive_block_size'):
            delattr(self, 'adaptive_block_size')
        
        self.breathing_rate_history = deque(maxlen=self.window_size)
        self.heart_rate_history = deque(maxlen=self.window_size)
        self.time_history = deque(maxlen=self.window_size)
        self.start_time = time.time()
        
    def initialize_tracking_points(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
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
        
        torso_height = abs((shoulder_mid_y - hip_mid_y) * h)
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
        
        shoulder_distance = abs((right_shoulder.x - left_shoulder.x) * w)
        self.adaptive_block_size = int(np.clip(min(torso_height, shoulder_distance) * 0.25, 30, 80))
        
        self.points_initialized = True
        
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
        
        signal_quality = (np.std(region[:, :, 0]) + np.std(region[:, :, 1]) + np.std(region[:, :, 2])) / 3
        
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
    
    def create_plotly_charts(self):
        if len(self.time_history) == 0:
            return None
        
        times = list(self.time_history)
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Breathing Rate', 'Heart Rate (rPPG Method)', 'Signal Quality', 'Confidence Scores'),
            vertical_spacing=0.08,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        if len(self.breathing_rate_history) > 0:
            rates = list(self.breathing_rate_history)
            fig.add_trace(
                go.Scatter(x=times, y=rates, name='Breathing Rate', line=dict(color='cyan', width=2)),
                row=1, col=1
            )
            
            fig.add_hrect(y0=12, y1=20, fillcolor="green", opacity=0.2, line_width=0, row=1, col=1)
            
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
            fig.update_yaxes(title_text="Breaths/min", range=[0, 80], row=1, col=1)
        
        if len(self.heart_rate_history) > 0:
            hr_rates = list(self.heart_rate_history)
            fig.add_trace(
                go.Scatter(x=times, y=hr_rates, name='Heart Rate', line=dict(color='red', width=2)),
                row=2, col=1
            )
            
            fig.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.2, line_width=0, row=2, col=1)
            
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
            fig.update_yaxes(title_text="BPM", range=[40, 200], row=2, col=1)
        
        if len(self.signal_history['chest']['G']) > 10:
            chest_sig = np.array(list(self.signal_history['chest']['G']))
            abdomen_sig = np.array(list(self.signal_history['abdomen']['G']))
            nose_sig = np.array(list(self.signal_history['nose']['G']))
            control_sig = np.array(list(self.signal_history['control']['G']))
            
            signals = {
                'chest': chest_sig,
                'abdomen': abdomen_sig,
                'nose': nose_sig,
                'control': control_sig
            }
            
            normalized_signals = {}
            for name, sig in signals.items():
                if len(sig) > 0 and np.std(sig) > 1e-6:
                    normalized_signals[name] = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
                else:
                    normalized_signals[name] = sig
            
            signal_len = len(chest_sig)
            if len(times) >= signal_len:
                plot_times = times[-signal_len:]
            else:
                plot_times = times
            
            colors = {
                'chest': 'red',
                'abdomen': 'green',
                'nose': 'cyan',
                'control': 'gray'
            }
            
            for name, sig in normalized_signals.items():
                if len(sig) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_times, y=sig, name=name.capitalize(), 
                            line=dict(color=colors[name], width=2, dash='dash' if name == 'control' else 'solid')
                        ),
                        row=3, col=1
                    )
            
            if len(plot_times) > 0:
                y_min = -5
                y_max = 5
                for sig in normalized_signals.values():
                    if len(sig) > 0:
                        y_min = min(y_min, np.min(sig))
                        y_max = max(y_max, np.max(sig))
                fig.update_yaxes(range=[y_min * 1.1, y_max * 1.1], row=3, col=1)
            
            fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
            fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        
        breathing_conf_history = [self.confidence] * len(times) if self.confidence else []
        heart_conf_history = [self.hr_confidence] * len(times) if self.hr_confidence else []
        
        if len(breathing_conf_history) > 0:
            fig.add_trace(
                go.Scatter(x=times, y=breathing_conf_history, name='Breathing Confidence', line=dict(color='cyan', width=2)),
                row=4, col=1
            )
        if len(heart_conf_history) > 0:
            fig.add_trace(
                go.Scatter(x=times, y=heart_conf_history, name='Heart Rate Confidence', line=dict(color='red', width=2)),
                row=4, col=1
            )
        
        fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)
        fig.update_yaxes(title_text="Confidence %", range=[0, 100], row=4, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def process_frame(self, frame):
        self.recent_frames.append(frame.copy())
        h, w = frame.shape[:2]
        
        processing_frame = cv2.resize(frame, self.processing_size)
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        current_time = time.time() - self.start_time
        self.frames_processed += 1
        
        detection_successful = False
        
        if pose_results.pose_landmarks:
            detection_successful = True
            self.frames_since_detection = 0
            self.last_valid_landmarks = pose_results.pose_landmarks
            
            landmarks = pose_results.pose_landmarks.landmark
            processing_w, processing_h = self.processing_size
            xs = [lm.x * processing_w for lm in landmarks if lm.visibility > 0.5]
            ys = [lm.y * processing_h for lm in landmarks if lm.visibility > 0.5]
            
            if xs and ys:
                person_width = max(xs) - min(xs)
                person_height = max(ys) - min(ys)
                person_size = person_width * person_height
                
                if person_size > self.primary_person_size * 0.7:
                    self.primary_person_size = max(person_size, self.primary_person_size)
                elif person_size < self.primary_person_size * 0.3:
                    return frame
            
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2)
            )
            
            self.initialize_tracking_points(landmarks, processing_frame.shape)
            
            if self.tracking_points:
                scale_x = w / self.processing_size[0]
                scale_y = h / self.processing_size[1]
                
                for region in self.tracking_points:
                    x, y = self.tracking_points[region]
                    self.tracking_points[region] = (
                        int(x * scale_x),
                        int(y * scale_y)
                    )
                
                if hasattr(self, 'adaptive_block_size'):
                    avg_scale = (scale_x + scale_y) / 2
                    self.adaptive_block_size = int(self.adaptive_block_size * avg_scale)
        
        elif self.last_valid_landmarks and self.frames_since_detection < 30:
            detection_successful = True
            self.frames_since_detection += 1
            
            self.mp_drawing.draw_landmarks(
                frame,
                self.last_valid_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(128, 128, 128), thickness=1)
            )
            
            landmarks = self.last_valid_landmarks.landmark
            if not self.points_initialized or not self.tracking_points:
                self.initialize_tracking_points(landmarks, processing_frame.shape)
                
                if self.tracking_points:
                    scale_x = w / self.processing_size[0]
                    scale_y = h / self.processing_size[1]
                    
                    for region in self.tracking_points:
                        x, y = self.tracking_points[region]
                        self.tracking_points[region] = (
                            int(x * scale_x),
                            int(y * scale_y)
                        )
                    
                    if hasattr(self, 'adaptive_block_size'):
                        avg_scale = (scale_x + scale_y) / 2
                        self.adaptive_block_size = int(self.adaptive_block_size * avg_scale)
        else:
            if self.last_valid_landmarks is not None:
                self.last_valid_landmarks = None
                self.points_initialized = False
                self.tracking_points = []
        
        if self.points_initialized and self.tracking_points:
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
        
        self.breathing_rate, self.confidence = self.estimate_breathing_rate()
        self.heart_rate, self.hr_confidence = self.estimate_heart_rate()
        
        self.breathing_rate_history.append(self.breathing_rate)
        self.heart_rate_history.append(self.heart_rate)
        self.time_history.append(current_time)
        
        if len(self.time_history) > 0:
            cutoff_time = current_time - 10.0
            br_values = [br for t, br in zip(self.time_history, self.breathing_rate_history) 
                        if t >= cutoff_time and br > 0]
            self.avg_breathing_rate = sum(br_values) / len(br_values) if br_values else self.breathing_rate
            
            hr_values = [hr for t, hr in zip(self.time_history, self.heart_rate_history) 
                        if t >= cutoff_time and hr > 0]
            self.avg_heart_rate = sum(hr_values) / len(hr_values) if hr_values else self.heart_rate
        
        return frame

st.set_page_config(
    page_title="Breathing & Heart Rate Monitor",
    page_icon="💓",
    layout="wide"
)

if 'monitor' not in st.session_state:
    try:
        st.session_state.monitor = BreathingMonitorResearch()
        st.session_state.processing = False
    except Exception as e:
        st.error(f"Error initializing monitor: {e}")
        st.stop()

st.title("💓 Breathing & Heart Rate Monitor (rPPG)")
st.sidebar.header("Settings")

use_webcam = st.sidebar.checkbox("Use Webcam", value=False)
video_file = st.sidebar.file_uploader("Or Upload Video File", type=['mp4', 'avi', 'mov'])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Video Feed")
    video_placeholder = st.empty()
    video_info_placeholder = st.empty()

with col2:
    st.subheader("Vital Signs")
    metrics_placeholder = st.empty()
    charts_placeholder = st.empty()

if 'initialized_display' not in st.session_state:
    st.session_state.initialized_display = True

if st.sidebar.button("Start Monitoring", disabled=st.session_state.processing):
    if not use_webcam and video_file is None:
        st.sidebar.error("Please select webcam or upload a video")
    else:
        st.session_state.processing = True
        st.session_state.monitor.reset_data()
        st.rerun()

if st.sidebar.button("Stop Monitoring"):
    st.session_state.processing = False
    st.rerun()

refresh_controls = st.sidebar.container()
auto_video = refresh_controls.checkbox("Auto-refresh video", value=False, key="auto_video")
auto_metrics = refresh_controls.checkbox("Auto-refresh metrics", value=True, key="auto_metrics")
auto_charts = refresh_controls.checkbox("Auto-refresh charts", value=True, key="auto_charts")

refresh_controls.write("---")
if refresh_controls.button("Update Video", disabled=not st.session_state.processing, key="update_video"):
    st.session_state.update_video = True
    st.rerun()
if refresh_controls.button("Update Metrics", disabled=not st.session_state.processing, key="update_metrics"):
    st.session_state.update_metrics = True
    st.rerun()
if refresh_controls.button("Update Charts", disabled=not st.session_state.processing, key="update_charts"):
    st.session_state.update_charts = True
    st.rerun()

with metrics_placeholder.container():
    if len(st.session_state.monitor.time_history) > 0:
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("Breathing Rate", f"{st.session_state.monitor.breathing_rate:.1f} BPM",
                     f"Avg: {st.session_state.monitor.avg_breathing_rate:.1f}")
            st.metric("Confidence", f"{st.session_state.monitor.confidence:.1f}%")
        
        with metric_col2:
            st.metric("Heart Rate", f"{st.session_state.monitor.heart_rate:.1f} BPM",
                     f"Avg: {st.session_state.monitor.avg_heart_rate:.1f}")
            st.metric("HR Confidence", f"{st.session_state.monitor.hr_confidence:.1f}%")
    else:
        st.info("📹 Start monitoring to see vital signs")

with charts_placeholder.container():
    fig = st.session_state.monitor.create_plotly_charts()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

if st.session_state.processing:
    if use_webcam:
        if 'cap' not in st.session_state or st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
        
        cap = st.session_state.cap
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed_frame = st.session_state.monitor.process_frame(frame)
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                video_info_placeholder.write("🔴 Live Webcam Feed")
        
    elif video_file is not None:
        if 'video_path' not in st.session_state:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.getvalue())
                st.session_state.video_path = tmp_file.name
            
            if 'cap' not in st.session_state:
                st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)
        
        cap = st.session_state.cap
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                processed_frame = st.session_state.monitor.process_frame(frame)
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                video_info_placeholder.write(f"📹 Video Frame: {st.session_state.monitor.frames_processed}")
            else:
                st.session_state.cap.release()
                st.session_state.processing = False
                video_info_placeholder.write("✅ Video Complete")
else:
    video_info_placeholder.empty()

if st.session_state.processing:
    if auto_video:
        if 'last_video_time' not in st.session_state:
            st.session_state.last_video_time = time.time()
        if time.time() - st.session_state.last_video_time > 0.3:
            st.session_state.last_video_time = time.time()
            time.sleep(0.1)
            st.rerun()
    
    if auto_metrics or auto_charts:
        if 'last_metrics_time' not in st.session_state:
            st.session_state.last_metrics_time = time.time()
        if time.time() - st.session_state.last_metrics_time > 1.0:
            st.session_state.last_metrics_time = time.time()
            time.sleep(0.1)
            st.rerun()

if not st.session_state.processing:
    if 'cap' in st.session_state:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        st.session_state.cap = None
    if 'video_path' in st.session_state:
        import os
        if os.path.exists(st.session_state.video_path):
            os.unlink(st.session_state.video_path)
        del st.session_state.video_path
    if 'last_video_time' in st.session_state:
        del st.session_state.last_video_time
    if 'last_metrics_time' in st.session_state:
        del st.session_state.last_metrics_time
