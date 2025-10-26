#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time
import gradio as gr
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
        self.last_valid_landmarks = None
        self.frames_since_detection = 0
        self.recent_frames = deque(maxlen=150)
        
        self.signal_history = {
            'chest': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)},
            'abdomen': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)},
            'nose': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)},
            'control': {'B': deque(maxlen=self.window_size), 'G': deque(maxlen=self.window_size), 'R': deque(maxlen=self.window_size)}
        }
        
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
        
        return {'B': b_value, 'G': g_value, 'R': r_value}
    
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
            
            peaks, properties = find_peaks(combined_signal, distance=min_distance, prominence=prominence, height=min_height)
            
            if len(peaks) < 3:
                peaks_retry, _ = find_peaks(combined_signal, distance=int(min_distance * 0.8), prominence=prominence * 0.7, height=min_height * 0.5)
                if len(peaks_retry) > len(peaks):
                    peaks = peaks_retry
            
            if len(peaks) >= 4:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                breathing_rate = max(8, min(breathing_rate, 80))
                confidence = 80
            elif len(peaks) == 3:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                breathing_rate = max(8, min(breathing_rate, 80))
                confidence = 60
            elif len(peaks) == 2:
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                breathing_rate = max(8, min(breathing_rate, 80))
                confidence = 40
            else:
                continue
            
            all_breathing_rates.append(breathing_rate)
            all_confidences.append(confidence)
        
        if len(all_breathing_rates) == 0:
            return max(8, self.breathing_rate * 0.95), max(0, self.confidence * 0.8)
        
        weighted_rate = np.mean(all_breathing_rates)
        avg_confidence = np.mean(all_confidences)
        
        if self.breathing_rate > 0:
            weighted_rate = 0.3 * self.breathing_rate + 0.7 * weighted_rate
        
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
        peaks, properties = find_peaks(filtered_signal, distance=min_distance, prominence=prominence)
        
        if len(peaks) < 3:
            return max(40, self.heart_rate * 0.9), max(0, self.hr_confidence * 0.7)
        
        time_duration = len(filtered_signal) / self.fps
        heart_rate = (len(peaks) / time_duration) * 60
        heart_rate = max(40, min(heart_rate, 200))
        confidence = 70
        
        if self.heart_rate > 0:
            heart_rate = 0.4 * self.heart_rate + 0.6 * heart_rate
        
        return heart_rate, confidence
    
    def create_plotly_charts(self):
        if len(self.time_history) == 0:
            return None
        
        times = list(self.time_history)
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Breathing Rate', 'Heart Rate', 'Signal Quality', 'Confidence'),
            vertical_spacing=0.08,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        if len(self.breathing_rate_history) > 0:
            rates = list(self.breathing_rate_history)
            fig.add_trace(go.Scatter(x=times, y=rates, name='Breathing', line=dict(color='cyan', width=2)), row=1, col=1)
            fig.add_hrect(y0=12, y1=20, fillcolor="green", opacity=0.2, line_width=0, row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="BPM", range=[0, 80], row=1, col=1)
        
        if len(self.heart_rate_history) > 0:
            hr_rates = list(self.heart_rate_history)
            fig.add_trace(go.Scatter(x=times, y=hr_rates, name='Heart', line=dict(color='red', width=2)), row=2, col=1)
            fig.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.2, line_width=0, row=2, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="BPM", range=[40, 200], row=2, col=1)
        
        if len(self.signal_history['chest']['G']) > 10:
            signals = {
                'chest': np.array(list(self.signal_history['chest']['G'])),
                'abdomen': np.array(list(self.signal_history['abdomen']['G'])),
                'nose': np.array(list(self.signal_history['nose']['G'])),
                'control': np.array(list(self.signal_history['control']['G']))
            }
            
            normalized = {}
            for name, sig in signals.items():
                if len(sig) > 0 and np.std(sig) > 1e-6:
                    normalized[name] = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
                else:
                    normalized[name] = sig
            
            signal_len = len(signals['chest'])
            plot_times = times[-signal_len:] if len(times) >= signal_len else times
            
            colors = {'chest': 'red', 'abdomen': 'green', 'nose': 'cyan', 'control': 'gray'}
            for name, sig in normalized.items():
                if len(sig) > 0:
                    fig.add_trace(go.Scatter(x=plot_times, y=sig, name=name.capitalize(),
                                           line=dict(color=colors[name], width=2)), row=3, col=1)
            
            if len(plot_times) > 0:
                y_ranges = [np.max(np.abs(sig)) for sig in normalized.values() if len(sig) > 0]
                y_max = max(y_ranges) if y_ranges else 5
                fig.update_yaxes(range=[-y_max * 1.1, y_max * 1.1], row=3, col=1)
            
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        
        breathing_conf = [self.confidence] * len(times) if self.confidence else []
        heart_conf = [self.hr_confidence] * len(times) if self.hr_confidence else []
        
        if len(breathing_conf) > 0:
            fig.add_trace(go.Scatter(x=times, y=breathing_conf, name='BR Confidence', line=dict(color='cyan')), row=4, col=1)
        if len(heart_conf) > 0:
            fig.add_trace(go.Scatter(x=times, y=heart_conf, name='HR Confidence', line=dict(color='red')), row=4, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)
        fig.update_yaxes(title_text="%", range=[0, 100], row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True, hovermode='x unified', template='plotly_dark')
        return fig
    
    def process_frame(self, frame):
        self.recent_frames.append(frame.copy())
        h, w = frame.shape[:2]
        processing_frame = cv2.resize(frame, self.processing_size)
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        current_time = time.time() - self.start_time
        self.frames_processed += 1
        
        if pose_results.pose_landmarks:
            self.frames_since_detection = 0
            self.last_valid_landmarks = pose_results.pose_landmarks
            landmarks = pose_results.pose_landmarks.landmark
            processing_w, processing_h = self.processing_size
            
            self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
            self.initialize_tracking_points(landmarks, processing_frame.shape)
            
            if self.tracking_points:
                scale_x = w / self.processing_size[0]
                scale_y = h / self.processing_size[1]
                
                for region in self.tracking_points:
                    x, y = self.tracking_points[region]
                    self.tracking_points[region] = (int(x * scale_x), int(y * scale_y))
                
                if hasattr(self, 'adaptive_block_size'):
                    avg_scale = (scale_x + scale_y) / 2
                    self.adaptive_block_size = int(self.adaptive_block_size * avg_scale)
        
        elif self.last_valid_landmarks and self.frames_since_detection < 30:
            self.frames_since_detection += 1
            landmarks = self.last_valid_landmarks.landmark
            if not self.points_initialized:
                self.initialize_tracking_points(landmarks, processing_frame.shape)
        
        if self.points_initialized and self.tracking_points:
            for region_name, point in self.tracking_points.items():
                signals = self.extract_region_signal(frame, point)
                if signals:
                    for channel in ['B', 'G', 'R']:
                        self.signal_history[region_name][channel].append(signals[channel])
                
                color = {'chest': (0, 0, 255), 'abdomen': (0, 255, 0), 'nose': (255, 255, 0)}.get(region_name, (128, 128, 128))
                block_radius = self.adaptive_block_size // 2 if hasattr(self, 'adaptive_block_size') else self.block_size // 2
                
                cv2.rectangle(frame, (point[0] - block_radius, point[1] - block_radius),
                             (point[0] + block_radius, point[1] + block_radius), color, 2)
                cv2.circle(frame, point, 5, color, -1)
                cv2.circle(frame, point, 8, (255, 255, 255), 2)
        
        self.breathing_rate, self.confidence = self.estimate_breathing_rate()
        self.heart_rate, self.hr_confidence = self.estimate_heart_rate()
        
        self.breathing_rate_history.append(self.breathing_rate)
        self.heart_rate_history.append(self.heart_rate)
        self.time_history.append(current_time)
        
        if len(self.time_history) > 0:
            cutoff_time = current_time - 10.0
            br_values = [br for t, br in zip(self.time_history, self.breathing_rate_history) if t >= cutoff_time and br > 0]
            self.avg_breathing_rate = sum(br_values) / len(br_values) if br_values else self.breathing_rate
            
            hr_values = [hr for t, hr in zip(self.time_history, self.heart_rate_history) if t >= cutoff_time and hr > 0]
            self.avg_heart_rate = sum(hr_values) / len(hr_values) if hr_values else self.heart_rate
        
        return frame

monitor = BreathingMonitorResearch()

def process_frame(frame):
    if frame is None:
        return None, "", None
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_frame = monitor.process_frame(frame)
    result_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    metrics = f"""Breathing Rate: {monitor.breathing_rate:.1f} BPM (Avg: {monitor.avg_breathing_rate:.1f})
Heart Rate: {monitor.heart_rate:.1f} BPM (Avg: {monitor.avg_heart_rate:.1f})
Confidence: BR={monitor.confidence:.1f}%, HR={monitor.hr_confidence:.1f}%"""
    
    fig = monitor.create_plotly_charts()
    
    return result_frame, metrics, fig


def process_video(video):
    if video is None:
        return None, "No video provided"
    
    vid = cv2.VideoCapture(video)
    if not vid.isOpened():
        return None, "Failed to open video"
    
    output_frames = []
    frame_count = 0
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        processed_frame = monitor.process_frame(frame)
        output_frames.append(processed_frame)
        
        frame_count += 1
        if frame_count > 500:
            break
    
    vid.release()
    
    if len(output_frames) == 0:
        return None, "No frames processed"
    
    height, width = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = 'output_processed.mp4'
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
    
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    metrics = f"""Breathing Rate: {monitor.breathing_rate:.1f} BPM (Avg: {monitor.avg_breathing_rate:.1f})
Heart Rate: {monitor.heart_rate:.1f} BPM (Avg: {monitor.avg_heart_rate:.1f})
Confidence: BR={monitor.confidence:.1f}%, HR={monitor.hr_confidence:.1f}%"""
    
    fig = monitor.create_plotly_charts()
    
    return out_path, metrics, fig

def reset_monitor():
    monitor.reset_data()
    return "✅ Monitor Reset"

def stream_webcam():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield None, "Failed to open webcam", None
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = monitor.process_frame(frame)
            result_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            metrics = f"""Breathing Rate: {monitor.breathing_rate:.1f} BPM (Avg: {monitor.avg_breathing_rate:.1f})
Heart Rate: {monitor.heart_rate:.1f} BPM (Avg: {monitor.avg_heart_rate:.1f})
Confidence: BR={monitor.confidence:.1f}%, HR={monitor.hr_confidence:.1f}%"""
            
            fig = monitor.create_plotly_charts()
            
            yield result_frame, metrics, fig
            time.sleep(0.033)
    finally:
        cap.release()

with gr.Blocks(title="Breathing & Heart Rate Monitor") as demo:
    gr.Markdown("# 💓 Breathing & Heart Rate Monitor (rPPG)")
    
    with gr.Tabs():
        with gr.TabItem("📹 Real-time Webcam"):
            with gr.Row():
                with gr.Column():
                    btn_start = gr.Button("Start Webcam", variant="primary")
                    btn_reset = gr.Button("Reset Monitor", variant="secondary")
                
                with gr.Column():
                    webcam_output = gr.Image(label="Processed Feed")
                    metrics_output = gr.Textbox(label="Vital Signs", lines=4)
                    chart_output = gr.Plot()
            
            btn_start.click(fn=stream_webcam, outputs=[webcam_output, metrics_output, chart_output])
            btn_reset.click(fn=reset_monitor, outputs=[])
        
        with gr.TabItem("🎥 Video Upload"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    process_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column():
                    video_result = gr.Video(label="Processed Video")
                    video_metrics = gr.Textbox(label="Results", lines=5)
                    video_chart = gr.Plot()
            
            process_btn.click(fn=process_video, inputs=video_input, 
                            outputs=[video_result, video_metrics, video_chart])

demo.launch()

