"""
Infant Video Breathing Rate Analyzer
Analyzes pre-recorded infant videos to measure breathing rate
Uses research-validated multi-point BGR signal processing
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import sys
import os
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass Butterworth filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to signal"""
    if len(data) < order * 3:  # Need sufficient data points
        return data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class InfantVideoAnalyzer:
    def __init__(self, video_path):
        """Initialize analyzer with video file"""
        self.video_path = video_path
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Better accuracy for analysis
            min_detection_confidence=0.5
        )
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Video Information:")
        print(f"{'='*60}")
        print(f"File: {os.path.basename(video_path)}")
        print(f"Duration: {self.duration:.1f} seconds")
        print(f"FPS: {self.fps:.1f}")
        print(f"Total Frames: {self.total_frames}")
        print(f"{'='*60}\n")
        
        # Signal storage
        self.signal_history = {
            'chest': {'B': [], 'G': [], 'R': []},
            'abdomen': {'B': [], 'G': [], 'R': []},
            'nose': {'B': [], 'G': [], 'R': []}
        }
        
        self.breathing_rates = []
        self.timestamps = []
        self.tracking_points = None
        
    def initialize_tracking_points(self, landmarks, frame_shape):
        """Initialize tracking points based on detected pose"""
        h, w = frame_shape[:2]
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        nose = landmarks[0]
        
        # Calculate tracking points
        chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        chest_y = int((left_shoulder.y + right_shoulder.y) / 2 * h + 40)
        
        abdomen_x = int((left_hip.x + right_hip.x) / 2 * w)
        abdomen_y = int((left_hip.y + right_hip.y) / 2 * h - 40)
        
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h + 15)
        
        self.tracking_points = {
            'chest': (chest_x, chest_y),
            'abdomen': (abdomen_x, abdomen_y),
            'nose': (nose_x, nose_y)
        }
        
    def extract_region_signal(self, frame, center_point, block_size=15):
        """Extract BGR signal from region"""
        x, y = center_point
        half_block = block_size // 2
        h, w = frame.shape[:2]
        
        y_start = max(y - half_block, 0)
        y_end = min(y + half_block, h)
        x_start = max(x - half_block, 0)
        x_end = min(x + half_block, w)
        
        region = frame[y_start:y_end, x_start:x_end]
        
        if region.size == 0:
            return None
        
        return {
            'B': np.mean(region[:, :, 0]),
            'G': np.mean(region[:, :, 1]),
            'R': np.mean(region[:, :, 2])
        }
    
    def analyze_video(self):
        """Process entire video and extract breathing signals"""
        frame_idx = 0
        
        print("Analyzing video frames...")
        print(f"Progress: ", end='', flush=True)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Progress indicator
            if frame_idx % 30 == 0:
                progress = (frame_idx / self.total_frames) * 100
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Initialize tracking points on first detection
                if self.tracking_points is None:
                    self.initialize_tracking_points(landmarks, frame.shape)
                    print(f"\n✓ Infant detected! Tracking points initialized.")
                
                # Extract signals from each region
                for region_name, point in self.tracking_points.items():
                    signals = self.extract_region_signal(frame, point)
                    if signals:
                        for channel in ['B', 'G', 'R']:
                            self.signal_history[region_name][channel].append(signals[channel])
            
            frame_idx += 1
        
        print(f"\n✓ Analysis complete! Processed {frame_idx} frames.\n")
        self.cap.release()
        
    def calculate_breathing_rate(self):
        """Calculate breathing rate from collected signals"""
        print("Calculating breathing rate...")
        
        all_rates = []
        
        # Analyze each region
        for region in ['chest', 'abdomen', 'nose']:
            region_signals = []
            
            # Process each BGR channel
            for channel in ['B', 'G', 'R']:
                sig = np.array(self.signal_history[region][channel])
                
                if len(sig) > 60:  # Need at least 2 seconds
                    try:
                        # Apply bandpass filter for infant breathing (0.33-1.0 Hz = 20-60 breaths/min)
                        filtered = apply_bandpass_filter(sig, 0.33, 1.0, self.fps, order=4)
                        region_signals.append(filtered)
                    except:
                        region_signals.append(sig - np.mean(sig))
            
            if len(region_signals) == 0:
                continue
            
            # Average across BGR channels
            combined_signal = np.mean(region_signals, axis=0)
            
            # Find peaks
            min_distance = int(self.fps * 0.6)  # Minimum 0.6 seconds between breaths
            
            signal_std = np.std(combined_signal)
            if signal_std < 1e-6:
                continue
            
            prominence = signal_std * 0.25
            
            peaks, properties = find_peaks(
                combined_signal,
                distance=min_distance,
                prominence=prominence
            )
            
            if len(peaks) >= 2:
                # Calculate breathing rate
                time_duration = len(combined_signal) / self.fps
                breathing_rate = (len(peaks) / time_duration) * 60
                
                # Sanity check for infant range
                breathing_rate = max(15, min(breathing_rate, 70))
                
                all_rates.append({
                    'region': region,
                    'rate': breathing_rate,
                    'peaks': len(peaks),
                    'signal_strength': signal_std,
                    'confidence': min(100, signal_std * 2000 + len(peaks) * 10)
                })
                
                print(f"  {region.capitalize():8s}: {breathing_rate:5.1f} BPM (peaks: {len(peaks):2d}, confidence: {all_rates[-1]['confidence']:.0f}%)")
        
        return all_rates
    
    def plot_results(self, results):
        """Create visualization of results"""
        if len(self.signal_history['chest']['G']) == 0:
            print("⚠️  No data to plot")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Time array
        time_array = np.arange(len(self.signal_history['chest']['G'])) / self.fps
        
        # Plot 1: Raw signals
        for region in ['chest', 'abdomen', 'nose']:
            sig = np.array(self.signal_history[region]['G'])
            if len(sig) > 0:
                axes[0].plot(time_array[:len(sig)], sig, label=region.capitalize(), alpha=0.7)
        axes[0].set_title('Raw Green Channel Signals', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Intensity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Filtered chest signal
        chest_sig = np.array(self.signal_history['chest']['G'])
        if len(chest_sig) > 60:
            try:
                filtered = apply_bandpass_filter(chest_sig, 0.33, 1.0, self.fps, order=4)
                axes[1].plot(time_array[:len(filtered)], filtered, 'r-', linewidth=2)
                axes[1].set_title('Filtered Chest Signal (Breathing Band: 0.33-1.0 Hz)', fontsize=12, fontweight='bold')
                axes[1].set_ylabel('Amplitude')
                axes[1].grid(True, alpha=0.3)
            except:
                pass
        
        # Plot 3: Breathing rate summary
        if results:
            regions = [r['region'] for r in results]
            rates = [r['rate'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            colors = ['red' if r['region'] == 'chest' else 'green' if r['region'] == 'abdomen' else 'cyan' for r in results]
            bars = axes[2].bar(regions, rates, color=colors, alpha=0.7)
            axes[2].axhline(y=20, color='orange', linestyle='--', label='Min Normal (20)')
            axes[2].axhline(y=60, color='orange', linestyle='--', label='Max Normal (60)')
            axes[2].set_title('Breathing Rate by Region', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Breaths per Minute')
            axes[2].set_ylim(0, 80)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Add confidence labels on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{conf:.0f}%\nconf.',
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Summary text
        axes[3].axis('off')
        if results:
            # Calculate weighted average
            total_conf = sum(r['confidence'] for r in results)
            if total_conf > 0:
                weighted_avg = sum(r['rate'] * r['confidence'] for r in results) / total_conf
            else:
                weighted_avg = np.mean([r['rate'] for r in results])
            
            best_result = max(results, key=lambda x: x['confidence'])
            
            summary = f"""
INFANT BREATHING ANALYSIS RESULTS
{'='*50}

Video Duration: {self.duration:.1f} seconds
Analysis Quality: {'GOOD' if best_result['confidence'] > 50 else 'MODERATE' if best_result['confidence'] > 30 else 'LOW'}

BREATHING RATE (Most Confident):
  Region: {best_result['region'].upper()}
  Rate: {best_result['rate']:.1f} breaths/minute
  Confidence: {best_result['confidence']:.0f}%

WEIGHTED AVERAGE (All Regions):
  Rate: {weighted_avg:.1f} breaths/minute

REFERENCE RANGES:
  Newborn (0-3 months): 30-60 breaths/min
  Infant (3-12 months): 24-40 breaths/min
  Toddler (1-3 years): 20-30 breaths/min

STATUS: {'✓ NORMAL' if 20 <= weighted_avg <= 60 else '⚠️ OUTSIDE NORMAL RANGE'}
            """
            axes[3].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                        verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.video_path.replace('.mp4', '_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Analysis chart saved: {output_path}")
        
        plt.show()


def main():
    """Main analysis function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_infant_video.py <video_file.mp4>")
        print("\nExample: python analyze_infant_video.py 002.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("INFANT BREATHING RATE ANALYZER")
    print("Research-Validated Multi-Point BGR Analysis")
    print("="*60)
    
    # Create analyzer
    analyzer = InfantVideoAnalyzer(video_path)
    
    # Analyze video
    analyzer.analyze_video()
    
    # Calculate breathing rate
    results = analyzer.calculate_breathing_rate()
    
    if not results:
        print("\n❌ Could not calculate breathing rate. Possible issues:")
        print("  - Infant not clearly visible")
        print("  - Video too short (need at least 5 seconds)")
        print("  - Poor lighting conditions")
        sys.exit(1)
    
    # Display results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    # Get best result
    best_result = max(results, key=lambda x: x['confidence'])
    
    # Calculate weighted average
    total_conf = sum(r['confidence'] for r in results)
    if total_conf > 0:
        weighted_avg = sum(r['rate'] * r['confidence'] for r in results) / total_conf
    else:
        weighted_avg = np.mean([r['rate'] for r in results])
    
    print(f"\n🎯 INFANT BREATHING RATE: {weighted_avg:.1f} breaths/minute")
    print(f"   (Based on {len(results)} region(s): {', '.join(r['region'] for r in results)})")
    print(f"\n   Most Confident: {best_result['region'].upper()} at {best_result['rate']:.1f} BPM")
    print(f"   Confidence: {best_result['confidence']:.0f}%")
    
    # Status
    if 20 <= weighted_avg <= 60:
        print(f"\n✓ STATUS: NORMAL (within 20-60 breaths/min range)")
    elif weighted_avg < 20:
        print(f"\n⚠️  STATUS: BELOW NORMAL (< 20 breaths/min)")
    else:
        print(f"\n⚠️  STATUS: ABOVE NORMAL (> 60 breaths/min)")
    
    print(f"\n{'='*60}\n")
    
    # Plot results
    analyzer.plot_results(results)
    
    # Cleanup
    analyzer.pose.close()


if __name__ == "__main__":
    main()

