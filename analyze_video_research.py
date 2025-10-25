#!/usr/bin/env python3
"""
Video Analysis with Research-Validated Methods
Analyzes pre-recorded videos with age detection, breathing rate, and heart rate (rPPG)
"""

import sys
import cv2

# Import the research monitor class
from breathing_monitor_research import BreathingMonitorResearch

def main():
    video_path = '/Users/aidenm/Testch/005.mp4'
    
    print("="*60)
    print("Video Analysis - Research-Validated Monitor")
    print("Age Detection + Breathing + Heart Rate (rPPG)")
    print("="*60)
    print(f"Analyzing: {video_path}")
    print()
    
    # Initialize monitor
    monitor = BreathingMonitorResearch()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f}fps")
    print(f"⏱️  Duration: {duration:.1f} seconds ({frame_count} frames)")
    print()
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Screenshot")
    print("  SPACE - Pause/Resume")
    print()
    print("Starting analysis...")
    print()
    
    frame_num = 0
    paused = False
    
    # Statistics tracking
    breathing_rates = []
    heart_rates = []
    confidences = []
    hr_confidences = []
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video analysis complete!")
                break
            
            frame_num += 1
            
            # Process frame
            processed_frame = monitor.process_frame(frame)
            
            # Get plot
            plot_image = monitor.update_plot()
            
            # Combine views
            # Resize for display
            display_frame = cv2.resize(processed_frame, (960, 540))
            display_plot = cv2.resize(plot_image, (960, 540))
            
            # Stack horizontally
            combined = cv2.hconcat([display_frame, display_plot])
            
            # Add progress bar
            progress = frame_num / frame_count
            bar_width = combined.shape[1]
            bar_height = 10
            progress_bar = int(progress * bar_width)
            
            cv2.rectangle(combined, (0, combined.shape[0] - bar_height), 
                         (progress_bar, combined.shape[0]), (0, 255, 0), -1)
            cv2.rectangle(combined, (progress_bar, combined.shape[0] - bar_height),
                         (bar_width, combined.shape[0]), (50, 50, 50), -1)
            
            # Add frame info
            elapsed = frame_num / fps if fps > 0 else 0
            info_text = f"Frame: {frame_num}/{frame_count} | Time: {elapsed:.1f}/{duration:.1f}s | Progress: {progress*100:.1f}%"
            cv2.putText(combined, info_text, (10, combined.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Video Analysis', combined)
            
            # Collect statistics
            if monitor.breathing_rate > 0:
                breathing_rates.append(monitor.breathing_rate)
            if monitor.heart_rate > 0:
                heart_rates.append(monitor.heart_rate)
            confidences.append(monitor.confidence)
            hr_confidences.append(monitor.hr_confidence)
            
            # Print progress every 5 seconds
            if frame_num % (fps * 5) == 0:
                print(f"⏱️  {elapsed:.0f}s | Age: {monitor.age_category} | BR: {monitor.breathing_rate:.1f} ({monitor.confidence:.0f}%) | HR: {monitor.heart_rate:.0f} ({monitor.hr_confidence:.0f}%)")
        
        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("\n⚠️  Analysis stopped by user")
            break
        elif key == ord('s'):
            screenshot_name = f'analysis_frame_{frame_num}.jpg'
            cv2.imwrite(screenshot_name, combined)
            print(f"📸 Screenshot saved: {screenshot_name}")
        elif key == ord(' '):
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 Analysis Summary")
    print("="*60)
    
    if len(breathing_rates) > 0:
        import numpy as np
        print(f"\n👤 Detected: {monitor.age_category.upper()} (Est. {monitor.body_size_cm:.0f}cm)")
        print(f"\n💨 Breathing Rate:")
        print(f"   Average: {np.mean(breathing_rates):.1f} BPM")
        print(f"   Min: {np.min(breathing_rates):.1f} BPM")
        print(f"   Max: {np.max(breathing_rates):.1f} BPM")
        print(f"   Std Dev: {np.std(breathing_rates):.1f}")
        print(f"   Avg Confidence: {np.mean(confidences):.1f}%")
        
        if len(heart_rates) > 0:
            print(f"\n💓 Heart Rate:")
            print(f"   Average: {np.mean(heart_rates):.0f} BPM")
            print(f"   Min: {np.min(heart_rates):.0f} BPM")
            print(f"   Max: {np.max(heart_rates):.0f} BPM")
            print(f"   Std Dev: {np.std(heart_rates):.1f}")
            print(f"   Avg Confidence: {np.mean(hr_confidences):.1f}%")
        
        # Age-appropriate assessment
        ranges = monitor.get_normal_ranges()
        print(f"\n📋 Assessment ({ranges['label']}):")
        
        avg_br = np.mean(breathing_rates)
        if avg_br < ranges['br_low']:
            br_status = "⚠️  LOW"
        elif avg_br > ranges['br_high']:
            br_status = "⚠️  HIGH"
        else:
            br_status = "✅ NORMAL"
        print(f"   Breathing: {br_status} (Normal: {ranges['br_normal'][0]}-{ranges['br_normal'][1]})")
        
        if len(heart_rates) > 0:
            avg_hr = np.mean(heart_rates)
            if avg_hr < ranges['hr_low']:
                hr_status = "⚠️  LOW"
            elif avg_hr > ranges['hr_high']:
                hr_status = "⚠️  HIGH"
            else:
                hr_status = "✅ NORMAL"
            print(f"   Heart Rate: {hr_status} (Normal: {ranges['hr_normal'][0]}-{ranges['hr_normal'][1]})")
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

