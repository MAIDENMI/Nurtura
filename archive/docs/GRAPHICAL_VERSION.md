# Graphical Breathing Monitor

## Overview

The graphical version displays **real-time graphs** of breathing patterns, similar to medical heart rate monitors!

## Features

### 📊 Two Live Graphs

1. **Breathing Rate Graph (Top)**
   - Shows breathing rate in breaths per minute (BPM)
   - Color-coded zones:
     - 🟢 **Green zone (20-60 BPM)**: Normal infant breathing
     - 🔴 **Red zones**: Abnormal rates (too high/low)
   - 30-second rolling window

2. **Motion Waveform (Bottom)**
   - Real-time breathing motion signal
   - See each breath as it happens
   - Threshold line shows detection sensitivity
   - Similar to ECG/EKG waveforms

### 🎨 Visual Feedback

- **Color-coded status**:
  - 🟢 Green: Normal breathing (20-60 BPM)
  - 🔴 Red: LOW (< 20 BPM) or HIGH (> 60 BPM)
  - ⚪ White: Detecting...

- **Two windows**:
  - Main camera view with torso detection
  - Separate graph window with real-time plots

## Usage

### Quick Start

```bash
./run.sh graph              # macOS/Linux
run.bat graph              # Windows
```

### Manual Run

```bash
source venv/bin/activate
python breathing_monitor_graphical.py
```

## Controls

- **'q'** - Quit application
- **'s'** - Save screenshot (saves both video frame AND graphs!)

Screenshots are saved as:
- `screenshot_N.jpg` - Camera view
- `graph_N.jpg` - Graph view

## What You'll See

### Window 1: "Infant Breathing Monitor"
- Live camera feed
- Green box around detected torso
- Current breathing rate with color-coded status
- Motion intensity value

### Window 2: "Breathing Graphs"
- **Top graph**: Breathing rate over time
  - Y-axis: Breaths per minute (0-80)
  - X-axis: Time in seconds
  - Normal range highlighted in green

- **Bottom graph**: Motion signal waveform
  - Y-axis: Motion intensity
  - X-axis: Time in seconds
  - Red dashed line: Detection threshold
  - Watch the breathing pattern like a heart monitor!

## Understanding the Graphs

### Breathing Rate Graph
```
80 ┤           ╭─╮                    HIGH (Red zone)
60 ┤───────────┼─┼────────────────    Normal range (Green)
40 ┤      ╭────╯ ╰───╮
20 ┤──────┼──────────┼─────────────    Normal range
 0 ┤      ╰──────────╰────────────    LOW (Red zone)
   └─────────────────────────────▶
        Time (seconds)
```

### Motion Waveform
```
   ┤     ╱╲    ╱╲    ╱╲    ╱╲       <- Breathing cycles
   ┤    ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲      
   ┤   ╱    ╲╱    ╲╱    ╲╱    ╲     
   ┤──╱──────────────────────────▶   <- Threshold
   └─────────────────────────────▶
        Time (seconds)
```

Each peak/valley represents a breath!

## Normal Breathing Rates

For reference:
- **Newborn (0-3 months)**: 30-60 breaths/min
- **Infant (3-12 months)**: 24-40 breaths/min
- **Toddler (1-3 years)**: 20-30 breaths/min

The graphs show these ranges visually!

## Performance Notes

- Graphs update every 5 frames for smooth performance
- 30-second rolling window (adjustable)
- Dark theme for easy viewing
- Optimized for real-time display

## Troubleshooting

### Graphs Not Showing
```bash
# Make sure matplotlib is installed
source venv/bin/activate
pip install matplotlib
```

### Graphs Flickering
- This is normal on some systems
- Try reducing window size in code if needed

### Slow Performance
- Close other applications
- Reduce camera resolution in code:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  ```

### Display Issues on macOS
- Graphs use matplotlib backend
- Should work out-of-the-box on macOS
- If issues, try: `export MPLBACKEND=MacOSX`

## Customization

Edit `breathing_monitor_graphical.py` to customize:

### Change Graph Colors
```python
self.line1.plot([], [], color='#00ff00')  # Breathing rate (green)
self.line2.plot([], [], color='#00ffff')  # Motion (cyan)
```

### Adjust Time Window
```python
monitor = BreathingMonitorGraphical(window_size=60)  # 60 frames
# Increase for more history, decrease for faster response
```

### Change Graph Update Rate
```python
if frame_count % 5 == 0:  # Update every 5 frames
# Lower number = faster updates (but more CPU usage)
```

## Comparison with Other Versions

| Feature | Basic | Advanced | **Graphical** |
|---------|-------|----------|---------------|
| Breathing rate display | ✅ | ✅ | ✅ |
| Pose detection | ✅ | ✅ | ✅ |
| Configuration file | ❌ | ✅ | ❌ |
| Data logging | ❌ | ✅ | ❌ |
| **Real-time graphs** | ❌ | ❌ | **✅** |
| **Visual waveform** | ❌ | ❌ | **✅** |
| **Color zones** | ❌ | ❌ | **✅** |
| Screenshots with graphs | ❌ | ❌ | **✅** |

## Medical Disclaimer

⚠️ **This is NOT a medical device!** 

The graphical display is for:
- Educational purposes
- Research and development
- Demonstration of breathing monitoring concepts

**Always consult healthcare professionals for medical monitoring.**

## Tips for Best Results

1. **Good lighting** - Graphs are more stable with consistent lighting
2. **Stable camera** - Mount on tripod for cleaner waveforms
3. **Wait 10-15 seconds** - Graphs need time to establish baseline
4. **Watch the waveform** - Each peak/valley is a breath!
5. **Screenshot feature** - Capture interesting patterns with 's' key

## Future Enhancements

Possible additions:
- [ ] Heart rate detection (from face color changes)
- [ ] Audio alerts for abnormal rates
- [ ] Export graphs as video
- [ ] Multiple infant tracking
- [ ] Historical data playback
- [ ] Statistical analysis overlay

## Example Session

```bash
$ ./run.sh graph

Starting Graphical Breathing Monitor with Real-Time Graphs...
============================================================
Infant Breathing Monitor - Graphical Version
============================================================

Initializing...
✓ Camera opened

Controls:
  'q' - Quit
  's' - Screenshot

Starting monitor...

# Two windows open:
# 1. Camera feed with torso detection
# 2. Real-time graphs updating continuously

# Press 's' to save:
✓ Screenshots saved: screenshot_1.jpg & graph_1.jpg

# Press 'q' to quit:
Shutting down...
✓ Cleanup complete
```

Enjoy monitoring with beautiful, real-time graphs! 📊✨

