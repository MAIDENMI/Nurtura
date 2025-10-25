# Infant Breathing Monitor AI

A lightweight, real-time breathing monitoring system using computer vision and pose detection. Designed to run efficiently on Raspberry Pi devices.

## Features

- **Automatic Torso Detection**: Uses MediaPipe Pose to automatically detect and track the infant's torso
- **Motion-Based Breathing Detection**: Measures subtle motion using optical flow analysis
- **Real-Time Monitoring**: Displays breathing rate in breaths per minute
- **Raspberry Pi Optimized**: Uses lightweight models and efficient processing
- **Simple UI**: Visual feedback with bounding box and breathing rate display

## How It Works

1. **Pose Detection**: MediaPipe Pose detects key body landmarks (shoulders and hips)
2. **Torso Extraction**: Automatically crops the torso region based on detected landmarks
3. **Motion Analysis**: Optical flow calculates subtle movements in the torso area
4. **Breathing Rate Estimation**: Peak detection algorithm estimates breaths per minute
5. **Visual Feedback**: Real-time display of results with overlay graphics

## Installation

### Quick Setup (All Platforms)

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

The setup script will:
- ✓ Create an isolated virtual environment
- ✓ Install all dependencies
- ✓ Verify the installation
- ✓ Create helper scripts

### Manual Installation

#### On Raspberry Pi

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y libatlas-base-dev libhdf5-dev

# Run automated setup
./setup.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### On Desktop (macOS/Linux/Windows)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**⚠️ Always use a virtual environment to avoid conflicts with system packages!**

## Usage

### Test Your Camera First

Before running the breathing monitor, test your camera:

```bash
python test_camera.py
```

This will help you verify:
- Camera is connected and working
- Correct camera index
- Video feed quality

### Basic Usage

```bash
python breathing_monitor.py
```

- The camera will start automatically
- Position the infant so their torso is visible to the camera
- The system will display a green bounding box around the detected torso
- Breathing rate will be shown at the top of the screen
- Press **'q'** to quit

### 📊 Graphical Version (NEW!)

**Want to see real-time graphs like a heart rate monitor?**

```bash
./run.sh graph                    # macOS/Linux
run.bat graph                     # Windows
```

The graphical version includes:
- **Real-time breathing rate graph** with color-coded zones
- **Live motion waveform** showing each breath
- **Visual feedback** similar to medical monitors
- **Dual window display** (camera + graphs)
- See `GRAPHICAL_VERSION.md` for details

### Advanced Usage (with Configuration)

For more control and features, use the advanced version:

```bash
python breathing_monitor_advanced.py
```

The advanced version includes:
- Configuration file support (`config.py`)
- Data logging to CSV
- Alert system for abnormal breathing rates
- FPS display
- Screenshot capture (press 's')
- Better error handling

**To customize settings:**
1. Open `config.py`
2. Adjust parameters as needed
3. Run the advanced version

### Configuration

You can adjust the sensitivity and window size by modifying the `BreathingMonitor` initialization:

```python
monitor = BreathingMonitor(
    window_size=30,          # Number of frames for moving average (default: 30)
    breathing_threshold=0.02  # Sensitivity for motion detection (default: 0.02)
)
```

**Parameters:**
- `window_size`: Larger values = smoother but slower response (recommended: 20-60)
- `breathing_threshold`: Lower values = more sensitive to small movements (recommended: 0.01-0.05)

## Expected Breathing Rates

For reference, typical infant breathing rates are:
- **Newborn (0-3 months)**: 30-60 breaths/min
- **Infant (3-12 months)**: 24-40 breaths/min
- **Toddler (1-3 years)**: 20-30 breaths/min

## Performance Tips

### For Raspberry Pi

1. **Use a Raspberry Pi 4** with at least 2GB RAM for best performance
2. **Reduce camera resolution** if needed:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
   ```
3. **Run headless** for better performance (without display):
   - Install `opencv-python-headless` instead of `opencv-python`
   - Save frames to file or stream over network instead of displaying

4. **Disable desktop environment** on Raspberry Pi OS Lite for maximum performance

### General Optimization

- Ensure good lighting conditions
- Keep the camera stable (mount it)
- Minimize background motion
- Position camera 1-2 meters from the infant

## Troubleshooting

### Camera Not Detected
```bash
# Test camera access
ls /dev/video*

# Try different camera index
cap = cv2.VideoCapture(1)  # Instead of 0
```

### Low Frame Rate on Raspberry Pi
- Reduce camera resolution
- Ensure adequate power supply (5V 3A recommended)
- Close other applications
- Use Raspberry Pi 4 or newer

### Pose Not Detected
- Ensure infant's torso is fully visible
- Improve lighting conditions
- Check if infant is too close or too far from camera
- Adjust `min_detection_confidence` (lower for easier detection)

### Inaccurate Breathing Rate
- Adjust `breathing_threshold` parameter
- Increase `window_size` for smoother readings
- Ensure minimal background motion
- Verify infant is lying relatively still (except for breathing)

## System Requirements

**Minimum:**
- Raspberry Pi 3B+ or equivalent
- 1GB RAM
- USB camera or Raspberry Pi Camera Module
- Python 3.7+

**Recommended:**
- Raspberry Pi 4 (2GB+ RAM)
- Good quality camera (720p or higher)
- Stable mounting solution
- Python 3.9+

## Important Notes

⚠️ **Medical Disclaimer**: This system is for educational and research purposes only. It is NOT a medical device and should NOT be used as a substitute for professional medical monitoring equipment or advice. Always consult healthcare professionals for infant health monitoring.

## Project Structure

```
Testch/
├── breathing_monitor.py           # Basic version (standalone)
├── breathing_monitor_advanced.py  # Advanced version (uses config.py)
├── breathing_monitor_graphical.py # With real-time graphs! 📊
├── config.py                      # Configuration settings
├── config_raspberry_pi.py         # Raspberry Pi optimized config
├── test_camera.py                 # Camera testing utility
├── setup.sh / setup.bat           # Automated setup scripts
├── run.sh / run.bat               # Quick run scripts
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
├── README.md                      # This file (full documentation)
├── QUICKSTART.md                  # 5-minute quick start guide
├── INSTALL.md                     # Detailed installation guide
├── GRAPHICAL_VERSION.md           # Guide for graphical version
├── MACOS_CAMERA_PERMISSIONS.md    # macOS camera setup
└── venv/                          # Virtual environment (after setup)
```

### Which Version Should I Use?

**Basic Version (`breathing_monitor.py`):**
- Single file, easy to understand
- No dependencies on other project files
- Good for learning and simple testing
- Manual parameter adjustment in code

**Enhanced Version (`breathing_monitor_enhanced.py`):** ⭐ **RECOMMENDED!**
- **Based on published research** (MICCAI 2023)
- Improved optical flow parameters
- Better chest ROI selection
- Signal processing with bandpass filtering
- **Confidence scoring** shows measurement quality
- Research-validated techniques
- Best accuracy for infant breathing

**Graphical Version (`breathing_monitor_graphical.py`):**
- **Real-time graphs and waveforms**
- Visual breathing pattern display
- Color-coded zones (normal/abnormal)
- Like a medical heart rate monitor
- Perfect for demonstrations and monitoring

**Advanced Version (`breathing_monitor_advanced.py`):**
- Modular design with configuration file
- Data logging and alerts
- More features (FPS display, screenshots)
- Easier parameter tuning without editing code
- Better for deployment and production use

**Video Analysis (`breathing_monitor_video.py`):**
- Test with pre-recorded videos
- Perfect for validating on infant videos
- Playback controls and statistics
- No camera needed

## Future Improvements

- [ ] Add alert system for abnormal breathing rates
- [ ] Log breathing data to CSV file
- [ ] Add multiple infant tracking
- [ ] Implement deep learning-based respiration detection
- [ ] Add web interface for remote monitoring
- [ ] Support for thermal cameras (more accurate in darkness)

## Technical Details

### Libraries Used
- **OpenCV**: Video capture and optical flow analysis
- **MediaPipe**: Lightweight pose detection
- **NumPy**: Numerical computations

### Algorithm Overview
1. Capture video frame
2. Convert to RGB for MediaPipe
3. Detect pose landmarks
4. Extract torso bounding box from shoulder/hip landmarks
5. Calculate optical flow between consecutive torso frames
6. Track motion magnitude over time
7. Detect peaks in motion signal
8. Convert peaks to breaths per minute

## License

This project is provided as-is for educational purposes.

## Contributing

Feel free to submit issues and enhancement requests!

