# Quick Start Guide

Get up and running with the Infant Breathing Monitor in 5 minutes!

## Step 1: Automated Setup (Recommended)

### macOS / Linux:
```bash
./setup.sh
```

### Windows:
```cmd
setup.bat
```

This will automatically:
- Create an isolated virtual environment
- Install all dependencies
- Verify the installation
- Create helper scripts

### Manual Setup (Alternative):
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**✨ Using a virtual environment prevents package conflicts with your global Python installation!**

## Step 2: Test Your Camera

### Using Quick Run Scripts:
```bash
./run.sh test          # macOS/Linux
run.bat test          # Windows
```

### Manual:
```bash
source venv/bin/activate  # Activate venv first!
python test_camera.py
```

✓ You should see a live video feed. If not, see troubleshooting below.

## Step 3: Run the Monitor

### Using Quick Run Scripts (Easy):
```bash
./run.sh              # Basic version (macOS/Linux)
./run.sh graph        # 📊 With real-time graphs! (macOS/Linux)
./run.sh advanced     # Advanced version (macOS/Linux)

run.bat               # Basic version (Windows)
run.bat graph         # 📊 With real-time graphs! (Windows)
run.bat advanced      # Advanced version (Windows)
```

**Recommended:** Try the **graphical version** first to see beautiful real-time graphs!

### Manual:
```bash
source venv/bin/activate  # Activate venv first!
python breathing_monitor.py           # Basic
python breathing_monitor_graphical.py # 📊 With graphs!
python breathing_monitor_advanced.py  # Advanced
```

## Step 4: Position the Camera

1. **Distance**: 1-2 meters from the infant
2. **Angle**: Point camera at infant's torso (chest area)
3. **Lighting**: Ensure good, even lighting
4. **Stability**: Mount camera or keep it very stable

## What You'll See

- **Green box**: Detected torso region
- **Breathing rate**: Displayed at top in breaths/min
- **Normal range**: 20-60 breaths/min for infants

## Controls

- **q**: Quit the application
- **s**: Take screenshot (advanced version only)

## Troubleshooting

### Camera Not Found
```bash
# Try different camera index
python test_camera.py 1
# or
python test_camera.py 2
```

Then update `config.py`:
```python
CAMERA_INDEX = 1  # Change to your camera index
```

### Pose Not Detected
1. Make sure infant's torso is fully visible
2. Improve lighting
3. Lower detection confidence in `config.py`:
   ```python
   MIN_DETECTION_CONFIDENCE = 0.3  # Try lower value
   ```

### Low FPS on Raspberry Pi
Edit `config.py`:
```python
IS_RASPBERRY_PI = True  # Auto-optimizes settings
```

Or manually adjust:
```python
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
MODEL_COMPLEXITY = 0
TORSO_RESIZE = (64, 64)
```

### Inaccurate Readings
Adjust sensitivity in `config.py`:
```python
BREATHING_THRESHOLD = 0.03  # Increase if too sensitive
WINDOW_SIZE = 45            # Increase for smoother readings
```

## Tips for Best Results

1. **Keep the environment stable**
   - Minimize background motion
   - Avoid moving curtains, fans
   - Keep pets away

2. **Ensure good lighting**
   - Use natural or soft artificial light
   - Avoid harsh shadows
   - Keep lighting constant

3. **Camera placement**
   - Use a tripod or stable mount
   - Position at slight downward angle
   - Ensure infant's full torso is visible

4. **Calibration period**
   - Wait 10-15 seconds after starting
   - System needs time to establish baseline
   - Initial readings may be inaccurate

## Next Steps

- **Enable logging**: Set `ENABLE_LOGGING = True` in `config.py` to save data
- **Enable alerts**: Set `ENABLE_ALERTS = True` for abnormal rate warnings
- **Customize display**: Adjust colors, text size in `config.py`
- **Read full README**: See `README.md` for detailed information

## Important Reminder

⚠️ **This is NOT a medical device!** Use only for educational purposes. Always consult healthcare professionals for infant monitoring.

## Need Help?

Check the full README.md for:
- Detailed setup instructions
- Configuration options
- Technical details
- Performance optimization tips

