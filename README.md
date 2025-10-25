# 👶 Infant Breathing & Heart Rate Monitor AI

An open-source AI-powered monitoring system that uses computer vision to track both **breathing rate** and **heart rate** in real-time using just a standard webcam.

## 🌟 Features

### Dual Monitoring System
- ✅ **Breathing Rate Detection** - Multi-region motion tracking (chest, abdomen, nose)
- ✅ **Heart Rate Detection** - Research-validated rPPG (remote photoplethysmography)
- ✅ **Real-Time Analysis** - Live camera feed with instant measurements
- ✅ **Video Analysis** - Analyze pre-recorded videos
- ✅ **Confidence Scoring** - Quality metrics for both measurements
- ✅ **Visual Feedback** - Color-coded alerts and status indicators
- ✅ **Multi-Platform** - Works on Windows, macOS, Raspberry Pi

### Research-Validated Methods
- **Breathing Detection**: BGR channel analysis with bandpass filtering (0.12-0.75 Hz)
- **Heart Rate Detection**: rPPG method based on van der Kooij & Naber (2019) - [Published Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC6797647/)
  - 95-97% accuracy compared to pulse oximetry
  - Uses GREEN channel analysis of facial skin
  - Detects subtle color changes from blood flow
  - Works with standard 30fps webcams

## 📊 What It Measures

| Measurement | Method | Normal Range (Infants) | Accuracy |
|------------|--------|----------------------|----------|
| **Breathing Rate** | Motion + BGR analysis | 20-60 breaths/min | High |
| **Heart Rate** | rPPG (facial color changes) | 100-160 BPM | 95-97%* |

*Based on van der Kooij & Naber (2019) validation study

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Nurtura-AI/breathing-monitor.git
cd breathing-monitor

# Setup (automatic installation)
# macOS/Linux:
bash setup.sh

# Windows:
setup.bat
```

### 2. Run the Monitor

```bash
# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Run research-validated version (Breathing + Heart Rate)
python breathing_monitor_research.py
```

## 📹 What You'll See

### Live Camera Display
```
Breathing: 32.4 BPM -- NORMAL    [Color-coded status]
Heart Rate: 122 BPM -- NORMAL    [Color-coded status]
BR Conf: 78%  |  HR Conf: 65%   [Quality indicators]
```

### Tracking Points
- 🔴 **Red** - Chest (breathing)
- 🟢 **Green** - Abdomen (breathing)
- 🔵 **Cyan** - Nose (breathing + heart rate via rPPG)
- ⚪ **Gray** - Control (background reference)

### Real-Time Graphs
1. **Breathing Rate** - Cyan line showing breaths per minute
2. **Heart Rate** - Red line showing beats per minute (rPPG)
3. **Signal Quality** - Raw signals from all tracking regions
4. **Confidence Scores** - Quality metrics for both measurements

## 🎯 How It Works

### Breathing Detection
1. **Pose Detection** - MediaPipe identifies body landmarks
2. **Multi-Region Tracking** - Monitors chest, abdomen, and nose movement
3. **BGR Analysis** - Extracts color changes from tracked regions
4. **Bandpass Filtering** - Isolates breathing frequencies (0.12-0.75 Hz)
5. **Peak Detection** - Identifies breath cycles
6. **Weighted Averaging** - Combines signals for robust measurement

### Heart Rate Detection (rPPG)
1. **Facial Tracking** - Focuses on nose region (most stable facial area)
2. **GREEN Channel Extraction** - Hemoglobin absorbs green light most
3. **Bandpass Filtering** - Isolates cardiac frequencies (0.7-4.0 Hz = 42-240 BPM)
4. **Peak Detection** - Identifies heartbeats from subtle color changes
5. **Confidence Scoring** - Evaluates signal quality and peak regularity

**The Science**: When your heart beats, it pumps blood through facial capillaries, causing tiny color changes invisible to the human eye but detectable by cameras. This is the same principle used in medical pulse oximeters!

## 📖 Documentation

### 🌟 Start Here
- [**HOW_IT_WORKS.md**](HOW_IT_WORKS.md) - **Complete explanation of how breathing & heart rate detection works** ⭐
- [**ACCURACY_FAQ.md**](ACCURACY_FAQ.md) - **Detailed accuracy info, validation, when to trust readings** ⭐

### Technical Details
- [**RPPG_HEART_RATE.md**](RPPG_HEART_RATE.md) - Heart rate detection method (rPPG)
- [**ENHANCED_TRACKING.md**](ENHANCED_TRACKING.md) - Body tracking improvements
- [**AGE_DETECTION.md**](AGE_DETECTION.md) - Age classification system

### Getting Started
- [**QUICKSTART.md**](QUICKSTART.md) - Step-by-step guide for beginners
- [**INSTALL.md**](INSTALL.md) - Installation troubleshooting
- [**TEST_WITH_VIDEO.md**](TEST_WITH_VIDEO.md) - Testing with video files
- [**VERSION_COMPARISON.md**](VERSION_COMPARISON.md) - Compare different monitor versions

## 🔬 Research Foundation

### Heart Rate Detection
**Van der Kooij, K., & Naber, M. (2019)**. "An open-source remote heart rate imaging method with practical apparatus and algorithms." *Behavior Research Methods*, 51(5), 2106-2119.
- DOI: https://doi.org/10.3758/s13428-019-01256-8
- Full Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6797647/

**Key Validation Results:**
- 97.8% accuracy at rest vs pulse oximetry
- 95.3% accuracy after exercise (high heart rates)
- Works with consumer 30fps cameras
- Robust under ambient lighting conditions

## 🖥️ System Requirements

### Minimum
- Python 3.11+
- Webcam (30fps recommended)
- 4GB RAM
- Modern CPU (dual-core+)

### Recommended
- Python 3.11+
- HD Webcam (720p, 30fps)
- 8GB RAM
- Quad-core CPU
- Good lighting conditions

### Tested Platforms
- ✅ macOS (M1/M2/Intel)
- ✅ Windows 10/11
- ✅ Raspberry Pi 4 (with optimizations)
- ✅ Linux (Ubuntu 20.04+)

## 🎮 Available Versions

| File | Features | Best For |
|------|----------|----------|
| `breathing_monitor_research.py` | **Breathing + Heart Rate** (rPPG) | Most complete monitoring |
| `breathing_monitor_graphical.py` | Breathing + graphs | Breathing-only with visualization |
| `breathing_monitor.py` | Basic breathing | Simple, fast monitoring |
| `analyze_infant_video.py` | Video file analysis | Post-recording analysis |

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection settings
BREATHING_THRESHOLD = 0.03  # Sensitivity
WINDOW_SIZE = 75  # Analysis window (frames)

# Alert thresholds
BREATHING_RATE_LOW = 20
BREATHING_RATE_HIGH = 60
HEART_RATE_LOW = 60  # Adult range (adjust for infants)
HEART_RATE_HIGH = 100
```

## 🎯 Use Cases

### Infant Monitoring
- Track breathing and heart rate during sleep
- Detect irregular patterns
- Peace of mind for parents

### Research & Education
- Study respiratory and cardiac physiology
- Demonstrate rPPG technology
- Computer vision education

### Elderly Care
- Non-invasive vital signs monitoring
- Remote health tracking
- Early warning system

## 🛡️ Safety & Limitations

### ✅ Works Best When:
- Infant/subject's face is visible
- Good lighting (not too dark/bright)
- Minimal camera movement
- Subject is relatively still

### ⚠️ Limitations:
- Requires visible facial skin (nose area for HR)
- Accuracy decreases in very dark environments
- Movement can temporarily affect readings
- Not a substitute for medical equipment

### ❌ NOT Suitable For:
- **Medical diagnosis**
- **Critical care monitoring**
- **Emergency situations**
- **Replacing FDA-approved medical devices**

**Medical Disclaimer**: This is a monitoring and educational tool ONLY. Always consult healthcare professionals for medical advice and use certified medical devices for critical monitoring.

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Improving heart rate accuracy
- Multi-infant tracking
- Mobile app development
- Medical validation studies
- Raspberry Pi optimizations

## 📜 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **MediaPipe** - Google's pose detection framework
- **van der Kooij & Naber** - rPPG research foundation
- **OpenCV** - Computer vision library
- **SciPy** - Signal processing tools

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Nurtura-AI/breathing-monitor/issues)
- **Documentation**: See `/docs` folder
- **Research**: [rPPG Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6797647/)

## 🌟 Star Us!

If you find this project useful, please give it a ⭐ on GitHub!

---

**Made with ❤️ for safer infant monitoring**

*Last Updated: October 2025*
