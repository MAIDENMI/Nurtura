# 👶 Nurtura - AI-Powered Infant Breathing Monitor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research-validated, contactless infant breathing rate monitoring using computer vision and AI.**

![Nurtura Banner](https://img.shields.io/badge/Status-Research%20Project-orange)

---

## 🌟 What is Nurtura?

Nurtura is an **AI-powered breathing monitor** that uses your camera to measure infant breathing rates in real-time. No wearables, no contact required—just a camera and computer vision.

Built using **research-validated methods** from published respiratory rate monitoring papers, Nurtura combines:
- 🎯 **Multi-point body tracking** (chest, abdomen, nose)
- 🎨 **BGR signal analysis** across all color channels
- 📊 **Bandpass filtering** for accurate respiratory frequencies
- 💯 **Confidence scoring** to show measurement reliability

---

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MAIDENMI/Nurtura.git
cd Nurtura

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Run the Monitor

**Real-time Camera Monitoring:**
```bash
python breathing_monitor_research.py
```

**Analyze Pre-recorded Videos:**
```bash
python analyze_infant_video.py your_video.mp4
```

---

## 🎯 Key Features

### Real-Time Monitoring (`breathing_monitor_research.py`)
- ✅ **Live breathing rate calculation** in breaths per minute (BPM)
- ✅ **Multi-region tracking**: Chest (torso), abdomen, nose, and control point
- ✅ **Dynamic tracking** that follows movement
- ✅ **Real-time confidence scoring** (0-100%)
- ✅ **Live signal quality graphs**
- ✅ **Color-coded status** (Normal / Low / High)

### Video Analysis (`analyze_infant_video.py`)
- 📹 **Analyze recorded infant videos**
- 📊 **Generate detailed analysis charts**
- 🎯 **Breathing rate with confidence metrics**
- 💾 **Save results as PNG images**

---

## 🔬 How It Works

Nurtura uses a sophisticated multi-step process:

1. **Body Detection**: MediaPipe AI detects infant body landmarks
2. **Multi-Point Tracking**: Tracks 4 key regions (chest, abdomen, nose, control)
3. **BGR Signal Extraction**: Analyzes Blue, Green, Red color channels at each point
4. **Signal Filtering**: Applies Butterworth bandpass filter (0.12-0.75 Hz)
5. **Peak Detection**: Uses scipy to find breathing cycles
6. **Rate Calculation**: Converts peaks to breaths per minute
7. **Confidence Scoring**: Validates measurement reliability

### Research Foundation

Based on validated methods from respiratory rate monitoring research:
- Multi-point BGR signal analysis
- Butterworth bandpass filtering for breathing frequencies
- Control point normalization to eliminate noise
- Dynamic peak detection with prominence thresholding

---

## 📊 Expected Breathing Rates

| Age Group | Normal Range (breaths/min) |
|-----------|---------------------------|
| **Newborn (0-3 months)** | 30-60 |
| **Infant (3-12 months)** | 24-40 |
| **Toddler (1-3 years)** | 20-30 |
| **Adult at rest** | 12-20 |

---

## 🎮 Controls

### Real-Time Monitor
- **'q'** - Quit the program
- **'s'** - Take screenshot

### Display Windows
1. **Camera Feed**: Shows live video with tracking points
2. **Analysis Graphs**: Real-time breathing rate and signal quality

---

## 📋 Requirements

- **Python**: 3.9+
- **Camera**: Any webcam or camera module
- **OS**: macOS, Linux, Windows, Raspberry Pi

### Python Dependencies
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 🏗️ Project Structure

```
Nurtura/
├── breathing_monitor_research.py   # Main real-time monitor ⭐
├── analyze_infant_video.py         # Video analysis tool ⭐
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Setup script
├── README.md                        # This file
├── archive/                         # Archived versions
│   ├── old_versions/               # Previous implementations
│   ├── test_videos/                # Test videos and results
│   └── docs/                       # Additional documentation
└── venv/                           # Virtual environment
```

---

## ⚠️ Important Disclaimer

**This is a research and educational project.**

❌ **NOT a medical device**  
❌ **NOT FDA approved**  
❌ **NOT for clinical diagnosis**  
❌ **Should NOT replace professional medical equipment**

Always consult healthcare professionals for infant health monitoring. This tool is for:
- ✅ Research purposes
- ✅ Educational demonstrations
- ✅ Technology exploration
- ✅ Algorithm development

---

## 🚀 Advanced Usage

### Custom Configuration

Edit `config.py` to adjust:
- Camera resolution
- Detection sensitivity
- Alert thresholds
- Filter parameters

### Raspberry Pi Deployment

Optimized for Raspberry Pi 4 (2GB+ RAM):
```bash
# Use lower resolution for better performance
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
```

---

## 🤝 Contributing

This project is open for contributions! Areas for improvement:
- Algorithm optimization
- Mobile app development
- Cloud integration
- Additional validation studies
- UI/UX enhancements

---

## 📚 Technical Details

### Signal Processing Pipeline
```
Video Frame → Pose Detection → Multi-Point Tracking →
BGR Signal Extraction → Bandpass Filter (0.12-0.75 Hz) →
Peak Detection → Rate Calculation → Confidence Scoring
```

### Key Technologies
- **MediaPipe**: Body pose detection
- **OpenCV**: Video processing
- **SciPy**: Signal processing (Butterworth filters, peak detection)
- **NumPy**: Numerical computations
- **Matplotlib**: Real-time visualization

---

## 📝 Citation

If you use Nurtura in your research, please cite:
```
Nurtura - AI-Powered Infant Breathing Monitor
https://github.com/MAIDENMI/Nurtura
```

---

## 📄 License

This project is open-source under the MIT License.

---

## 👨‍💻 Author

**MAIDENMI**
- GitHub: [@MAIDENMI](https://github.com/MAIDENMI)
- Project: [Nurtura](https://github.com/MAIDENMI/Nurtura)

---

## 🙏 Acknowledgments

Built with research-validated methods from respiratory monitoring literature.  
Special thanks to the open-source community for MediaPipe, OpenCV, and SciPy.

---

**⚡ Built with AI assistance | 🔬 Research-validated | 💙 Made for infant safety research**
