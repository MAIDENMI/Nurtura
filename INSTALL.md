# Installation Guide

Complete installation instructions for all platforms.

## Why Virtual Environment?

**Always use a virtual environment!** This prevents:
- 🔴 Package version conflicts
- 🔴 System-wide installation issues
- 🔴 Breaking other Python projects
- 🔴 Permission problems

## Automated Installation (Recommended)

### macOS / Linux

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

### Windows

```cmd
setup.bat
```

### What the Setup Script Does

1. Creates a fresh virtual environment in `venv/`
2. Activates the virtual environment
3. Upgrades pip to latest version
4. Installs all dependencies from `requirements.txt`
5. Verifies all packages installed correctly
6. Creates helper scripts for easy activation

---

## Manual Installation

If you prefer to install manually or need more control:

### Step 1: Create Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv

# Windows
python -m venv venv
```

### Step 2: Activate Virtual Environment

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import cv2; import numpy; import mediapipe; print('Success!')"
```

---

## Raspberry Pi Installation

### Prerequisites

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y libatlas-base-dev libhdf5-dev libopencv-dev

# For Raspberry Pi Camera Module
sudo apt-get install -y python3-picamera
```

### Install Project

```bash
# Navigate to project directory
cd /home/pi/Testch

# Run setup
chmod +x setup.sh
./setup.sh

# Copy Raspberry Pi optimized config
cp config_raspberry_pi.py config.py
```

### Enable Camera

```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
```

---

## Troubleshooting

### "Permission Denied" on Linux/Mac

```bash
chmod +x setup.sh run.sh
```

### "Python not found" on Windows

Install Python from [python.org](https://www.python.org/downloads/)
- ✓ Check "Add Python to PATH" during installation

### OpenCV Installation Fails

```bash
# Try installing system OpenCV first
# Ubuntu/Debian:
sudo apt-get install python3-opencv

# macOS:
brew install opencv

# Then reinstall in venv:
pip install opencv-python==4.8.1.78
```

### MediaPipe Installation Issues

MediaPipe requires:
- Python 3.7-3.11 (not 3.12+)
- 64-bit Python
- Compatible OS (Windows 10+, macOS 10.14+, Ubuntu 18.04+)

```bash
# Check Python version
python --version

# Check if 64-bit
python -c "import struct; print(struct.calcsize('P') * 8)"
# Should output: 64
```

### Low Disk Space on Raspberry Pi

```bash
# Check available space
df -h

# If low, expand filesystem
sudo raspi-config
# Navigate to: Advanced Options -> Expand Filesystem
```

### Import Errors

```bash
# Make sure virtual environment is activated
which python
# Should show: /path/to/Testch/venv/bin/python

# If not, activate it:
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

---

## Verifying Your Installation

### Check All Dependencies

```bash
source venv/bin/activate
pip list
```

You should see:
- opencv-python (4.8.1.78)
- numpy (1.24.3)
- mediapipe (0.10.8)

### Test Camera

```bash
./run.sh test      # macOS/Linux
run.bat test       # Windows
```

### Validate Configuration

```bash
python config.py
```

---

## Updating Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Update all packages
pip install --upgrade -r requirements.txt
```

---

## Uninstalling

```bash
# Simply delete the virtual environment
rm -rf venv

# Or remove entire project
cd ..
rm -rf Testch
```

Your system Python remains untouched! ✨

---

## Next Steps

Once installation is complete:

1. **Test Camera**: `./run.sh test` or `python test_camera.py`
2. **Run Monitor**: `./run.sh` or `python breathing_monitor.py`
3. **Read Quick Start**: See `QUICKSTART.md`
4. **Configure**: Edit `config.py` for your setup

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review `README.md` for detailed documentation
3. Ensure virtual environment is activated
4. Verify Python version (3.7-3.11)
5. Check system requirements

**Remember**: Always activate your virtual environment before running the project!
```bash
source venv/bin/activate  # You should see (venv) in your prompt
```

