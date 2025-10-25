# Python Version Compatibility Issue

## Problem

You have **Python 3.13**, but **MediaPipe currently only supports Python 3.7-3.11**.

MediaPipe is essential for the pose detection functionality and hasn't been updated for Python 3.13 yet.

## Solution: Install Python 3.11

### Option 1: Using Homebrew (Recommended for macOS)

```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
# Should show: Python 3.11.x

# Remove old venv
rm -rf venv

# Create venv with Python 3.11
python3.11 -m venv venv

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using pyenv (Cross-platform)

```bash
# Install pyenv
brew install pyenv

# Install Python 3.11
pyenv install 3.11.7

# Set local Python version for this project
cd /Users/aidenm/Testch
pyenv local 3.11.7

# Remove old venv and recreate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 3: Download from python.org

1. Visit: https://www.python.org/downloads/
2. Download Python 3.11.x (latest 3.11 version)
3. Install it
4. Create venv with: `/usr/local/bin/python3.11 -m venv venv`

## Quick Setup After Installing Python 3.11

```bash
# Remove old virtual environment
rm -rf venv

# Create new venv with Python 3.11
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test it
python test_camera.py
```

## Updated Setup Script

I've created a new setup script that uses Python 3.11 automatically:

```bash
./setup_py311.sh
```

## Why Python 3.13 Doesn't Work

MediaPipe uses compiled C++ code and pre-built wheels. These need to be built for each Python version. Python 3.13 is very new (released October 2024), and many packages haven't caught up yet.

## Check Python Version

```bash
# Check all installed Python versions
ls /Library/Frameworks/Python.framework/Versions/
ls /opt/homebrew/Cellar/python*

# Check specific version
python3.11 --version
```

## Future: When Will MediaPipe Support 3.13?

MediaPipe typically updates a few months after new Python releases. Check their GitHub for updates:
- https://github.com/google/mediapipe/issues

You can track Python 3.13 support here:
- https://pypi.org/project/mediapipe/#history

## Alternative: Use Docker (Advanced)

If you don't want to install another Python version, you can use Docker:

```bash
# Create a Dockerfile with Python 3.11
# (See DOCKER.md for instructions)
docker build -t breathing-monitor .
docker run -it --device=/dev/video0 breathing-monitor
```

## Need Help?

If you're stuck, the easiest solution is:
```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python breathing_monitor.py
```

