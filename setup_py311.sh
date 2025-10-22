#!/bin/bash
# Setup script for Infant Breathing Monitor with Python 3.11
# This version explicitly uses Python 3.11 (required for MediaPipe)

set -e  # Exit on error

echo "=========================================="
echo "Infant Breathing Monitor - Setup"
echo "Python 3.11 Version"
echo "=========================================="
echo ""

# Check if Python 3.11 is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✓ Found Python 3.11"
elif command -v python3 &> /dev/null; then
    VERSION=$(python3 --version 2>&1 | grep -oE '3\.[0-9]+' | head -1)
    if [[ "$VERSION" == "3.11" ]]; then
        PYTHON_CMD="python3"
        echo "✓ Found Python 3.11"
    else
        echo "❌ Python 3.11 not found!"
        echo ""
        echo "You have Python $VERSION, but MediaPipe requires Python 3.11 or earlier."
        echo ""
        echo "Please install Python 3.11:"
        echo "  brew install python@3.11"
        echo ""
        echo "Then run this script again."
        echo ""
        echo "See PYTHON_VERSION_FIX.md for detailed instructions."
        exit 1
    fi
else
    echo "❌ Python not found!"
    echo "Please install Python 3.11 first."
    exit 1
fi

# Display Python version
echo "Using: $($PYTHON_CMD --version)"
echo ""

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "⚠️  Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$PYTHON_CMD -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Verify Python version in venv
VENV_VERSION=$(python --version 2>&1)
echo "Virtual environment Python: $VENV_VERSION"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import cv2; print('✓ OpenCV installed')"
python -c "import numpy; print('✓ NumPy installed')"
python -c "import mediapipe; print('✓ MediaPipe installed')"

# Create activation helper
cat > activate.sh << 'EOF'
#!/bin/bash
# Quick activation script
source venv/bin/activate
echo "✓ Virtual environment activated"
echo "Run: python breathing_monitor.py"
EOF
chmod +x activate.sh

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo "  # or simply:"
echo "  source activate.sh"
echo ""
echo "To test your camera:"
echo "  python test_camera.py"
echo ""
echo "To run the monitor:"
echo "  python breathing_monitor.py"
echo "  # or advanced version:"
echo "  python breathing_monitor_advanced.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""

