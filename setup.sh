#!/bin/bash
# Setup script for Infant Breathing Monitor
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Infant Breathing Monitor - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists."
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    else
        echo "Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import cv2; import numpy; import mediapipe; print('✓ All packages installed successfully')"

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

