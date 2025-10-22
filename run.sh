#!/bin/bash
# Quick run script - automatically activates venv and runs the monitor

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first: ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Parse arguments
if [ "$1" == "enhanced" ]; then
    echo "Starting Enhanced Breathing Monitor (AIRFlowNet-inspired)..."
    python breathing_monitor_enhanced.py
elif [ "$1" == "advanced" ]; then
    echo "Starting Advanced Breathing Monitor..."
    python breathing_monitor_advanced.py
elif [ "$1" == "graph" ] || [ "$1" == "graphical" ]; then
    echo "Starting Graphical Breathing Monitor with Real-Time Graphs..."
    python breathing_monitor_graphical.py
elif [ "$1" == "video" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide a video file"
        echo "Usage: ./run.sh video <video_file>"
        echo "Example: ./run.sh video baby_video.mp4"
    else
        echo "Analyzing video: $2"
        python breathing_monitor_video.py "$2"
    fi
elif [ "$1" == "test" ]; then
    echo "Starting Camera Test..."
    python test_camera.py
elif [ "$1" == "config" ]; then
    echo "Validating Configuration..."
    python config.py
else
    echo "Starting Basic Breathing Monitor..."
    echo ""
    echo "Available versions:"
    echo "  ./run.sh                    - Basic version"
    echo "  ./run.sh enhanced           - ⭐ Enhanced (AIRFlowNet-inspired)"
    echo "  ./run.sh graph              - With real-time graphs"
    echo "  ./run.sh video <file>       - Test with video file"
    echo "  ./run.sh advanced           - Advanced features"
    echo "  ./run.sh test               - Test camera"
    echo ""
    python breathing_monitor.py
fi

