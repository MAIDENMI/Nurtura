#!/bin/bash
# Script to help download a test baby video using yt-dlp

echo "=============================================="
echo "Download Test Video for Breathing Monitor"
echo "=============================================="
echo ""

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp is not installed. Installing..."
    echo ""
    
    # Install yt-dlp
    if command -v brew &> /dev/null; then
        echo "Using Homebrew to install yt-dlp..."
        brew install yt-dlp
    else
        echo "Please install yt-dlp manually:"
        echo "  brew install yt-dlp"
        echo "  # or"
        echo "  pip install yt-dlp"
        exit 1
    fi
fi

echo "Suggested search terms for baby breathing videos:"
echo "  - 'sleeping baby breathing'"
echo "  - 'newborn baby sleeping'"
echo "  - 'infant chest breathing'"
echo ""
echo "Important: Only use videos you have permission to use!"
echo ""
echo "To download a video:"
echo "  yt-dlp -f 'best[height<=480]' -o baby_video.mp4 'VIDEO_URL'"
echo ""
echo "Then test it:"
echo "  python breathing_monitor_video.py baby_video.mp4"
echo ""

