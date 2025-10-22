"""
Optimized Configuration for Raspberry Pi
Copy this to config.py or import settings from here

For Raspberry Pi 4 with 2GB+ RAM
"""

# ====================
# CAMERA SETTINGS
# ====================

CAMERA_INDEX = 0
CAMERA_WIDTH = 320      # Lower resolution for better performance
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# ====================
# POSE DETECTION
# ====================

MODEL_COMPLEXITY = 0    # Lightweight model (essential for Pi)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ====================
# BREATHING DETECTION
# ====================

WINDOW_SIZE = 30
BREATHING_THRESHOLD = 0.02
TORSO_PADDING = 20
TORSO_RESIZE = (64, 64)  # Smaller size for faster processing

# ====================
# OPTICAL FLOW PARAMETERS
# ====================

FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 2         # Reduced for performance
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

# ====================
# ALERT SETTINGS
# ====================

MIN_NORMAL_BREATHING_RATE = 20
MAX_NORMAL_BREATHING_RATE = 60
ENABLE_ALERTS = True    # Enable for infant monitoring

# ====================
# DISPLAY SETTINGS
# ====================

WINDOW_NAME = "Infant Breathing Monitor"
FLIP_HORIZONTAL = True
SHOW_TORSO_BOX = True
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
TEXT_FONT = 0
TEXT_SCALE = 0.6        # Slightly smaller for lower resolution
TEXT_THICKNESS = 2
SHOW_FPS = True

# ====================
# LOGGING SETTINGS
# ====================

ENABLE_LOGGING = True   # Recommended for monitoring
LOG_FILE = "breathing_data.csv"
LOG_INTERVAL = 2.0      # Log every 2 seconds

# ====================
# RASPBERRY PI OPTIMIZATIONS
# ====================

IS_RASPBERRY_PI = True

# ====================
# VALIDATION
# ====================

def validate_config():
    """Validate configuration parameters"""
    assert 0 <= MODEL_COMPLEXITY <= 2, "MODEL_COMPLEXITY must be 0, 1, or 2"
    assert 0.0 <= MIN_DETECTION_CONFIDENCE <= 1.0, "MIN_DETECTION_CONFIDENCE must be between 0.0 and 1.0"
    assert 0.0 <= MIN_TRACKING_CONFIDENCE <= 1.0, "MIN_TRACKING_CONFIDENCE must be between 0.0 and 1.0"
    assert WINDOW_SIZE > 0, "WINDOW_SIZE must be positive"
    assert BREATHING_THRESHOLD > 0, "BREATHING_THRESHOLD must be positive"
    assert TORSO_PADDING >= 0, "TORSO_PADDING must be non-negative"
    print("✓ Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print("\nRaspberry Pi Optimized Configuration:")
    print(f"  Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
    print(f"  Model Complexity: {MODEL_COMPLEXITY}")
    print(f"  Torso Resize: {TORSO_RESIZE}")
    print(f"  Optical Flow Levels: {FLOW_LEVELS}")
    print(f"  Logging: {'Enabled' if ENABLE_LOGGING else 'Disabled'}")
    print(f"  Alerts: {'Enabled' if ENABLE_ALERTS else 'Disabled'}")

