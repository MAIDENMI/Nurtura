"""
Configuration file for Infant Breathing Monitor
Adjust these parameters to tune the system for your specific setup
"""

# ====================
# CAMERA SETTINGS
# ====================

# Camera index (0 for default camera, 1+ for additional cameras)
CAMERA_INDEX = 0

# Camera resolution (lower = faster on Raspberry Pi)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Target FPS (actual FPS may be lower on Raspberry Pi)
CAMERA_FPS = 30


# ====================
# POSE DETECTION
# ====================

# Model complexity: 0 (lightweight), 1 (balanced), 2 (accurate)
# Use 0 for Raspberry Pi, 1-2 for more powerful systems
MODEL_COMPLEXITY = 0

# Minimum confidence for pose detection (0.0 to 1.0)
# Lower = easier detection but more false positives
MIN_DETECTION_CONFIDENCE = 0.5

# Minimum confidence for tracking (0.0 to 1.0)
MIN_TRACKING_CONFIDENCE = 0.5


# ====================
# BREATHING DETECTION
# ====================

# Number of frames to use for moving average window
# Higher = smoother readings but slower response
# Recommended: 20-60 frames
WINDOW_SIZE = 30

# Threshold for motion detection (sensitivity)
# Lower = more sensitive to small movements
# Typical range: 0.01 to 0.05
# Adjust based on camera distance and lighting
BREATHING_THRESHOLD = 0.02

# Torso padding in pixels (added around detected torso region)
TORSO_PADDING = 20

# Size for torso normalization (width, height)
# Smaller = faster processing
TORSO_RESIZE = (100, 100)


# ====================
# OPTICAL FLOW PARAMETERS
# ====================

# Farneback optical flow parameters
# These generally work well, but can be adjusted for specific scenarios
FLOW_PYR_SCALE = 0.5      # Image pyramid scale
FLOW_LEVELS = 3            # Number of pyramid levels
FLOW_WINSIZE = 15          # Averaging window size
FLOW_ITERATIONS = 3        # Iterations at each pyramid level
FLOW_POLY_N = 5           # Size of pixel neighborhood
FLOW_POLY_SIGMA = 1.2     # Standard deviation for Gaussian


# ====================
# ALERT SETTINGS
# ====================

# Alert thresholds for breathing rate (breaths per minute)
# These are typical ranges for infants
MIN_NORMAL_BREATHING_RATE = 20   # Below this = potential issue
MAX_NORMAL_BREATHING_RATE = 60   # Above this = potential issue

# Enable/disable alerts
ENABLE_ALERTS = False


# ====================
# DISPLAY SETTINGS
# ====================

# Display window name
WINDOW_NAME = "Infant Breathing Monitor"

# Flip camera horizontally (for selfie-view)
FLIP_HORIZONTAL = True

# Display torso bounding box
SHOW_TORSO_BOX = True

# Bounding box color (BGR format)
BOX_COLOR = (0, 255, 0)  # Green

# Bounding box thickness
BOX_THICKNESS = 2

# Text settings
TEXT_COLOR = (0, 255, 0)    # Green
TEXT_FONT = 0               # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# Display FPS counter
SHOW_FPS = True


# ====================
# LOGGING SETTINGS
# ====================

# Enable data logging to CSV
ENABLE_LOGGING = False

# Log file path
LOG_FILE = "breathing_data.csv"

# Log interval in seconds
LOG_INTERVAL = 1.0


# ====================
# RASPBERRY PI OPTIMIZATIONS
# ====================

# Set to True if running on Raspberry Pi
IS_RASPBERRY_PI = False

# Additional performance settings for Raspberry Pi
if IS_RASPBERRY_PI:
    # Use lower resolution
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    
    # Reduce processing size
    TORSO_RESIZE = (64, 64)
    
    # Use fewer optical flow levels
    FLOW_LEVELS = 2


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
    print("\nCurrent Configuration:")
    print(f"  Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS")
    print(f"  Model Complexity: {MODEL_COMPLEXITY}")
    print(f"  Window Size: {WINDOW_SIZE} frames")
    print(f"  Breathing Threshold: {BREATHING_THRESHOLD}")
    print(f"  Raspberry Pi Mode: {IS_RASPBERRY_PI}")

