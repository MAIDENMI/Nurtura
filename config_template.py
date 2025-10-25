"""
Configuration file for Infant Breathing Monitor
Copy this to config.py and fill in your credentials
"""

# ====================
# CAMERA SETTINGS
# ====================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ====================
# POSE DETECTION
# ====================

MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ====================
# BREATHING DETECTION
# ====================

WINDOW_SIZE = 30
BREATHING_THRESHOLD = 0.02
TORSO_PADDING = 20
TORSO_RESIZE = (100, 100)

# ====================
# OPTICAL FLOW PARAMETERS
# ====================

FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

# ====================
# ALERT SETTINGS
# ====================

MIN_NORMAL_BREATHING_RATE = 20
MAX_NORMAL_BREATHING_RATE = 60
ENABLE_ALERTS = False

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
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
SHOW_FPS = True

# ====================
# LOGGING SETTINGS
# ====================

ENABLE_LOGGING = False
LOG_FILE = "breathing_data.csv"
LOG_INTERVAL = 1.0

# ====================
# RASPBERRY PI OPTIMIZATIONS
# ====================

IS_RASPBERRY_PI = False

if IS_RASPBERRY_PI:
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    TORSO_RESIZE = (64, 64)
    FLOW_LEVELS = 2

# ====================
# SNOWFLAKE DATABASE SETTINGS
# ====================

# Snowflake connection parameters
# IMPORTANT: Use environment variables for security in production
SNOWFLAKE_ACCOUNT = ""  # e.g., "xy12345.us-east-1.aws"
SNOWFLAKE_USER = ""
SNOWFLAKE_PASSWORD = ""
SNOWFLAKE_DATABASE = ""  # e.g., "HEALTH_MONITORING"
SNOWFLAKE_SCHEMA = "PUBLIC"
SNOWFLAKE_WAREHOUSE = ""  # e.g., "COMPUTE_WH"

# Table name for vital signs data
SNOWFLAKE_TABLE = "VITAL_SIGNS_DATA"

# Enable/disable database logging
ENABLE_SNOWFLAKE_LOGGING = False

# Interval for saving data to Snowflake (in seconds)
SNOWFLAKE_LOG_INTERVAL = 10.0

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

