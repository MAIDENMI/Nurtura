"""
Simple script to test camera connectivity
Run this before running the breathing monitor to ensure your camera works
"""

import cv2
import sys

def test_camera(camera_index=0):
    """Test camera connection and display video feed"""
    print(f"Testing camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open camera {camera_index}")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Try different camera index: python test_camera.py 1")
        print("  3. On Linux, check permissions: ls -l /dev/video*")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✓ Camera {camera_index} opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print("\nPress 'q' to quit\n")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error reading frame")
                break
            
            frame_count += 1
            
            # Add text overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"✓ Test complete. Captured {frame_count} frames.")
        return True


def list_available_cameras():
    """Try to detect available cameras"""
    print("Scanning for available cameras...")
    available = []
    
    for i in range(5):  # Check first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    if available:
        print(f"✓ Found {len(available)} camera(s): {available}")
    else:
        print("✗ No cameras found")
    
    return available


if __name__ == "__main__":
    print("=" * 60)
    print("Camera Test Utility")
    print("=" * 60)
    print()
    
    # List available cameras
    cameras = list_available_cameras()
    print()
    
    # Test specific camera
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"✗ Invalid camera index: {sys.argv[1]}")
            print("Usage: python test_camera.py [camera_index]")
            sys.exit(1)
    
    test_camera(camera_index)

