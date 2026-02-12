import time
import sys
from vision_code.GolfBallDetection_cv2 import GolfBallTracker

def main():

    # Start the golf ball tracker
    tracker = GolfBallTracker(show_window=True)
    tracker.start()
    
    time.sleep(1)
    print("\n--- Tracking Active ---")
    print("Press Ctrl+C to stop (or 'q' in the window).")

    try:
        while tracker.running:
            
            # Getting most recent ball location
            location = tracker.get_latest_location()
            
            # If a location was found then display the coordinates it was found at relative to the camera
            if location:
                x, y, z = location
                output = f"Ball Found | X: {x:>7.2f}mm | Y: {y:>7.2f}mm | Z: {z:>7.2f}mm"

            # If no ball was detected on last sample or being tracked in record then output that it's searching
            else:
                output = "Searching...                                                  "
            
            # Syscall to output properly in terminal
            sys.stdout.write('\r' + output)
            sys.stdout.flush()
            time.sleep(0.5)

    # Clean up on quit
    finally:
        print("\nCleaning up camera and windows...")
        tracker.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
