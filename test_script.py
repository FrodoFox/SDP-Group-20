import time
import sys
from GolfBallDetection import GolfBallTracker 

def main():
    # 1. Initialize the tracker
    tracker = GolfBallTracker(show_window=True)
    
    # 2. Start the camera and tracking threads
    print("Initializing Camera and Threads...")
    tracker.start()
    
    # Give the camera a moment to initialize
    time.sleep(1)
    
    print("\n--- Tracking Active (New Line Output) ---")
    print("Press Ctrl+C to stop the test.")
    
    try:
        while tracker.running:
            # 3. Pull the latest coordinates
            location = tracker.get_latest_location()
            
            if location:
                x, y, z = location
                print(f"Ball Found | X: {x:>7.2f}mm | Y: {y:>7.2f}mm | Z: {z:>7.2f}mm")
            else:
                print(f"Searching... [{time.strftime('%H:%M:%S')}]")
            
            # Poll every 0.5 seconds as requested
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nManual stop detected.")
    finally:
        print("Cleaning up camera and windows...")
        tracker.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()