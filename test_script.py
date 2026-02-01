import time
import sys
from GolfBallDetection import GolfBallTracker 

def main():

    # Instantiating the golfball tracker
    tracker = GolfBallTracker(show_window=True)
    
    print("Initializing Camera and Threads...")
    tracker.start()
    
    # Give the camera a moment to initialize
    time.sleep(1)
    
    print("\n--- Tracking Active (New Line Output) ---")
    print("Press Ctrl+C to stop the test (or q on the window).")
    
    try:
        while tracker.running:

            # Polling the current location of the ball being tracked
            location = tracker.get_latest_location()
            
            # If a ball is being tracked then display its coordinates or say searching
            if location:
                x, y, z = location
                output = f"Ball Found | X: {x:>7.2f}mm | Y: {y:>7.2f}mm | Z: {z:>7.2f}mm"
            else:
                output = "Searching...                                                  "   # A lot of whitespace to fully clear prior text
            
            # Use sys.stdout to bypass print's default buffering
            sys.stdout.write('\r' + output)
            sys.stdout.flush()

            # Polling every 0.5s to not flood the terminal
            time.sleep(0.5)

    finally:
        print("\nCleaning up camera and windows...")
        tracker.stop()
        sys.exit(0)

# DO STUFF ON START
if __name__ == "__main__":
    main()