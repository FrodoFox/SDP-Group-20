import time
import sys
import socket
import json
from GolfBallDetection import GolfBallTracker

# ----- UDP SETUP -----
UDP_IP = "weepinbell"  # Ubuntu machine IP
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def main():
    # Start the golf ball tracker
    tracker = GolfBallTracker(show_window=True)
    print("Initializing Camera and Threads...")
    tracker.start()
    
    time.sleep(1)
    print("\n--- Tracking Active ---")
    print("Press Ctrl+C to stop (or 'q' in the window).")

    try:
        while tracker.running:
            location = tracker.get_latest_location()
            if location:
                x, y, z = location
                output = f"Ball Found | X: {x:>7.2f}mm | Y: {y:>7.2f}mm | Z: {z:>7.2f}mm"
                
                # ----- SEND DATA TO UBUNTU -----
                message = json.dumps({'x': x, 'y': y, 'z': z}).encode('utf-8')
                sock.sendto(message, (UDP_IP, UDP_PORT))
            else:
                output = "Searching...                                                  "
            
            sys.stdout.write('\r' + output)
            sys.stdout.flush()
            time.sleep(0.5)
    finally:
        print("\nCleaning up camera and windows...")
        tracker.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
