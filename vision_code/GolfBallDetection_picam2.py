import sys
import os
import cv2
import numpy as np
import math
from picamera2 import Picamera2
import threading
import time
import json
from collections import deque
from ultralytics import YOLO
import xmlrpc.client
import socket

# --- CONFIGURATION ---
# Camera Config
BRIGHTNESS       = -0.25        # Offsets the black level of the image
ANALOGUE_GAIN    = 0            # Apply a gain to each pixels value (maintain under ~2 to avoid noise)
GAMMA            = 0.75         # Gamma > 1.0 brightens shadows, < 1.0 darkens them.
EXPOSURE_TIME    = 12000        # Exposure Time (camera shutter open in ms)
BRIGHTNESS_CHECK = 0            # Final brightness threshold of white for the ball (zero works but not brilliantly)

# YOLO & Tracking Config
YOLO_MODEL       = "yolov8n.pt" # YOLO model
SPORTS_BALL_ID   = 32           # (COCO) Class ID for sports balls (not just golf balls)
YOLO_CONF        = 0.05         # Confidence threshold (Raise if false positives, lower if misses detections)
ROI_Y_FRACTION   = 0.35         # Only search bottom 55% of image
MAX_MISSED       = 5            # Frame limit for maximum allowed missed frames
SMOOTH           = 0.5          # Smoothing factor for coordinates

# JSON Calibration Config
CALIB_FILE       = "camera_calib.json"
DEFAULT_FOCAL    = 1675.0

# Connection Config
#socket.setdefaulttimeout(0.05)
TURTLE_IP = "fe80::2ecf:67ff:fe0c:f9e5%eth0"
RECEIVER_URL = f"http://[{TURTLE_IP}]:8000/"

# Stopping Yolo From Trying to Update over Ethernet
os.environ['YOLO_OFFLINE'] = 'True'

class GolfBallTracker:

    # --- INITIALISATION FUNCTION ---
    def __init__(self, show_window=False):

        # Using print statements to show what's initialising when for debugging and error detection
        print(f"Tracker: Starting")

        # Killing any prior instances to ensure the camera isn't overrun
        os.system("pkill -o -9 python3 > /dev/null 2>&1")

        # Values to allow depth perception and calculation of distance to golf ball
        self.BALL_DIAM_MM = 42.67

        # Initialise Remote Connection to Weepinbell
        self.remote_pi = xmlrpc.client.ServerProxy(RECEIVER_URL, allow_none=True)
        print(f"Tracker (Network): Connecting to {RECEIVER_URL}")

        # --- JSON LOGIC: Load Focal Length from file ---
        self.FOCAL_LENGTH = self.load_calibration()

        # Scalings of resolution for display and processing
        self.DISPLAY_SCALE      = 0.5
        self.WIDTH, self.HEIGHT = 1280, 720
        self.show_window        = show_window
        
        # Computes the center coordinate of the FULL frame
        self.cx = self.WIDTH // 2
        self.cy = self.HEIGHT // 2

        # Initialisation of camera
        print(f"Tracker (Camera): Initialising Instance")
        self.picam2 = Picamera2()
        
        # Configuring resolution
        print(f"Tracker (Camera): Configuring Resolution")
        camera_config = self.picam2.create_video_configuration(main={"size": (self.WIDTH, self.HEIGHT), "format": "BGR888"})
        self.picam2.configure(camera_config)

        # Setting camera controls
        print(f"Tracker (Camera): Configuring Settings Using Local Parameters")
        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": EXPOSURE_TIME,
            "AnalogueGain": ANALOGUE_GAIN,
            "Brightness": BRIGHTNESS
        })

        # Initialise YOLO Model
        print(f"Tracker (YOLO): Loading Model - {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL)
        
        # Using a deque to ensure processing of most recent frame
        self.frame_queue   = deque(maxlen=1)
        self.display_queue = deque(maxlen=1)
        self.running       = True
        self.current_gain  = 1.2
        
        # External access variables
        self.location_lock = threading.Lock()
        self.latest_xyz    = None

        # Initialise gamma lookup table
        print(f"Tracker (Camera): Building Gamma Table")
        self.gamma_lut = self.build_gamma_lut(GAMMA)

    # --- Load JSON Calibration ---
    def load_calibration(self):
        if os.path.exists(CALIB_FILE):
            try:
                with open(CALIB_FILE, 'r') as f:
                    data = json.load(f)
                    print(f"Tracker (JSON): Loaded calibration from {CALIB_FILE}")
                    return data.get("focal_length", DEFAULT_FOCAL)
            except Exception as e:
                print(f"Tracker (JSON): Error loading file, using default. {e}")
        return DEFAULT_FOCAL

    # --- Save JSON Calibration ---
    def save_calibration(self, new_focal):
        try:
            with open(CALIB_FILE, 'w') as f:
                json.dump({"focal_length": new_focal}, f)
            print(f"Tracker (JSON): Saved new focal length {new_focal:.1f} to {CALIB_FILE}")
            self.FOCAL_LENGTH = new_focal
        except Exception as e:
            print(f"Tracker (JSON): Failed to save calibration. {e}")

    # --- FUNCTION TO CONSTRUCT A GAMMA TABLE BASED ON ANY PARAMETERS IT WAS HANDED BEFORE RUNTIME ---
    def build_gamma_lut(self, gamma):

        # Creating a lookup table to map specific gamma curve values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return table

    # --- FUNCTION TO APPLY GAMMA TABLE TO A FRAME IT'S PASSED
    def apply_gamma(self, image):

        # Applies computed gamma table to the frame
        return cv2.LUT(image, self.gamma_lut)

    # --- FUNCTION TO CAPTURE A SINGLE FRAME FROM CAMERA (MAIN THREAD ONLY) ---
    def grab_frame(self):

        # Captures the frame as a BGR numpy array
        bgr_frame = self.picam2.capture_array()

        # Applying gamma correction directly to the captured BGR frame
        processed_frame = self.apply_gamma(bgr_frame)

        # Checks brightness and adjusts analogue gain
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        avg = np.mean(gray)

        # Altering the analogue gain if the images average is too dark
        new_gain = 1.7 if avg < 70 else 1.2
        if new_gain != self.current_gain:
            self.picam2.set_controls({"AnalogueGain": new_gain})
            self.current_gain = new_gain

        # Puts the new frame onto the queue
        self.frame_queue.append(processed_frame)


    # --- START FUNCTION TO START THREADS AND CATCH AN ERROR IF THE CAMERA FAILS ---
    def start(self):

        # Start Camera
        try:
            self.picam2.start()
            print("Tracker (Camera): Picamera2 started successfully")
        except Exception as e:
            print(f"Tracker (Camera): Failed to start camera: {e}")
            return

        # Starting background threads for tracking and processing
        print(f"Threading: Starting Threads")
        print(f"Threading: Starting Ball Tracker")
        threading.Thread(target=self.tracking_loop, daemon=True).start()

        print("Golf Ball Tracker Service Started.")


    # --- STOP FUNCTION TO RELEASE HARDWARE AND REMOVE CV2 CREATED WINDOWS
    def stop(self):

        # Release hardware at the end
        self.running = False
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Tracker: Free'd Hardware and Stopped")

    # --- FUNCTION TO SIMPLY RETURN A TUPLE OF THE MOST RECENTLY TRACKED BALL ---
    def get_latest_location(self):

        # Returns latest known location of any tracked ball as (x, y, z) tuple
        # Returns coordinates relative to the middle of the frame (0,0) as the midpoint rather than the top left
        with self.location_lock:
            return self.latest_xyz

    # --- FUNCTION TO COMPUTE RELATIVE OFFSET FROM THE MIDDLE OF THE CAMERA FROM THE ACTUAL COORDIANTES ---
    def compute_xyz(self, x1, y1, x2, y2):

        # Computes the box size (around the ball)
        width_pix   = max(1.0, (x2 - x1))

        # Z = (focal_length * real_diameter) / pixel_width
        z = (self.FOCAL_LENGTH * self.BALL_DIAM_MM) / width_pix
        
        # X and Y offsets from center of camera
        x_off = (x1 + (width_pix * 0.5) - self.cx) * z / self.FOCAL_LENGTH
        y_off = (y1 + ((y2 - y1) * 0.5) - self.cy) * z / self.FOCAL_LENGTH
        
        return x_off, y_off, z

    # --- TRANSMISSION TO OTHER PI FUNCTION ---
    def transmit_coords(self, coords):
        
        if coords is None:
            return

        # --- HELPER FUNCTION (THAT DOES IT ON A THREAD TO NOT BREAK MAIN LOOP) ---
        def send():
            try:
                x, y, z = coords

                # Sends the coords "wrapped" with the send_coords so that it knows what flag they hit
                self.remote_pi.send_coords(float(x), float(y), float(z))
            except Exception as e:
                # Catching errors to avoid crashes if the reciever is off
                pass

        # Fire the thread and immediately return control to YOLO
        threading.Thread(target=send, daemon=True).start()

    # --- LOOP TO ACTUALLY DETECT AND TRACK COORDINATES OF GOLF BALLS ---
    def tracking_loop(self):

        tracked_box = None  # Stores (x1, y1, x2, y2) of last ball
        missed = 0

        # --- FPS Counter Initialisation ---
        prev_time = time.time()
        fps = 0

        while self.running:
            if not self.frame_queue:
                time.sleep(0.001)
                continue
            
            # Pop the latest frame and get the centers for the FULL frame
            full_frame = self.frame_queue.pop()
            h          = self.HEIGHT                # A single call to retrieve the value from the class
            w          = self.WIDTH                 # Then can rely on the quicker stack for fetching from there

            # --- ROI LOGIC ---
            # Crops the height of the ROI to be the % declared in config
            roi_y_start = int(h * ROI_Y_FRACTION)
            
            # Dynamic Cropping - if there was a ball being tracked, then crop the frame around there and look there
            if tracked_box and missed == 0:
                tx1, ty1, tx2, ty2 = tracked_box
                
                # Add padding
                pad = int((tx2 - tx1) * 2.0) 
                
                # Dynamic box limits (clamped to frame size)
                d_x0 = max(0, int(tx1 - pad))
                d_y0 = max(roi_y_start, int(ty1 - pad)) # Ensure we don't go above the floor horizon
                d_x1 = min(w, int(tx2 + pad))
                d_y1 = min(h, int(ty2 + pad))
                
                # Check the box exists within the limits of the frame (and not croped by edges of camera)
                if (d_x1 - d_x0) > 10 and (d_y1 - d_y0) > 10:
                    # Use the cropped frame around the prior ROI
                    proc_frame         = full_frame[d_y0:d_y1, d_x0:d_x1]
                    offset_x, offset_y = d_x0, d_y0
                else:
                    # Fallback to base crop
                    proc_frame         = full_frame[roi_y_start:h, 0:w]
                    offset_x, offset_y = 0, roi_y_start
            else:
                # No tracking so just uses the bottom % of the screen
                proc_frame             = full_frame[roi_y_start:h, 0:w]
                offset_x, offset_y     = 0, roi_y_start

            # --- YOLO INFERENCE ---
            # This line of code returns boxes of the detections on where a golf ball would be
            detections = self.model.predict(proc_frame, conf=YOLO_CONF, verbose=False)[0]

            best_box    = None
            best_area   = 0

            # Iterate through detections to find the largest "Sports Ball"
            if detections.boxes is not None:
                for box in detections.boxes:

                    # Ignores everything else the AI sees and focusses on sports balls
                    if int(box.cls[0]) == SPORTS_BALL_ID:
                        bx1, by1, bx2, by2 = map(float, box.xyxy[0])
                        area = (bx2 - bx1) * (by2 - by1)

                        # Finding the ball closest to the camera by filtering for the biggest one
                        if area > best_area:
                            best_area = area
                            best_box = (bx1, by1, bx2, by2)

            # --- Calculate FPS ---
                curr_time = time.time()
                elapsed   = curr_time - prev_time
                prev_time = curr_time
                fps       = (fps * 0.9) + ((1.0 / max(0.001, elapsed)) * 0.1)   # Smoothing FPS number so it doesn't flicker around too much

            # --- HANDLE RESULTS ---
            if best_box:
                bx1, by1, bx2, by2 = best_box
                
                # Map coordinates back to the FULL frame after using ROI
                final_x1 = bx1 + offset_x
                final_y1 = by1 + offset_y
                final_x2 = bx2 + offset_x
                final_y2 = by2 + offset_y

                # Calculate the coordinates to display on the frame
                x_off, y_off, z = self.compute_xyz(final_x1, final_y1, final_x2, final_y2)
                
                # Smooth the data
                if self.latest_xyz is not None:

                    sx, sy, sz = self.latest_xyz

                    self.latest_xyz = (
                        SMOOTH * sx + (1-SMOOTH) * x_off,
                        SMOOTH * sy + (1-SMOOTH) * y_off,
                        SMOOTH * sz + (1-SMOOTH) * z
                    )
                else:
                    self.latest_xyz = (x_off, y_off, z)

                # Calling the function to transmit the coordinates to the other pi
                self.transmit_coords(self.latest_xyz)
                
                # Update tracking box for next frame's ROI
                tracked_box = (final_x1, final_y1, final_x2, final_y2)
                missed = 0

            else:

                # If no ball is found increase missed frames
                missed += 1
                if missed > MAX_MISSED: 
                    tracked_box = None
                    with self.location_lock:
                        self.latest_xyz = None
            
            # --- ADDING FRAME TO DISPLAY QUEUE ---
            self.display_queue.append((full_frame, tracked_box, fps))
        
    # --- HELPER FUNCTION TO GET LAST ANNOTATED FRAME FOR MAIN THREAD ---
    def get_display_frame(self):
        if self.display_queue:
            return self.display_queue.pop()
        return None

# --- MAIN THREAD ---
if __name__ == "__main__":

    # Defaulting show_win to true for debug run
    show_win = True

    # Checking for a third argument on execution to change show_win
    if len(sys.argv) > 1:
        show_win = sys.argv[1].lower() == 'true'

    tracker = GolfBallTracker(show_window=show_win)
    tracker.start()

    try:
        while tracker.running:

            # Capture frame on main thread
            tracker.grab_frame()

            # Get latest processed frame + metadata from worker
            data = tracker.get_display_frame()

            if tracker.show_window and data is not None:

                full_frame, tracked_box, fps = data

                h = tracker.HEIGHT
                w = tracker.WIDTH

                # Drawing ROI line
                roi_y_start = int(h * ROI_Y_FRACTION)
                cv2.line(full_frame, (0, roi_y_start), (w, roi_y_start), (0, 0, 255), 1)

                # Drawing bound around ball if detected
                if tracked_box is not None:
                    x1, y1, x2, y2 = tracked_box
                    cv2.rectangle(full_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Drawing coordinate text
                    loc = tracker.get_latest_location()
                    if loc:
                        x_off, y_off, z = loc
                        cv2.putText(full_frame, f"X:{int(x_off)} Y:{int(y_off)} Z:{int(z)}mm", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw FPS
                color = (0, 255, 0) if tracked_box else (0, 0, 255)
                cv2.putText(full_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("Golf Tracker Service (YOLO)", full_frame)

                # Handle keyboard inputs
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    tracker.running = False
                    break
                elif key & 0xFF == ord('c'):
                    tracker.save_calibration()
                    break

            else:
                if data is not None:

                   full_frame, tracked_box, fps = data

                   if tracked_box is not None:
                       loc = tracker.get_latest_location()
                       if loc is not None:
                           x, y, z = loc  # Unpack the latest coordinates
                           print(f"📥 BALL TRACKED -> X: {x:.2f} Y: {y:.2f} Z: {z:.2f}     FPS: {fps}")

    except KeyboardInterrupt:
        pass

    finally:
        tracker.stop()