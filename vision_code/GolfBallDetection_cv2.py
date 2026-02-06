import sys
import os

# --- PATHING FOR DEPENDENCIES ON TURTLEBOT ---
#script_dir  = os.path.dirname(os.path.abspath(__file__))
#deps_path   = os.path.join(script_dir, "..", "..", "dependencies")
#sys.path.insert(0, os.path.abspath(deps_path))

import cv2
import numpy as np
import math
import threading
import time
from collections import deque

# --- CAMERA CONFIG ---
EXPOSURE_TIME    = 10000
GAMMA            = 0.8
ANALOGUE_GAIN    = 1.5
BRIGHTNESS       = -0.2
BRIGHTNESS_CHECK = 0

# --- BALL TRACKING CONFIG ---
MAX_MISSED = 15                 # Frame limit for maximum allowed missed frames before a ball is called no longer detected
SMOOTH = 0.5

class GolfBallTracker:
    def __init__(self, show_window=False):

        # Values to allow depth perception and calculation of distance to golf ball
        self.BALL_DIAM_MM = 42.67
        self.FOCAL_LENGTH = 1675 

        # Scalings of resolution for display and processing of image (to not overload the CPU)
        self.DETECT_SCALE       = 0.5
        self.DISPLAY_SCALE      = 0.3
        self.WIDTH, self.HEIGHT = 1920, 1080
        self.show_window        = show_window

        # Initialisation of the camera with configured parameters
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Configuring resolution via OpenCV properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        # Same as prior parameters - just in the CV2 library
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)             # Stops the camera from automatically brightening (1=Manual)
        self.cap.set(cv2.CAP_PROP_EXPOSURE,      EXPOSURE_TIME) # Exposure Time
        self.cap.set(cv2.CAP_PROP_GAIN,          ANALOGUE_GAIN) # Analogue Gain
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,    BRIGHTNESS)    # Offsets the black level

        # Using a deque in order to ensure it always has the most recent frame
        self.frame_queue  = deque(maxlen=1)
        self.running      = True
        self.current_gain = 1.2             # Initial current gain for proceedural alteration
        
        # External access variables
        self.location_lock = threading.Lock()
        self.latest_xyz    = None               # Stores (x, y, z) offsets
        
        # Initialise gamma lookup table
        self.gamma_lut = self.build_gamma_lut(GAMMA)

    def build_gamma_lut(self, gamma):

        # Creating a lookup table to map specific gamma curve values (better than altering hardware)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return table

    def apply_gamma(self, image):

        # Applies computes gamma table to the frame
        return cv2.LUT(image, self.gamma_lut)

    def camera_stream(self):
        frame_count = 0
        while self.running:

            # Captures the frame as a BGR numpy array (FAR SLOWER BUT NEEDS DONE ON CV2)
            ret, bgr_frame = self.cap.read()
            
            if not ret:
                continue

            # Convert BGR to YUV (This is the line that was magically avoided using picam2. This slows it down A LOT)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

            if frame_count % 30 == 0:               # Every 30 frames, check the average light level and adjust accordingly
                avg = np.mean(gray)
                new_gain = 1.7 if avg < 70 else 1.2
                if new_gain != self.current_gain:

                    # Adjusting Analogue Gain via CV2 property
                    self.cap.set(cv2.CAP_PROP_GAIN, new_gain)
                    self.current_gain = new_gain

            # Puts the new frame onto the queue
            self.frame_queue.append(gray)
            frame_count += 1

    def start(self):

        # Starting the camera stream
        if not self.cap.isOpened():
            print("Failed to open camera. Check hardware connection.")
            return

        # Starting background threads for tracking and processing
        threading.Thread(target=self.camera_stream, daemon=True).start()
        threading.Thread(target=self.tracking_loop, daemon=True).start()

    def stop(self):

        # Release hardware at the end
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def get_latest_location(self):

        # Returns latest known location of any tracked ball
        with self.location_lock:
            return self.latest_xyz

    def tracking_loop(self):

        # Initialising math stuff
        backSub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=25, detectShadows=False) # Creating the background detection to find what's static and mobile in the frame
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))                                   # Creating a CLAHE kernel with clipping and a larger (desired) grid size for better comparisons and contrast

        # Initialising values for tracking ball
        tracked = None
        missed = 0

        try:
            while self.running:
                if not self.frame_queue:
                    time.sleep(0.001)
                    continue
                
                # Image pre-processing
                gray_raw  = self.frame_queue.pop()      # Taking the first frame from the queue
                gray_full = self.apply_gamma(gray_raw)  # Applying the gamma LUT in order to increase / decrease shadows

                # Collecting sizes of the frame before applying ROI logic
                gray_small = cv2.resize(gray_full, None, fx=self.DETECT_SCALE, fy=self.DETECT_SCALE)
                h, w = gray_small.shape

                # ROI Logic - if ball existed in this area before then check around this area again...
                if tracked and missed == 0:

                    # Collecting the x, y and radius values from the ball that was being tracked before
                    tx, ty, tr = tracked[0]*self.DETECT_SCALE, tracked[1]*self.DETECT_SCALE, tracked[2]*self.DETECT_SCALE
                    pad = int(tr * 2.5)

                    # Selecting limits for the ROI using the radius of the prior detected circle with some padding for movement and error correction
                    x0, y0 = max(0, int(tx-pad)), max(0, int(ty-pad))
                    x1, y1 = min(w, int(tx+pad)), min(h, int(ty+pad))

                    # Checking the box selected is a real box before clipping the frame and recording an offset of the ROI window
                    if (x1-x0) > 10 and (y1-y0) > 10:
                        proc_frame = gray_small[y0:y1, x0:x1]
                        offset = (x0, y0)
                    
                    # ROI window isn't valid so it uses the full frame
                    else:
                        proc_frame, offset = gray_small, (0, 0)

                # No tracked ball so uses the full frame
                else:
                    proc_frame, offset = gray_small, (0, 0)

                # Pipelining image altering / processing
                contrasted = clahe.apply(proc_frame)                      # Applys the mask to increase contrast
                static     = backSub.apply(contrasted, learningRate=0.01) # Checks what's static and dynamic with prior mask
                
                # Using the second derivative (Laplacian) to detect curvature and rapid transitions
                gradients  = cv2.convertScaleAbs(cv2.Laplacian(contrasted, cv2.CV_16S, ksize=3))
                combined   = cv2.bitwise_and(gradients, static)

                circles  = cv2.HoughCircles(
                    combined,                               # Input altered image
                    cv2.HOUGH_GRADIENT,                     # Checks gradients of light changes
                    dp=1.4,                                 # Grid size for looking for points
                    minDist=40,                             # Minimum distance in pixels between centers
                    param1=50,                              # Sensitivity of internal Canny detection
                    param2=25,                              # How many votes an object needs to be confirmed as a center
                    minRadius=int(10*self.DETECT_SCALE),    # Min search radius
                    maxRadius=int(150*self.DETECT_SCALE)    # Max search radius
                )

                # Finding the best candidate circle
                best = None
                if circles is not None:
                    circles = circles[0].astype(np.uint16)  # Array operation for speed
                    for cx, cy, cr in circles:
                        # Correcting for use of ROI using the offsets for x and y
                        check_y, check_x = int(cy + offset[1]), int(cx + offset[0])

                        if 0 <= check_y < h and 0 <= check_x < w:

                            # Final brightness check to ensure BRIGHT white balls are all that's detected
                            if gray_small[check_y, check_x] > BRIGHTNESS_CHECK:

                                # Scaling back to 1920x1080 baseline for depth math
                                best = (check_x / self.DETECT_SCALE, check_y / self.DETECT_SCALE, cr / self.DETECT_SCALE)
                                break

                # Handling the best case circle
                if best:
                    x, y, r = best
                    cx_f, cy_f = self.WIDTH // 2, self.HEIGHT // 2

                    # Calculating the depth of the ball from the camera (z)
                    z = (self.BALL_DIAM_MM * self.FOCAL_LENGTH) / (2 * r)
                    x_off = (x - cx_f) * (z / self.FOCAL_LENGTH)            # Distances from the middle of the camera lense
                    y_off = (y - cy_f) * (z / self.FOCAL_LENGTH)
                    
                    # Handling the updating of the currently tracked ball and resetting the number of frames missed on screen
                    current = (x, y, r, math.sqrt(x_off**2 + y_off**2), math.degrees(math.atan2(y_off, x_off)), z)
                    tracked = current if tracked is None else tuple(SMOOTH*o + (1-SMOOTH)*n for o, n in zip(tracked, current))
                    
                    # Updating the location being tracked so that it can be returned
                    with self.location_lock:
                        self.latest_xyz = (x_off, y_off, z)
                else:
                    # If no ball is found increase missed frames
                    missed += 1
                    if missed > MAX_MISSED: 
                        tracked = None

                        # Resetting the tracked location to be none
                        with self.location_lock:
                            self.latest_xyz = None

                # Displaying the CV2 window if that was desired during programming
                if self.show_window:

                    # Resizing the grayed window into the display scale and displaying (or trying to) in RGB
                    disp = cv2.resize(gray_full, None, fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE)
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

                    # Handling the displaying of the ball by checking if something is tracked and drawing a circle
                    if tracked:
                        d_x, d_y, d_r = int(tracked[0]*self.DISPLAY_SCALE), int(tracked[1]*self.DISPLAY_SCALE), int(tracked[2]*self.DISPLAY_SCALE)
                        cv2.circle(disp, (d_x, d_y), d_r, (0, 255, 0), 2)

                    # Creating the window.
                    cv2.imshow("Golf Tracker Service", disp)

                    # DON'T DELETE - OR THE FRAME WILL FREEZE
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        self.running = False
        
        finally:
            self.stop()