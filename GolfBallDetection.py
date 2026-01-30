import cv2
import numpy as np
import math
import threading
import time
from picamera2 import Picamera2

class GolfBallTracker:
    def __init__(self):
        # Values to allow depth perception and calculation of distance to golf ball
        self.BALL_DIAM_MM = 42.67
        self.FOCAL_LENGTH = 1675 

        # Scalings of resolution for display and processing of image (to not overload the CPU)
        self.DETECT_SCALE = 0.5
        self.DISPLAY_SCALE = 0.3

        # Initialisation of the camera with configured parameters
        """
        YUV420 allows for skipping of step to convert from RGB to LAB - cutting down on processing,
        especially for larger resolutions that would greatly increase the load on the CPU
        """
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"format": "YUV420", "size": (1920, 1080)})
        self.picam2.configure(config)
        self.picam2.start()

        # Camera Settings (Configure for detection params)
        self.picam2.set_controls({
            "AeEnable": False,                      # Stops the camera from automatically brightening a darkened room
            "FrameDurationLimits": (16666, 16666),  # Locking the frame rate to a max of 60 in order to eliminate jitter with processing variations
            "ExposureTime": 18000,                  # Controls how long the camera's shutter stays open
            "AnalogueGain": 1.2,                    # Gain is self explanitory, maintain under ~2 to avoid noise (but is auto adjusted later in code)
            "Brightness": -0.275                    # Offsets the black level of the image (Has a massive effect relative to the other two)
        })

        # Initialising parameters for capturing frames, threading and proceedurally altering gain to attain bare minimum light levels
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.current_gain = 1.2

    # Threaded function to capture frames on the camera
    def camera_stream(self):

        frame_count = 0
        while self.running:
            raw = self.picam2.capture_array()       # Caputures the frame as an np array
            gray = raw[:1080, :1920]                # Isolates light level (light level stored in first 0-n indexes of row and column)

            if frame_count % 30 == 0:               # Every 30 frames, check the average light level and adjust accordingly
                avg = np.mean(gray)

                """
                TODO: CHANGE TO INCREASE GRADUALLY FROM A LOWER GAIN
                TO HIGHER GAIN RATHER THAN CUT TO EITHER
                """
                new_gain = 1.7 if avg < 70 else 1.2
                if new_gain != self.current_gain:
                    self.picam2.set_controls({"AnalogueGain": new_gain})
                    self.current_gain = new_gain

            with self.lock:
                self.frame = gray
            frame_count += 1

    def track_golf_ball(self):
        # Initialising math stuff I don't know much about
        backSub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=25, detectShadows=False) # Used to detect what's static and dynamic in a background
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))                                     # Increases contrast in localised shapes within the image

        # Defining parameters for the tracking of the ball
        tracked = None
        SMOOTH = 0.5
        missed = 0
        MAX_MISSED = 15
        
        # Starting the thread to capture frames
        threading.Thread(target=self.camera_stream, daemon=True).start()

        try:
            while self.running:
                with self.lock:
                    if self.frame is None: continue # If there is no captured frame then move on
                    gray_full = self.frame.copy()

                # Creating a gray mask (at a lower resolution to save on processing power)
                gray_small = cv2.resize(gray_full, None, fx=self.DETECT_SCALE, fy=self.DETECT_SCALE)
                h, w = gray_small.shape

                # ROI Logic - uses the ROI if the ball is currently being tracked and the ball has missed 0 frames being tracked
                if tracked and missed == 0:
                    tx, ty, tr = tracked[0]*self.DETECT_SCALE, tracked[1]*self.DETECT_SCALE, tracked[2]*self.DETECT_SCALE

                    # Creates a padding box to look for the ball - 2.5x bigger to allow for ball movements and allow for error correction
                    pad = int(tr * 2.5)
                    x0, y0 = max(0, int(tx-pad)), max(0, int(ty-pad))
                    x1, y1 = min(w, int(tx+pad)), min(h, int(ty+pad))

                    # Cuts the image down so that instead of running HoughCircles on the entire processing frame it will only be a range of maybe 100x100 pixels of intense math
                    if (x1-x0) > 10 and (y1-y0) > 10:
                        proc_frame = gray_small[y0:y1, x0:x1]
                        offset = (x0, y0)
                else:
                    # If the ball isn't currently being tracked then it uses the full default image
                    proc_frame = gray_small
                    offset = (0, 0)

                # Pipelining image altering / processing
                contrasted = clahe.apply(proc_frame)                      # Applys the mask to increase contrast
                static     = backSub.apply(enhanced, learningRate=0.01)   # Checks what's static and dynamic with prior mask
                """
                Using the second derivative (laplacian - instead of canny (detects colour changes)) 
                to detect curvature and rapid transitions (depth)
                TODO: COULD MAYBE IMPLEMENT CANNY DETECTION ALONGSIDE LAPLACIAN DETECTION FOR 
                      NON-MONOCHROMATIC BALL AND BACKGROUNDS
                """
                gradients  = cv2.convertScaleAbs(cv2.Laplacian(enhanced, cv2.CV_16S, ksize=3))
                combined   = cv2.bitwise_and(grad, fg_mask)

                # Using Hough Circles (more math I don't know much about) to detect circles within the image
                """
                Essentially a mathematical voting algorithm to determine if a circle is present. For every dot on a
                backdrop, it'll draw a circle of radius x around it. If those circles all intersect at a midpoint (x, y)
                then that circle gains a vote.

                TODO: CHECK WHICH IS MORE EFFECTIVE - HOUGH CIRCLES or MORPHOLOICAL ANALYSIS WITH ELLIPSES 
                      (AS BALL FREQUENTLY DETECTED AS ELLIPSE IN YUV, NOT A PERFECT CIRCLE)
                """
                circles  = cv2.HoughCircles(
                    combined,                               # Input altered image (just as a reminder)
                    cv2.HOUGH_GRADIENT,                     # Checks gradients of light changes rather than every pixel (saving CPU usage)

                    dp=1.4,                                 # Grid size for looking for points (1.0 is same resolution as image)
                    minDist=40,                             # Minimum distance in CM between centers of detected circles

                    param1=50,                              # Sensitivity of internal Canny detection (high values look for more contrasting edges)
                    param2=25,                              # How many votes an object needs to be confirmed as a center

                    minRadius=int(10*self.DETECT_SCALE),    # Defined the physical size of the search (i.e. radius of each circle from point)
                    maxRadius=int(150*self.DETECT_SCALE)
                )

                # Finding the best candidate circle
                best = None
                if circles is not None:

                    # Converts the values to nearest whole number and converts to unsigned 16 bit integer ()
                    """
                    TODO: FIND OUT THE SEVERITY OF TRUNCATING THE CIRCLES[0] RATHER THAN ROUNDING IT THEN TYPE
                          CASTING AS THERE IS DATA LOSS.
                    """
                    #circles = np.uint16(circles[0])        # Single value operation
                    circles = circles[0].astype(np.uint16)  # Array operation - Speed increase using the array operation is drastic due to function call overhead
                    for cx, cy, cr in circles:

                        # Correcting for use of ROI using the offsets for x and y
                        check_y, check_x = int(cy + offset[1]), int(cx + offset[0])

                        if 0 <= check_y < h and 0 <= check_x < w:

                            # Final brightness check to ensure BRIGHT white balls are all that's detected
                            """
                            TODO: PLAY AROUND WITH THE FINAL BRIGHTNESS CHECKER ALONG WITH THE CAMERA VALUES.
                                  OR FIND OUT IF THERE IS A MATHEMATICAL RELATION BETWEEN THEM AND CALCULATE WHAT
                                  THE CENTER BRIGHTNESS SHOULD BE
                            """
                            if gray_small[check_y, check_x] > 160:
                                best = (check_x / self.DETECT_SCALE, check_y / self.DETECT_SCALE, cr / self.DETECT_SCALE)   # Scaling the detection coordinates back up to be displayed at standard scale
                                break

                # Handling the best case circle
                if best:
                    x, y, r = best
                    cx_f, cy_f = 1920 // 2, 1080 // 2

                    # Calculating the depth of the ball from the camera (z) as well as parameters for calculating the 
                    z = (self.BALL_DIAM_MM * self.FOCAL_LENGTH) / (2 * r)
                    x_off = (x - cx_f) * (z / self.FOCAL_LENGTH)            # Distances from the middle of the camera lense
                    y_off = (y - cy_f) * (z / self.FOCAL_LENGTH)
                    
                    # Handling the updating of the currently tracked ball and resetting the number of frames missed on screen
                    current = (x, y, r, math.sqrt(x_off**2 + y_off**2), math.degrees(math.atan2(y_off, x_off)), z)
                    tracked = current if tracked is None else tuple(SMOOTH*o + (1-SMOOTH)*n for o, n in zip(tracked, current))
                    missed = 0
                else:
                    # If no ball is found (there is no best candidate) then it needs to increase missed frames and update tracking accordingly
                    missed += 1
                    if missed > MAX_MISSED: tracked = None

                # Displaying the frame
                disp = cv2.resize(cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR), None, fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE)

                # Displaying the circle if a ball is being tracked
                if tracked:
                    cv2.circle(disp, (int(tracked[0]*self.DISPLAY_SCALE), int(tracked[1]*self.DISPLAY_SCALE)), int(tracked[2]*self.DISPLAY_SCALE), (0,255,0), 2)
                
                cv2.imshow("Light Level Tracker", disp)

        # Ending the camera to prevent errors in later testing (needs done to protect hardware and testing environment)
        finally:
            self.running = False
            self.picam2.stop()
            cv2.destroyAllWindows()

# ON START DO STUFF
if __name__ == "__main__":
    GolfBallTracker().track_golf_ball()