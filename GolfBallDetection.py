import cv2
import numpy as np
import math
import threading
import time
from picamera2 import Picamera2

class GolfBallTracker:
    def __init__(self):
        # CONSTANTS NEEDED FOR SCALING THE BALL AND DISTANCE CALCULATIONS
        self.BALL_DIAM_MM = 42.67        # As googled (42.87mm)
        self.FOCAL_LENGTH = 1357         # Focal length for the resolution used (1640x1232) in pixels

        # CAMERA CONFIGURATION
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"format": "YUV420", "size": (1640, 1232)})    # Using YUV420 format because the Y channel is essentially Grayscale/Lightness.
        self.picam2.configure(config)
        self.picam2.start()

        # THREADING VARIABLES
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    # FUNCTION TO THREAD THE FETCHING OF CAMERA FRAMES
    def camera_stream(self):
        while self.running:
            raw_data = self.picam2.capture_array()
            gray_frame = raw_data[:1232, :1640]     # In YUV420, the first 'h' rows are the Y (Lightness) channel.
            with self.lock:
                self.frame = gray_frame
            time.sleep(0.01)

    def track_patterned_golf_balls(self):

        # MATH STUFF I DON'T UNDERSTAND (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 

        # STARTING THE BUFFER FOR THE TRACKING
        tracked_ball = None     # (px_x, px_y, radius, r, theta, z)
        missed_frames = 0
        MAX_MISSED_FRAMES = 6
        SMOOTHING = 0.7         # EMA factor (closer to 1 = smoother)
        results = (0, 0, 0)

        # STARTING THE THREAD TO CAPTURE FRAMES
        stream_thread = threading.Thread(target=self.camera_stream, daemon=True)
        stream_thread.start()

        print("Press 'q' to stop.")

        try:
            while self.running:
                with self.lock:
                    if self.frame is None:
                        continue
                    # Working on the Lightness channel directly
                    l_channel = self.frame.copy()

                h, w = l_channel.shape   
                cx, cy = w // 2, h // 2

                # LAYERS OF IMAGE PROCESSING TO ISOLATE THE BALL
                # 1. Applying CLAHE to lightness level to normalize lighting conditions
                l_norm = clahe.apply(l_channel)                                                

                # 2. Blurring image slightly and looking for sudden shifts in colour
                blurred     = cv2.medianBlur(l_norm, 7)                                     # applying median blur to reduce noise and small details (7x7 kernel)
                edges       = cv2.Canny(blurred, 50, 150)                                   # using Canny edge detection to find sudden changes in colour

                # 3. Moving a kernel over the blended image to identify circles / ellipses
                kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                closed      = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # Creates empty black frame to layer edges onto
                mask_display = np.zeros_like(edges)

                # 4. Contour Detection
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

                best_candidate = None
                best_score = 0

                # TESTING CONTOUR AND DETERMINING A CONFIDENCE VALUE
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 400:
                        continue

                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue

                    # TESTING CONTOUR SHAPE TO FIND HOW CIRCULAR IT IS
                    circularity = (4 * math.pi * area) / (perimeter ** 2)   
                    hull = cv2.convexHull(cnt)                              # Converting the contour into a convex hull - Equivalent to stretching a rubber band around the shape detected by pixels
                    solidity = area / cv2.contourArea(hull)                 # Comparing the area of pixels detected by the camera to the area detected by putting a rubber band around it (filtering shadows and obstructions)

                    # COMPUTING A CONFIDENCE SCORE
                    if circularity > 0.6 and solidity > 0.85:
                        ((px_x, px_y), radius) = cv2.minEnclosingCircle(cnt)

                        score = circularity * solidity * area               
                        if score > best_score:
                            best_score = score
                            best_candidate = (cnt, px_x, px_y, radius)

                # UPDATING THE BUFFER
                if best_candidate is not None:
                    cnt, px_x, px_y, radius = best_candidate

                    # DRAWING THE SHAPE ON A FULL BLACK BACKDROP
                    cv2.drawContours(mask_display, [cnt], -1, 255, -1)

                    # CALCUATING COORDINTES RELATIVE TO THE CAMERA
                    z = (self.BALL_DIAM_MM * self.FOCAL_LENGTH) / (radius * 2)
                    x_off = (px_x - cx) * (z / self.FOCAL_LENGTH)
                    y_off = (px_y - cy) * (z / self.FOCAL_LENGTH)

                    # CALCULATING R AND THETA
                    r = math.sqrt(x_off ** 2 + y_off ** 2)
                    theta = math.degrees(math.atan2(y_off, x_off))

                    # UPDATING THE TRACKED BALL 
                    if tracked_ball is None:
                        tracked_ball = (px_x, px_y, radius, r, theta, z)
                    # if a ball is detected then it smoothes the movement of the buffer
                    else:
                        tracked_ball = tuple(
                            SMOOTHING * old + (1 - SMOOTHING) * new     
                            for old, new in zip(
                                tracked_ball,
                                (px_x, px_y, radius, r, theta, z)
                            )
                        )
                    missed_frames = 0
                
                # IF NO BALL IS DETECTED
                else:
                    missed_frames += 1
                    if missed_frames > MAX_MISSED_FRAMES:
                        tracked_ball = None

                # DRAWING BALL AND ACCOMPANYING TEXT
                display_frame = cv2.cvtColor(l_norm, cv2.COLOR_GRAY2BGR)    # (Converting L back to BGR only for the display window)
                if tracked_ball is not None:
                    px_x, px_y, radius, r, theta, z = tracked_ball
                    results = (r, theta, z)

                    cv2.circle(display_frame, (int(px_x), int(px_y)), int(radius), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Dist: {int(z)}mm", (int(px_x), int(px_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # DISPLAYING THE FRAMES OF THE NORMAL CAMERA AND THE ISOLATED MASK
                cv2.imshow("Original Feed (CLAHE Normalized)", cv2.resize(display_frame, (800, 600)))
                cv2.imshow("Golf Ball Isolation Mask (B&W)", cv2.resize(mask_display, (800, 600)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        finally:
            self.running = False
            self.picam2.stop()
            cv2.destroyAllWindows()

        return results

if __name__ == "__main__":
    tracker = GolfBallTracker()
    data = tracker.track_patterned_golf_balls()
    print(f"Final Data: {data}")
