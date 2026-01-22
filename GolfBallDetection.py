import cv2
import numpy as np
import math
import threading
import time
from picamera2 import Picamera2

class GolfBallTracker:
    def __init__(self):
        # 1. CAMERA SETTINGS
        self.CAMERA_RES = (1640, 1232)  # Binned high-res for better SNR
        self.PROC_WIDTH = 640           # Width for math processing (Fast!)
        self.scale_factor = self.PROC_WIDTH / self.CAMERA_RES[0]
        
        # 2. CALIBRATED MATH
        # Scale focal length: 1357 is for 1640px width.
        self.FOCAL_LENGTH = 1357 * self.scale_factor
        self.BALL_DIAM_MM = 42.67

        # 3. THREADING SHARED VARIABLES
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # 4. TRACKING STATE
        self.tracked_ball = None
        self.SMOOTHING = 0.6
        self.MAX_MISSED = 10
        self.missed_count = 0

        # Initialize Camera
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"format": "YUV420", "size": self.CAMERA_RES}
            )
            self.picam2.configure(config)
            self.picam2.start()
        except Exception as e:
            print(f"Camera Init Failed: {e}")
            self.running = False

    def capture_thread(self):
        """Dedicated thread to pull frames from the ISP as fast as possible."""
        while self.running:
            raw_frame = self.picam2.capture_array()
            # Extract Y channel (Grayscale) immediately - fastest for OpenCV
            gray = raw_frame[:self.CAMERA_RES[1], :self.CAMERA_RES[0]]
            
            with self.lock:
                self.frame = gray
            time.sleep(0.001) # Yield to main thread

    def run(self):
        # Start the capture thread
        t = threading.Thread(target=self.capture_thread, daemon=True)
        t.start()

        print("Tracking started. Press 'q' to quit.")
        
        try:
            while self.running:
                with self.lock:
                    if self.frame is None:
                        continue
                    # Work on a copy of the latest frame
                    working_frame = self.frame.copy()

                # A. FAST DOWNSIZE
                proc_res = (self.PROC_WIDTH, int(working_frame.shape[0] * self.scale_factor))
                small_frame = cv2.resize(working_frame, proc_res, interpolation=cv2.INTER_LINEAR)
                cx, cy = proc_res[0] // 2, proc_res[1] // 2

                # B. PATTERN-SENSITIVE PIPELINE
                # Adaptive thresholding is king for white-on-white 
                # It finds the "dimples" and "logo" even if the ball is white
                thresh = cv2.adaptiveThreshold(
                    small_frame, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )

                # C. MORPHOLOGY (Clean up the pattern noise)
                # We close the gaps between the dimple shadows to form a solid circle
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # D. FIND CONTOURS
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_candidate = None
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 100: continue

                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0: continue

                    circularity = (4 * math.pi * area) / (perimeter ** 2)
                    if circularity > 0.7:
                        ((px_x, px_y), radius) = cv2.minEnclosingCircle(cnt)
                        # We use area as a confidence score
                        if best_candidate is None or area > best_candidate[3]:
                            best_candidate = (px_x, px_y, radius, area)

                # D. COORDINATE MATH
                if best_candidate:
                    px_x, px_y, radius, _ = best_candidate
                    z = (self.BALL_DIAM_MM * self.FOCAL_LENGTH) / (radius * 2)
                    x_off = (px_x - cx) * (z / self.FOCAL_LENGTH)
                    y_off = (px_y - cy) * (z / self.FOCAL_LENGTH)
                    
                    r = math.sqrt(x_off**2 + y_off**2)
                    theta = math.degrees(math.atan2(y_off, x_off))
                    
                    new_data = (px_x, px_y, radius, r, theta, z)
                    
                    if self.tracked_ball is None:
                        self.tracked_ball = new_data
                    else:
                        self.tracked_ball = tuple(
                            self.SMOOTHING * old + (1 - self.SMOOTHING) * new 
                            for old, new in zip(self.tracked_ball, new_data)
                        )
                    self.missed_count = 0
                else:
                    self.missed_count += 1
                    if self.missed_count > self.MAX_MISSED:
                        self.tracked_ball = None

                # E. DISPLAY
                display_img = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2BGR)
                if self.tracked_ball:
                    bx, by, brad, _, _, bz = self.tracked_ball
                    cv2.circle(display_img, (int(bx), int(by)), int(brad), (0, 255, 0), 2)
                    cv2.putText(display_img, f"{int(bz)}mm", (int(bx), int(by)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Threaded Tracker", display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        finally:
            self.running = False
            self.picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = GolfBallTracker()
    tracker.run()