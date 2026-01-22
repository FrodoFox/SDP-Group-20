import cv2
import numpy as np
import math
from picamera2 import Picamera2

def track_patterned_golf_balls():
    # 1. CAMERA CONFIGURATION
    # Pi Camera V2.1 Max Resolution is 3280 x 2464
    # Note: Processing 8MP frames in real-time on a Pi will be slow. 
    RESOLUTION = (1640, 1232) 
    
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"format": "BGR888", "size": RESOLUTION})
    picam2.configure(config)
    picam2.start()

    # Constants for distance calculation
    BALL_DIAM_MM = 42.67    # Standard golf ball diameter in mm
    FOCAL_LENGTH = 1357     # in pixels, needs calibration for accuracy

    # Image Processing Tools
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Tracking State
    tracked_ball = None   # (px_x, px_y, radius, r, theta, z)
    missed_frames = 0
    MAX_MISSED_FRAMES = 6
    SMOOTHING = 0.7        
    results = (0, 0, 0)

    print(f"Camera Started at {RESOLUTION}. Press 'Ctrl+C' in terminal to stop.")

    try:
        while True:
            # 2. CAPTURE FRAME
            frame = picam2.capture_array()
            
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2

            # 3. COLOUR SPACE TRANSFORMATION & NORMALIZATION
            lab         = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b     = cv2.split(lab)
            l_norm      = clahe.apply(l)
            final_bgr   = cv2.cvtColor(cv2.merge((l_norm, a, b)), cv2.COLOR_LAB2BGR)

            # 4. EDGE DETECTION & MORPHOLOGY
            # We use a median blur to reduce noise before Canny
            blurred     = cv2.medianBlur(final_bgr, 7)
            edges       = cv2.Canny(blurred, 50, 150)

            # Morphological Closing to fill gaps in the detected edges of the ball
            kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed      = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 5. CONTOUR DETECTION
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_candidate = None
            best_score = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Scaled area check (higher resolution needs higher min area)
                if area < 1000: 
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue

                circularity = (4 * math.pi * area) / (perimeter ** 2)
                hull = cv2.convexHull(cnt)
                solidity = area / cv2.contourArea(hull)

                # Look for high circularity and solidity (typical of a sphere)
                if circularity > 0.6 and solidity > 0.85:
                    ((px_x, px_y), radius) = cv2.minEnclosingCircle(cnt)
                    score = circularity * solidity * area
                    if score > best_score:
                        best_score = score
                        best_candidate = (cnt, px_x, px_y, radius)

            # 6. COORDINATE CALCULATION & FILTERING
            if best_candidate is not None:
                cnt, px_x, px_y, radius = best_candidate
                
                # Z calculation: Dist = (KnownDiam * FocalLength) / PixelDiameter
                z = (BALL_DIAM_MM * FOCAL_LENGTH) / (radius * 2)
                x_off = (px_x - cx) * (z / FOCAL_LENGTH)
                y_off = (px_y - cy) * (z / FOCAL_LENGTH)

                r = math.sqrt(x_off ** 2 + y_off ** 2)
                theta = math.degrees(math.atan2(y_off, x_off))

                if tracked_ball is None:
                    tracked_ball = (px_x, px_y, radius, r, theta, z)
                else:
                    # Exponential Moving Average Smoothing
                    tracked_ball = tuple(
                        SMOOTHING * old + (1 - SMOOTHING) * new
                        for old, new in zip(tracked_ball, (px_x, px_y, radius, r, theta, z))
                    )
                missed_frames = 0
            else:
                missed_frames += 1
                if missed_frames > MAX_MISSED_FRAMES:
                    tracked_ball = None

            # 7. VISUALIZATION
            if tracked_ball is not None:
                px_x, px_y, radius, r, theta, z = tracked_ball
                results = (r, theta, z)

                cv2.circle(frame, (int(px_x), int(px_y)), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"Dist: {int(z)}mm", (int(px_x), int(px_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Displaying high-res windows can be heavy; resize for preview
            preview_frame = cv2.resize(frame, (800, 600))
            cv2.imshow("PiCam V2 Tracking", preview_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    return results

if __name__ == "__main__":
    data = track_patterned_golf_balls()
    print(f"Final Tracking Data (r, theta, z): {data}")