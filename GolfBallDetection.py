import cv2
import numpy as np
import math

def track_patterned_golf_balls():
    # Constants Required for scaling of goolf balls to detemine z coordiante
    BALL_DIAM_MM = 42.67        # As googled (42.87mm)
    FOCAL_LENGTH = 600          # Length between len's center and cameras sensor (varies per camera)

    cap = cv2.VideoCapture(0)   # Passing default camera to video capture
    if not cap.isOpened():      # Error handle case
        return []

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # Contrast Limited Adaptive Histogram Equalization - Basically a mathematical function to fix funky light levels

    print("Press 'q' to stop.")

    # INITIALISING BUFFER FOR CONTOUR AND CAMERA DISPLAY ( Essentially a data structure but those don't really exist in python )
    tracked_ball = None   # (px_x, px_y, radius, r, theta, z)
    missed_frames = 0
    MAX_MISSED_FRAMES = 6
    SMOOTHING = 0.7        # EMA factor (closer to 1 = smoother)

    try:

        while True:

            ret, frame = cap.read() # Handling raw frame data and breaking if a frame isn't returned
            if not ret:
                break

            h, w, _ = frame.shape   # Attainnig the height and width of a single frame by unpacking the first two params of the frame and throwing away everything else
            cx, cy = w // 2, h // 2

            # 1. Applying masks and light filtering to try and isolate shapes
            lab         = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)                        # Converts to the LAB colour space (lightness, colour spectrum (green to magenta), Colour spectrum (blue to yellow))
            l, a, b     = cv2.split(lab)
            l_norm      = clahe.apply(l)                                                # Uses the CLAHE algorithm from earlier to raise the contrast of shaded edges on the light level of the new colour space
            final_bgr   = cv2.cvtColor(cv2.merge((l_norm, a, b)), cv2.COLOR_LAB2BGR)    # Merges the contrasted light level, original a and original b back together and it converts back to standard RGB

            # 2. Blurring image slightly and looking for sudden shifts in colour for edge detection
            blurred     = cv2.medianBlur(final_bgr, 7)
            edges       = cv2.Canny(blurred, 50, 150)

            # 3. Moving a kernel over the blended image to identify circles / ellipses
            kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed      = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Creates empty black frame to layer edges onto - edges in this case are the width and height
            mask_display = np.zeros_like(edges)

            # 4. Contour Detection
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Function takes arrays 

            best_candidate = None
            best_score = 0


            # TESTING CONTOUR AND DETERMINING A CONFIDENCE VALUE
            for cnt in contours:

                # BASIC ERROR CORRECTION FOR MASSIVE SURFACE OR TINY SURFACE
                area = cv2.contourArea(cnt)
                if area < 400:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                # Testing contour shape to see how circular it is
                circularity = (4 * math.pi * area) / (perimeter ** 2)   # Checking how circular a detected object is by checking its perimeter against area (perfect circle: 1, square: 0.785)
                hull = cv2.convexHull(cnt)                              # Getting the shape of the circumpherance of whatever shape has been picked up
                solidity = area / cv2.contourArea(hull)                 # How much of the hull is filled in by the area (checking if it has holes in it)

                # Determining a confidence score based on above calculations
                if circularity > 0.6 and solidity > 0.85:
                    ((px_x, px_y), radius) = cv2.minEnclosingCircle(cnt)

                    # Simple confidence score
                    score = circularity * solidity * area
                    if score > best_score:
                        best_score = score
                        best_candidate = (cnt, px_x, px_y, radius)



            # UPDATING THE BUFFER
            if best_candidate is not None:
                cnt, px_x, px_y, radius = best_candidate

                # Drawing the isolated white value to the full black background
                cv2.drawContours(mask_display, [cnt], -1, 255, -1)

                # Calculating coordinates
                z = (BALL_DIAM_MM * FOCAL_LENGTH) / (radius * 2)
                x_off = (px_x - cx) * (z / FOCAL_LENGTH)
                y_off = (px_y - cy) * (z / FOCAL_LENGTH)

                # Calculating r and theta based off this relative to the camera (probably better to change this later)
                r = math.sqrt(x_off ** 2 + y_off ** 2)
                theta = math.degrees(math.atan2(y_off, x_off))

                # Update the buffer to track a completely new ball so using the raw coordinates
                if tracked_ball is None:
                    tracked_ball = (px_x, px_y, radius, r, theta, z)

                # Update the buffer with smoothed out values between the old and new coordinates
                else:
                    tracked_ball = tuple(
                        SMOOTHING * old + (1 - SMOOTHING) * new     # Smoothing function for movement I found online
                        for old, new in zip(
                            tracked_ball,
                            (px_x, px_y, radius, r, theta, z)
                        )
                    )

                # Reset the missed frames (used in determining if it's been too long since a buffer update)
                missed_frames = 0

            else:
                missed_frames += 1
                if missed_frames > MAX_MISSED_FRAMES:
                    tracked_ball = None



            # DRAWING BALL AND ACCOMPANYING TEXT
            if tracked_ball is not None:
                px_x, px_y, radius, r, theta, z = tracked_ball

                cv2.circle(
                    frame,
                    (int(px_x), int(px_y)),
                    int(radius),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Dist: {int(z)}mm",
                    (int(px_x), int(px_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )



            cv2.imshow("Original Feed (CLAHE Normalized)", frame)
            cv2.imshow("Golf Ball Isolation Mask (B&W)", mask_display)

            # Quit on "q" key input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Whatever happens release the camera and delete the frames
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return []


if __name__ == "__main__":
    track_patterned_golf_balls()