# yolo_ball_xyz.py
# Live YOLO “sports ball” detection + (X,Y,Z) estimate + FPS
# - Press 'c' to (re)calibrate at a known distance (default 0.30m)
# - Saves calibration to disk and auto-loads next run
# - Press 'q' to quit

import json
import os
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# --------------------------
# User-tunable constants
# --------------------------
BALL_DIAMETER_M = 0.04267          # golf ball diameter in meters
CALIB_Z0_M = 0.30                 # calibrate at 30 cm (you can change this)
CALIB_FILE = "camera_calib.json"  # saved in same folder as script

# COCO class id for "sports ball"
SPORTS_BALL_CLASS_ID = 32

# Performance / robustness knobs
FRAME_SIZE = (640, 480)           # keep fixed if you want calibration to remain valid
ROI_Y_FRACTION = 0.45             # only search bottom 55% of image (ball is on turf)
YOLO_MODEL = "yolov8n.pt"         # lightweight
YOLO_IMGSZ = 416                  # lower for faster, e.g. 320
YOLO_CONF = 0.25                  # raise if false positives, lower if misses

# Optional autofocus locking (Camera Module 3 is autofocus)
# If autofocus keeps changing, calibration can drift slightly.
LOCK_AUTOFOCUS = False            # set True to disable AF after it settles
AF_SETTLE_SECONDS = 1.0           # wait this long before locking (if enabled)
# --------------------------


def load_calibration():
    """Load fx from disk if present."""
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
            fx = float(data["fx"])
            print(f"Loaded calibration from {CALIB_FILE}: fx={fx:.1f}px")
            return fx
        except Exception as e:
            print(f"Warning: failed to load {CALIB_FILE}: {e}")
    return None


def save_calibration(fx):
    """Save fx to disk."""
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump({"fx": fx, "calib_Z0_m": CALIB_Z0_M, "ball_diam_m": BALL_DIAMETER_M, "frame_size": FRAME_SIZE}, f)
        print(f"Saved calibration to {CALIB_FILE}: fx={fx:.1f}px")
    except Exception as e:
        print(f"Warning: failed to save calibration: {e}")


def compute_xyz_from_box(x1, y1, x2, y2, fx, fy, cx, cy):
    """Estimate XYZ in camera frame using bbox width as apparent diameter."""
    w_px = max(1.0, (x2 - x1))
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    # Z from pinhole model: w_px ≈ fx * D / Z  =>  Z ≈ fx * D / w_px
    Z = (fx * BALL_DIAMETER_M) / w_px
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z, u, v, w_px


def main():
    # ---- Load calibration (optional) ----
    fx = load_calibration()
    fy = fx if fx is not None else None

    # ---- Camera init ----
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": FRAME_SIZE})
    picam2.configure(config)
    picam2.start()

    # Optional: let autofocus settle then lock it (helps stability)
    if LOCK_AUTOFOCUS:
        time.sleep(AF_SETTLE_SECONDS)
        # AfMode 0 disables continuous AF (keeps current lens position)
        # If this throws on your setup, just set LOCK_AUTOFOCUS=False.
        try:
            picam2.set_controls({"AfMode": 0})
            print("Autofocus locked (AfMode=0).")
        except Exception as e:
            print(f"Warning: couldn’t lock autofocus: {e}")

    # ---- YOLO init ----
    model = YOLO(YOLO_MODEL)

    print(f"Controls: c=(re)calibrate at {CALIB_Z0_M:.2f}m, q=quit")
    print("Tip: Aim camera mostly at floor/turf; keep walls out of frame.")

    # ---- FPS tracking ----
    prev_t = time.time()
    fps_ema = 0.0
    alpha = 0.15

    cx = cy = None

    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]
        if cx is None:
            cx, cy = w / 2.0, h / 2.0

        # ROI bottom part (reduces wall false positives + speeds up YOLO)
        roi_y0 = int(h * ROI_Y_FRACTION)
        roi = frame[roi_y0:h, 0:w]

        # YOLO inference
        results = model.predict(roi, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]

        # Pick the largest "sports ball" detection
        best = None
        best_area = 0.0
        if results.boxes is not None:
            for b in results.boxes:
                if int(b.cls[0]) != SPORTS_BALL_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2, float(b.conf[0]))

        # FPS update
        now = time.time()
        dt = max(1e-6, now - prev_t)
        inst_fps = 1.0 / dt
        fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
        prev_t = now

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw center crosshair (optional)
        cv2.drawMarker(frame, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

        if best is not None:
            x1, y1, x2, y2, conf = best
            y1_full = y1 + roi_y0
            y2_full = y2 + roi_y0

            # Draw box
            cv2.rectangle(frame, (int(x1), int(y1_full)), (int(x2), int(y2_full)), (0, 255, 0), 2)

            # Label above box
            label_y = max(20, int(y1_full) - 10)
            cv2.putText(frame, f"ball {conf:.2f}", (int(x1), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if fx is not None:
                X, Y, Z, u, v, w_px = compute_xyz_from_box(x1, y1_full, x2, y2_full, fx, fy, cx, cy)
                cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 0), -1)

                # Coordinates near box
                coord_text = f"X={X:+.2f}m Y={Y:+.2f}m Z={Z:.2f}m"
                coord_y = min(h - 10, int(y2_full) + 22)
                cv2.putText(frame, coord_text, (int(x1), coord_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Not calibrated (press 'c' at {CALIB_Z0_M:.2f}m)",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No ball detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("yolo_ball_xyz", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # (Re)calibrate whenever you want
        if key == ord("c"):
            if best is None:
                print("Calibrate failed: no ball detected.")
                continue

            x1, y1, x2, y2, conf = best
            w_px = max(1.0, (x2 - x1))
            fx = (w_px * CALIB_Z0_M) / BALL_DIAMETER_M
            fy = fx
            print(f"Calibrated fx≈{fx:.1f}px using bbox_w={w_px:.1f}px at Z0={CALIB_Z0_M:.2f}m")
            save_calibration(fx)

    cv2.destroyAllWindows()
    picam2.stop()


if __name__ == "__main__":
    main()


