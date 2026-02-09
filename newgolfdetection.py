import cv2
import numpy as np
from ultralytics import YOLO  # Requires 'pip install ultralytics'

# Known parameters
BALL_DIAMETER_MM = 42.7  # golf ball diameter
FOCAL_LENGTH_MM = 3.6    # approx Pi camera focal length in mm
SENSOR_WIDTH_MM = 3.76   # Pi Camera v3 sensor width in mm
IMAGE_WIDTH_PX = 640      # Resolution width
IMAGE_HEIGHT_PX = 480     # Resolution height

# Load YOLO model (you can train your own golf ball model)
model = YOLO("yolov8n.pt")  # replace with your fine-tuned golf ball model

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)

def estimate_distance(ball_pixel_diameter):
    """
    Estimate Z distance using similar triangles.
    Z = (F * real_width * image_width_px) / (pixel_width * sensor_width)
    """
    z = (FOCAL_LENGTH_MM * BALL_DIAMETER_MM * IMAGE_WIDTH_PX) / (ball_pixel_diameter * SENSOR_WIDTH_MM)
    return z  # in mm

def pixel_to_world(x_pixel, y_pixel, z_mm):
    """
    Convert pixel coordinates to real-world coordinates relative to camera.
    """
    x_centered = x_pixel - IMAGE_WIDTH_PX / 2
    y_centered = y_pixel - IMAGE_HEIGHT_PX / 2

    x_mm = x_centered * z_mm * SENSOR_WIDTH_MM / (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX)
    y_mm = y_centered * z_mm * SENSOR_WIDTH_MM / (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX)
    return x_mm, y_mm, z_mm

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]

    for box in results.boxes:
        # box.xyxy = [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w = x2 - x1
        h = y2 - y1

        # Estimate distance using width in pixels
        z = estimate_distance(w)
        x_pixel = x1 + w / 2
        y_pixel = y1 + h / 2

        x, y, z = pixel_to_world(x_pixel, y_pixel, z)

        # Draw bounding box and coordinates
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"x:{x:.1f} y:{y:.1f} z:{z:.1f}mm", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Golf Ball Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
