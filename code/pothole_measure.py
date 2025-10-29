#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
from picamera2 import Picamera2

# -----------------------------
# Configuration
# -----------------------------
CALIB_FILE = "calibrationFiles/calibration.yaml"  # Path to your YAML calibration
PX_PER_MM = None       # Optional: manually set pixels/mm; if None, will compute using checkerboard
CHECKER_PATTERN = (9,6)
SQUARE_SIZE_MM = 25.0  # size of one checkerboard square in mm
SAVE_DIR = "./pothole_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Load Calibration
# -----------------------------
with open(CALIB_FILE) as fr:
    c = yaml.safe_load(fr)

K = np.array(c['camera_matrix'])
dist = np.array(c['dist_coefs'])

# -----------------------------
# Helper Functions
# -----------------------------
def undistort_image(img):
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w,h), cv2.CV_16SC2)
    img_und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return img_und

def detect_checkerboard_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKER_PATTERN)
    if not found:
        return None
    p1, p2 = corners[0,0], corners[1,0]
    pixel_dist = np.linalg.norm(p1 - p2)
    px_mm = pixel_dist / SQUARE_SIZE_MM
    return px_mm

def detect_potholes(img, px_per_mm, min_area_mm2=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potholes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        width_mm = w / px_per_mm
        height_mm = h / px_per_mm
        area_mm2 = width_mm * height_mm
        if area_mm2 < min_area_mm2:
            continue
        potholes.append((x, y, w, h, width_mm, height_mm))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(img, f"{width_mm:.0f}x{height_mm:.0f}mm", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img, potholes

# -----------------------------
# Initialize Camera
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280,720)})
picam2.configure(config)
picam2.start()

print("Live pothole measurement started.")
print("Press 'q' to quit, 'c' to capture snapshot.")

# -----------------------------
# Main Loop
# -----------------------------
while True:
    frame = picam2.capture_array()
    frame_und = undistort_image(frame)

    # Compute px/mm from checkerboard if not manually set
    if PX_PER_MM is None:
        px_per_mm = detect_checkerboard_scale(frame_und)
        if px_per_mm is None:
            cv2.putText(frame_und, "Checkerboard not detected! Using last known scale or set manually.",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            if 'last_px_per_mm' in globals():
                px_per_mm = last_px_per_mm
            else:
                px_per_mm = 2.0  # fallback default
        else:
            last_px_per_mm = px_per_mm
    else:
        px_per_mm = PX_PER_MM

    annotated_frame, potholes = detect_potholes(frame_und, px_per_mm)
    cv2.imshow("Pothole Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # save snapshot
        filename = os.path.join(SAVE_DIR, "snapshot.png")
        cv2.imwrite(filename, annotated_frame)
        print(f"Snapshot saved to {filename}")

picam2.close()
cv2.destroyAllWindows()
