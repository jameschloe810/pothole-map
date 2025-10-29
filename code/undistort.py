#!/usr/bin/env python3
import cv2
import numpy as np
import os
import argparse
import yaml
from glob import glob
from picamera2 import Picamera2
import time

def load_calibration(calib_file):
    """Load calibration parameters from YAML."""
    with open(calib_file) as fr:
        c = yaml.safe_load(fr)
    K = np.array(c['camera_matrix'])
    dist = np.array(c['dist_coefs'])
    square_size = c.get('square_size_mm', 25.0)  # default 25 mm
    return K, dist, square_size

def undistort_and_measure(img, K, dist, square_size, pattern_size):
    """Undistort an image and measure checkerboard pixel/mm."""
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
    img_und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img_und, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    ppm = None
    if found:
        cv2.drawChessboardCorners(img_und, pattern_size, corners, found)
        p1, p2 = corners[0,0], corners[1,0]
        pixel_dist = np.linalg.norm(p1 - p2)
        ppm = pixel_dist / square_size
        cv2.putText(img_und, f"{ppm:.2f} px/mm",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    return img_und, ppm, found

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort images or capture live frame and measure checkerboard.')
    parser.add_argument('calibration', help='calibration.yaml file')
    parser.add_argument('--input_mask', default=None, help='input image mask (e.g., ./pictures/*.png)')
    parser.add_argument('--out', default='./measured', help='output directory')
    parser.add_argument('--pattern', type=int, nargs=2, default=[9, 6], help='Chessboard pattern size (cols rows)')
    parser.add_argument('--live', action='store_true', help='Capture live frame from Pi Camera')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    K, dist, square_size = load_calibration(args.calibration)
    pattern_size = tuple(args.pattern)

    if args.live:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (1280, 720)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        print("Live preview started. Press 'c' to capture frame, 'q' to quit.")

        while True:
            frame = picam2.capture_array()
            img_und, ppm, found = undistort_and_measure(frame, K, dist, square_size, pattern_size)
            display = cv2.resize(img_und, (960, 540))
            cv2.imshow("Undistorted + Measurement", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if found:
                    out_path = os.path.join(args.out, f"live_capture.png")
                    cv2.imwrite(out_path, img_und)
                    print(f"Captured checkerboard → {ppm:.2f} px/mm")
                    print("Saved captured frame to", out_path)
                else:
                    print("⚠️ Checkerboard not detected, not captured.")
            elif key == ord('q'):
                print("Quitting live preview...")
                break

        picam2.close()
        cv2.destroyAllWindows()

    else:
        # Batch process saved images
        if args.input_mask is None:
            print("No input_mask provided for batch processing. Exiting.")
            exit(1)
        for fn in glob(args.input_mask):
            print(f'Processing {fn}...')
            img = cv2.imread(fn)
            if img is None:
                print("Failed to load " + fn)
                continue

            img_und, ppm, found = undistort_and_measure(img, K, dist, square_size, pattern_size)
            if found:
                name, ext = os.path.splitext(os.path.basename(fn))
                out_path = os.path.join(args.out, name + '_measure' + ext)
                cv2.imwrite(out_path, img_und)
                print(f"Saved: {out_path} → Checkerboard → {ppm:.2f} px/mm")
            else:
                print(f"⚠️ Checkerboard not detected in {fn}")
