#!/usr/bin/env python3
import numpy as np
import cv2
import os
import argparse
import pickle
import yaml
from picamera2 import Picamera2, Preview

def main():
    parser = argparse.ArgumentParser(description='Calibrate camera using Picamera2 with live preview and manual frame capture.')
    parser.add_argument('--frames', type=int, default=20, help='Number of chessboard frames to capture')
    parser.add_argument('--pattern', type=int, nargs=2, default=[9, 6], help='Chessboard pattern size (cols rows)')
    parser.add_argument('--mm', type=float, default=25.0, help='Size of each square in mm')
    parser.add_argument('--debug_dir', default='./pictures', help='Directory to save captured frames')
    parser.add_argument('--output_dir', default='./calibrationFiles', help='Directory to save calibration results')
    args = parser.parse_args()

    # Create directories if missing
    os.makedirs(args.debug_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    pattern_size = tuple(args.pattern)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= args.mm

    obj_points = []
    img_points = []

    print("Starting capture with live preview...")
    print("Press Enter when you are ready to capture the next chessboard image.\n")

    frame_idx = 0
    while frame_idx < args.frames:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        # Draw corners if found
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
            cv2.putText(display, f"Chessboard detected! Frame {frame_idx+1}/{args.frames}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(display, "Chessboard NOT detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('Live Preview', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('\r') or key == 13:
            # Enter pressed: try to capture
            if found:
                obj_points.append(pattern_points.reshape(1, -1, 3))
                img_points.append(corners.reshape(1, -1, 2))
                cv2.imwrite(os.path.join(args.debug_dir, f"{frame_idx:04d}.png"), display)
                print(f"Captured chessboard image {frame_idx+1}/{args.frames}")
                frame_idx += 1
            else:
                print("Chessboard not detected. Adjust position and try again.")

    cv2.destroyAllWindows()

    if len(obj_points) == 0:
        print("No frames captured. Exiting.")
        picam2.close()
        return

    print("\nPerforming camera calibration...")
    h, w = gray.shape[:2]
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w,h), None, None)

    print("RMS error:", rms)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coefs.ravel())

    # Save calibration in .txt for compatibility
    np.savetxt(os.path.join(args.output_dir, "cameraMatrix.txt"), camera_matrix, delimiter=',')
    np.savetxt(os.path.join(args.output_dir, "cameraDistortion.txt"), dist_coefs, delimiter=',')

    # Save calibration in YAML for undistort.py
    calib_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
        "square_size_mm": args.mm
    }
    with open(os.path.join(args.output_dir, "calibration.yaml"), "w") as f:
        yaml.dump(calib_data, f)

    print(f"Calibration files saved in {args.output_dir}")
    picam2.close()

if __name__ == "__main__":
    main()
