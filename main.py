import os.path
from typing import Dict

import numpy as np
import cv2
from utils import Transform
from utils.Calibration import load_calibration_data
from utils.DefaultSettings import aruco_dictionary, aruco_detector_parameters
from utils.Sensor import ArucoFisheyePose

if __name__ == "__main__":
    cv2.namedWindow("360 cam")
    cam = cv2.VideoCapture(4)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)

    # %% Get calibrated parameters
    camera_matrix, dist_coeff = load_calibration_data('calibration/cameraParameters_aruco.xml')
    image_size, polar_center, r_px_length = load_calibration_data('./calibration/cameraParameters_Fisheye.npy')
    polar_transform = Transform.PolarTransform(image_size, polar_center, r_px_length)

    # %% Get perameters
    ret, frame = cam.read()
    width, height = frame.shape[:2]
    rad_per_px_h = 2 * np.pi / width
    rad_per_px_v = 0.55 * np.pi / height
    aruco_marker_length = 0.05
    aruco_dictionary = aruco_dictionary()
    aruco_parameters = aruco_detector_parameters()

    pose_est = ArucoFisheyePose(aruco_marker_length, camera_matrix, dist_coeff,
                                rad_per_px_h, rad_per_px_v,
                                aruco_dictionary,
                                aruco_parameters
                                )

    while True:
        ret, frame = cam.read()
        if ret:
            frame = polar_transform.transform(img=frame)
            poses = pose_est.aruco_pose_rel(img=frame)
            if poses is not None:
                print(poses[0, :])
            cv2.imshow('360 cam', frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            cv2.destroyAllWindows()
            break

    print("FINISHED calibration")
