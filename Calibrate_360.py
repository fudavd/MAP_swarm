import os.path

import numpy as np
import cv2
from utils import Calibration, Utils, Transform
from utils.Calibration import load_calibration_data
from utils.DefaultSettings import aruco_board

if __name__ == "__main__":
    cv2.namedWindow("360 cam")
    cam = cv2.VideoCapture(4)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)

    # %% Polar calibration
    ret, frame = cam.read()
    polar_calibrate = Calibration.FisheyeCalibration(directory='./calibration')
    # polar_calibrate.auto_calibrate(frame)
    polar_param_file = os.path.join('./calibration', 'cameraParameters_Fisheye.npy')
    if not os.path.isfile(polar_param_file):
        while True:
            ret, frame = cam.read()
            if ret:
                frame = polar_calibrate.capture_image(img=frame)
                cv2.imshow('360 cam', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
        polar_calibrate.calibrate('./calibration')
    # %% Polar transform
    image_size, polar_center, r_px_length = load_calibration_data(polar_param_file)
    polar_transform = Transform.PolarTransform(image_size, polar_center, r_px_length)

    # %% ARUCO calibration
    (board_w, board_h, square_s, marker_s, aruco_dict) = aruco_board()
    aruco_calibrate = Calibration.ArucoCalibration(board_w, board_h, square_s, marker_s, aruco_dict,
                                                   directory='./calibration')

    aruco_param_file = os.path.join('./calibration', 'cameraParameters_aruco.xml')
    if not os.path.isfile(aruco_param_file):
        while True:
            ret, frame = cam.read()
            if ret:
                frame = polar_transform.transform(img=frame)
                frame = aruco_calibrate.capture_image(img=frame)
                cv2.imshow('360 cam', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
        aruco_calibrate.get_image_list('./calibration')
        aruco_calibrate.calibrate('./calibration')

    camera_matrix, dist_coeff = load_calibration_data(aruco_param_file)
    print("FINISHED calibration")
