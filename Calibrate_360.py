import os.path

import numpy as np
import cv2
from utils import Calibration, Utils, Transform
from utils.Calibration import load_calibration_data
from utils.DefaultSettings import aruco_board
from utils.Transform import tan_y_map

if __name__ == "__main__":
    cv2.namedWindow("360 cam")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)

    # %% fisheye calibration
    ret, frame = cam.read()
    fisheye_calibrate = Calibration.FisheyeCalibration(directory='./calibration')
    # fisheye_calibrate.auto_calibrate(frame)
    fisheye_param_file = os.path.join('./calibration', 'cameraParameters_Fisheye.npy')
    if not os.path.isfile(fisheye_param_file):
        while True:
            ret, frame = cam.read()
            if ret:
                frame = fisheye_calibrate.capture_image(img=frame)
                cv2.imshow('360 cam', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
        fisheye_calibrate.calibrate('./calibration')
    # %% fisheye transform
    image_size, polar_center, r_px_length = load_calibration_data(fisheye_param_file)
    fisheye_transform = Transform.FisheyeTransform(image_size, polar_center, r_px_length)

    # %% ARUCO calibration
    (board_w, board_h, square_s, marker_s, aruco_dict) = aruco_board()
    aruco_calibrate = Calibration.ArucoCalibration(board_w, board_h, square_s, marker_s, aruco_dict,
                                                   directory='./calibration')

    aruco_param_file = os.path.join('./calibration', 'cameraParameters_aruco.xml')
    if not os.path.isfile(aruco_param_file):
        while True:
            ret, frame = cam.read()
            if ret:
                frame = fisheye_transform.transform(img=frame)
                frame = aruco_calibrate.capture_image(img=frame)
                cv2.imshow('360 cam', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
        aruco_calibrate.get_image_list('./calibration')
        aruco_calibrate.calibrate(fisheye_transform, './calibration')

    camera_matrix, dist_coeff = load_calibration_data(aruco_param_file)
    print("FINISHED calibration")
