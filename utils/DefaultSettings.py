from cv2 import aruco
import numpy as np


def aruco_board():
    aruco_dict = aruco_dictionary()
    board_width = 7
    board_height = 5
    square_size = 0.042
    marker_size = 0.021
    aruco_marker_length = 0.05
    return board_width, board_height, square_size, marker_size, aruco_dict

def aruco_dictionary():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    return aruco_dict

def aruco_detector_parameters():
    aruco_parameters = aruco.DetectorParameters_create()
    aruco_parameters.adaptiveThreshWinSizeMin = 3
    aruco_parameters.adaptiveThreshWinSizeMax = 60
    return aruco_parameters

def aruco_cube():
    aruco_dict = aruco_dictionary()
    cube_corners = [
        np.array([[0.0075, 0.0, 0.0575], [0.0575, 0.0, 0.0575], [0.0575, 0.0, 0.0075], [0.0075, 0.0, 0.0075]],
                 dtype=np.float32),
        np.array([[0.065, 0.0075, 0.0575], [0.065, 0.0575, 0.0575], [0.065, 0.0575, 0.0075], [0.065, 0.0075, 0.0075]],
                 dtype=np.float32),
        np.array([[0.0575, 0.065, 0.0575], [0.0075, 0.065, 0.0575], [0.0075, 0.065, 0.0075], [0.0575, 0.065, 0.0075]],
                 dtype=np.float32),
        np.array([[0.0, 0.0575, 0.0575], [0.0, 0.0075, 0.0575], [0.0, 0.0075, 0.0075], [0.0, 0.0575, 0.0075]],
                 dtype=np.float32),
        np.array([[0.0075, 0.0575, 0.065], [0.0575, 0.0575, 0.065], [0.0575, 0.0075, 0.065], [0.0075, 0.0075, 0.065]],
                 dtype=np.float32),
        np.array([[0.0575, 0.0075, 0.0], [0.0575, 0.0575, 0.0], [0.0075, 0.0575, 0.0], [0.0075, 0.0075, 0.0]],
                 dtype=np.float32)
        ]
    cube_ids = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    cube_board = aruco.Board_create(cube_corners, aruco_dict, cube_ids)
    return cube_board