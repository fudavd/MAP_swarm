import itertools
import os

from cv2 import aruco
import cv2
import numpy as np


def aruco_board():
    aruco_dict = aruco_dictionary()
    board_width = 5
    board_height = 7
    square_size = 0.042
    marker_size = 0.021
    return board_width, board_height, square_size, marker_size, aruco_dict

def aruco_dictionary():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    return aruco_dict

def cube_dictionary():
    bit_list = list(itertools.product([0, 1], repeat=9))
    aruco_dict = aruco.custom_dictionary(0, 3, 1)
    aruco_dict.bytesList = np.empty(shape=(2**9, 2, 4), dtype=np.uint8)
    for ind, bit in enumerate(bit_list):
        mybits = np.array(bit, dtype=np.uint8).reshape((3,3))
        aruco_dict.bytesList[ind] = aruco.Dictionary_getByteListFromBits(mybits)

def aruco_detector_parameters():
    aruco_parameters = aruco.DetectorParameters_create()
    aruco_parameters.adaptiveThreshWinSizeMin = 3
    aruco_parameters.adaptiveThreshWinSizeMax = 60
    return aruco_parameters

def aruco_cube(id_list=None):
    if id_list is None:
        id_list = [0, 1, 2, 3, 4, 5]
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
    cube_ids = np.array([id_list], dtype=np.int32)
    cube_board = aruco.Board_create(cube_corners, aruco_dict, cube_ids)
    return cube_board

def create_cube_markers(aruco_dict, id_list, dir: str='./marker_dir/', verbose=False):
    # id_list = np.arange(6*5).reshape((5,6)).tolist()

    # Set the size of the output marker image
    marker_size = 200

    for ind, cube_ids in enumerate(id_list):
        cube_path = os.path.join(dir, str(ind+1))
        os.makedirs(cube_path, exist_ok=True)
        # Specify IDs for the six markers
        marker_ids = cube_ids

        # Draw and save each marker
        for marker_id in marker_ids:
            marker = aruco.drawMarker(aruco_dict, marker_id, marker_size)
            marker_image = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
            marker_image[:marker.shape[0], :marker.shape[1]] = marker

            # Save the marker image
            marker_filename = os.path.join(cube_path, f"marker_{marker_id}.png")
            print(marker_filename)
            cv2.imwrite(marker_filename, marker_image)

            if verbose:
                # Display the marker
                cv2.imshow(f'Marker {marker_id}', marker_image)
                cv2.waitKey(500)  # Wait for 500 milliseconds between markers
        cv2.destroyAllWindows()

