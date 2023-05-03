import os.path
import time
from typing import Dict

import numpy as np
import cv2
from cv2 import aruco

import utils.Utils
from utils.Utils import search_file_list


class ArucoCalibration:
    def __init__(self, board_width: int, board_height: int, square_size: float, marker_size: float,
                 aruco_dict: cv2.aruco_Dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
                 directory: str = 'calibration'):
        """
        Calibration class to remove camera distortion using an Aruco board.
        :param directory:
        :param board_width: Number of Aruco tags per column
        :param board_height: Number of Aruco tags per row
        :param square_size: Size of Aruco tag border [m]
        :param marker_size: Size of Aruco tag [m]
        :param aruco_dict: Aruco board parameters
        :param directory: Directory path to save calibration parameters
        """
        self.images = []
        self.img_points = []
        self.dist_coeffs = None
        self.camera_matrix = None

        self.board_size = (board_width, board_height)
        self.square_size = square_size
        self.marker_size = marker_size
        self.dictionary = aruco_dict
        self.board = aruco.CharucoBoard_create(board_width, board_height, square_size, marker_size, aruco_dict)
        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_parameters.adaptiveThreshWinSizeMin = 3
        self.aruco_parameters.adaptiveThreshWinSizeMax = 60
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        self.directory = directory
        try:
            image_dir = os.path.join(directory, 'images')
            os.makedirs(image_dir, exist_ok=True)
        except:
            print("ERROR: images cannot be saved!")

        self.file_storage = cv2.FileStorage()

    def capture_image(self, img, directory: str = None):
        capture_time = time.time()
        if directory is None:
            directory = self.directory
        img2 = np.asarray(img, dtype="int32")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find aruco markers in the query image
        corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=self.dictionary, parameters=self.aruco_parameters)

        if ids is not None:
            # Outline the aruco markers found in our query image
            img = aruco.drawDetectedMarkers(
                image=img,
                corners=corners)

            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=self.board)

            if response > 5:
                # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                img = aruco.drawDetectedCornersCharuco(
                    image=img,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids)

                save_file = os.path.join(directory, 'images', f'{capture_time}_aruco.jpg')
                print(f'Saving: {save_file}')
                cv2.imwrite(save_file, img2)
                self.images.append(save_file)
        return img

    def get_image_list(self, directory: str = None):
        if directory is None:
            directory = self.directory
        self.images = search_file_list(directory, '_aruco.jpg')

    def calibrate(self, directory: str = None):
        if directory is None:
            directory = self.directory
        images = self.images
        assert len(images) > 0, "Calibration was unsuccessful. No images of charucoboards were found. Add images of " \
                                "charucoboards and use or alter the naming conventions used in this file."
        img = cv2.imread(images[0])
        image_size = img.shape[:-1]
        ids_all = []
        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = aruco.detectMarkers(
                image=gray,
                dictionary=self.dictionary,
                parameters=self.aruco_parameters)

            img = aruco.drawDetectedMarkers(
                image=img,
                corners=corners)

            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=self.board)

            if response > 5:
                print(image)
                self.img_points.append(charuco_corners)
                ids_all.append(charuco_ids)

                img = aruco.drawDetectedCornersCharuco(
                    image=img,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids)

                # proportion = max(img.shape) / 1000.0
                # img = cv2.resize(img, (int(img.shape[1] / proportion), int(img.shape[0] / proportion)))
                cv2.imshow('Charuco board', img)
                cv2.waitKey(0)
            else:
                print("Not able to detect a charuco board in image: {}".format(image))

        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        ret, camera_matrix, distortion_coefficients, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=self.img_points,
            charucoIds=ids_all,
            board=self.board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None)

        # Print matrix and distortion coefficient to the console
        print(camera_matrix)
        print(distortion_coefficients)

        self.file_storage.open(os.path.join(directory, "cameraParameters_aruco.xml"), cv2.FILE_STORAGE_WRITE)
        self.file_storage.write("cameraMatrix", camera_matrix)
        self.file_storage.write("dist_coeffs", distortion_coefficients)

        self.file_storage.release()
        return ret, camera_matrix, distortion_coefficients

    def load_calibration_data(self, directory: str = None):
        if directory is None:
            directory = self.directory
        self.file_storage.open(os.path.join(directory, "cameraParameters_aruco.xml"), cv2.FileStorage_READ)
        self.camera_matrix = self.file_storage.getNode("cameraMatrix").mat()
        self.dist_coeffs = self.file_storage.getNode("dist_coeffs").mat()
        return self.camera_matrix, self.dist_coeffs


class CircleGridCalibration:
    def __init__(self, n_cols: int, n_rows: int, directory: str = 'calibration'):
        """
        Calibration class to remove camera distortion using circular grids.
        :param n_cols: Number of circles per column
        :param n_rows: Number of circles per row
        :param directory: Directory path to save calibration parameters
        """
        self.images = []
        self.obj_points = []
        self.img_points = []
        self.dist_coeffs = None
        self.camera_matrix = None

        self.grid_size = (n_cols, n_rows)
        self.circ_grid = cv2.CALIB_CB_SYMMETRIC_GRID
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        self.directory = directory

        self.file_storage = cv2.FileStorage()

    def capture_image(self, img, directory: str = None):
        capture_time = time.time()
        if directory is None:
            directory = self.directory
        img2 = np.asarray(img, dtype="int32")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findCirclesGrid(gray, self.grid_size, self.circ_grid)
        if ret is True:
            cv2.drawChessboardCorners(img, self.grid_size, corners, ret)

            save_file = os.path.join(directory, f'{capture_time}_CircGrid.jpg')
            print(f'Saving: {save_file}')
            cv2.imwrite(save_file, img2)
            self.images.append(save_file)
        return img

    def get_image_list(self, directory: str = None):
        if directory is None:
            directory = self.directory
        self.images = search_file_list(directory, '_CircGrid.jpg')

    def calibrate(self, directory: str = None):
        if directory is None:
            directory = self.directory
        images = self.images
        assert len(images) < 1, "Calibration was unsuccessful. No images of charucoboards were found. Add images of " \
                                "charucoboards and use or alter the naming conventions used in this file."
        img = cv2.imread(images[0])
        image_size = img.shape[:-1]
        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findCirclesGrid(gray, self.grid_size, None)

            if ret:
                print(image)
                self.obj_points.append(self.board.objPoints)
                self.img_points.append(corners)

                cv2.drawChessboardCorners(img, self.grid_size, corners, ret)

                proportion = max(img.shape) / 1000.0
                img = cv2.resize(img, (int(img.shape[1] / proportion), int(img.shape[0] / proportion)))
                cv2.imshow('Charuco board', img)
                cv2.waitKey(0)
            else:
                print("Not able to detect a charuco board in image: {}".format(image))

        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            image_size, None, None)

        # Print matrix and distortion coefficient to the console
        print(camera_matrix)
        print(distortion_coefficients)

        self.file_storage.open(os.path.join(directory, "cameraParameters_CircleGrid.xml"), cv2.FILE_STORAGE_WRITE)
        self.file_storage.write("cameraMatrix", camera_matrix)
        self.file_storage.write("dist_coeffs", distortion_coefficients)

        self.file_storage.release()
        return ret, camera_matrix, distortion_coefficients

    def load_calibration_data(self, directory: str = None):
        if directory is None:
            directory = self.directory
        self.file_storage.open(os.path.join(directory, "cameraParameters_CircleGrid.xml"), cv2.FileStorage_READ)
        self.camera_matrix = self.file_storage.getNode("cameraMatrix").mat()
        self.dist_coeffs = self.file_storage.getNode("dist_coeffs").mat()
        return self.camera_matrix, self.dist_coeffs


class FisheyeCalibration:
    def __init__(self, directory: str = 'calibration'):
        """
        Calibration class to unwrap fisheye camera distortion using polar coordinate transform.
        :param directory: Directory path to save calibration parameters
        """
        self.fisheye_dictionary = None
        self.r_px_length = None
        self.image_size = None
        self.center_x = None
        self.center_y = None
        self.polar_center = None

        self.directory = directory
        self.display_polar_circle = True
        self.unwarp = False

    def capture_image(self, img):
        if self.image_size is None:
            self.image_size = img.shape[:-1]
            print(f'Set image size to {self.image_size[1]}x{self.image_size[0]}')
        if self.center_x is None or self.center_x is None:
            self.center_y = int(self.image_size[0] / 2)
            self.center_x = int(self.image_size[1] / 2)
            self.r_px_length = min(self.center_y, self.center_x)
            print(f'Set polar coordinates to: ({self.center_x}, {self.center_y}), {self.r_px_length}')

        if self.display_polar_circle:
            img = cv2.circle(img, (self.center_x, self.center_y), self.r_px_length,
                             (255, 0, 0), thickness=10, lineType=8, shift=0)
        if self.unwarp:
            img = cv2.warpPolar(img,
                                dsize=self.image_size,
                                center=(self.center_x, self.center_y),
                                maxRadius=self.r_px_length, flags=cv2.INTER_LANCZOS4)[:, int(self.image_size[0] / 2):]
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        self.waitkey(1)
        return img

    def waitkey(self, time_out=1):
        k = cv2.waitKey(time_out)
        if k % 256 == 32:  # spacebar
            self.unwarp = not self.unwarp
        elif k % 256 == 119:  # w-a-s-d move circle
            self.center_y -= 1
        elif k % 256 == 97:
            self.center_x -= 1
        elif k % 256 == 115:
            self.center_y += 1
        elif k % 256 == 100:
            self.center_x += 1
        elif k % 256 == 99:  # c
            self.display_polar_circle = not self.display_polar_circle

    def auto_calibrate(self, img):
        polar_center, r_px_length = utils.Utils.find_center(img)
        (self.center_x, self.center_y) = polar_center
        self.polar_center = polar_center
        self.r_px_length = r_px_length

    def calibrate(self, directory: str = None):
        if directory is None:
            directory = self.directory

        self.polar_center = (self.center_x, self.center_y)
        self.fisheye_dictionary = {
            'image_size': self.image_size,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'polar_center': self.polar_center,
            'r_px_length': self.r_px_length
        }
        np.save(os.path.join(directory, 'cameraParameters_Fisheye.npy'), self.fisheye_dictionary)
        return self.polar_center, self.r_px_length

    def load_calibration_data(self, directory: str = None):
        if directory is None:
            directory = self.directory
        self.fisheye_dictionary = np.load(os.path.join(directory, 'cameraParameters_Fisheye.npy'),
                                          allow_pickle=True).item()
        self.load_dictionary(self.fisheye_dictionary)

    def load_dictionary(self, dictionary: Dict):
        for item in dictionary.items():
            self.__setattr__(item[0], item[1])
            print(f'Set {item[0]}: {item[1]}')
        return self.image_size, self.polar_center, self.r_px_length


def load_calibration_data(file: str):
    """
    Load specific calibration parameters associated with <file>
    :param file: Name of the file
    :return: Calibration variables
    """
    suffix = file.split('.')[-1]
    if suffix == "xml":
        file_storage = cv2.FileStorage()
        file_storage.open(file, cv2.FileStorage_READ)
        camera_matrix = file_storage.getNode("cameraMatrix").mat()
        dist_coeffs = file_storage.getNode("dist_coeffs").mat()
        return camera_matrix, dist_coeffs
    if suffix == "npy":
        np_array = np.load(file, allow_pickle=True)
        if "cameraParameters_Fisheye.npy" in file:
            fisheye_dict: Dict = np_array.item()
            return fisheye_dict['image_size'], fisheye_dict['polar_center'], fisheye_dict['r_px_length']

