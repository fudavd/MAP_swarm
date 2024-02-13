import numpy as np
import cv2
import copy
from cv2 import aruco

from utils.Utils import euler_decom, rotationMatrixToEulerAngles


class ArucoFisheyePose:
    def __init__(self, aruco_marker_length: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 rad_per_px_h: float, rad_per_px_v: float,
                 dictionary: cv2.aruco_Dictionary, parameters: cv2.aruco_DetectorParameters):
        """
        Fish eye relative position/heading sensor using Aruco
        :param aruco_marker_length: Length of the marker [m]
        :param camera_matrix: Camera_matrix from calibration
        :param dist_coeffs: Distance coefficients from calibration
        :param rad_per_px_h: Horizontal angular scaling
        :param rad_per_px_v: Vertical angular scaling
        :param dictionary: Aruco marker dictionary
        :param parameters: Aruco detection parameters
        """
        self.dictionary = dictionary
        self.parameters = parameters
        self.aruco_marker_length = aruco_marker_length
        self.camera_matrix = camera_matrix
        self.dist_coeffs = np.zeros_like(dist_coeffs)
        self.rad_per_px_h = rad_per_px_h
        self.rad_per_px_v = rad_per_px_v

    def aruco_pose_rel(self, img, fisheye_t):
        # detects aruco aruco_tags and returns list of ids and coords of centre
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dictionary,
                                                              parameters=self.parameters)
        if ids is not None:
            # corners_undist = copy.deepcopy(corners)
            spherical_coord = np.zeros((len(ids), 3))
            corners_undist = []
            for i in range(len(ids)):
                # aruco.drawAxis(img, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.aruco_marker_length)
                aruco.drawDetectedMarkers(img, corners)

                # corners_array = corners[i][0]
                # angle_h = corners_array[:, 0] * self.rad_per_px_h
                # angle_v = 0.5 * np.pi - corners_array[:, 1] * self.rad_per_px_v
                # rel_phi = angle_h.mean()
                # rel_theta = angle_v.mean()
                # spherical_coord[i, :] = (rel_phi, rel_theta, 1)
                # x_centered = corners_array[:, 0] - np.mean(corners_array[:, 0])
                # corners_undist[i][0][:, 0] = x_centered + 1024 / 2
                coord = fisheye_t.spherical2tangent(corners[i])
                corners_undist.append(coord)
            cam_mat = copy.deepcopy(self.camera_matrix)
            # cam_mat[0, 1] = -self.camera_matrix[0, 1] * np.cos(rel_theta) / np.sin(rel_theta)

            rvec_undist, tvecs_undist, _ = aruco.estimatePoseSingleMarkers(corners_undist, self.aruco_marker_length,
                                                                           cam_mat,
                                                                           self.dist_coeffs)

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners,
                                                              self.aruco_marker_length,
                                                              cam_mat,
                                                              self.dist_coeffs)

            for i in range(len(ids)):
                aruco.drawAxis(img, cam_mat, self.dist_coeffs, rvecs[i], tvecs[i], self.aruco_marker_length)
                aruco.drawDetectedMarkers(img, corners)

            rel_angles = np.zeros((len(ids), 1))
            polar_coordinates = np.zeros((len(ids), 2))

            # Get Euler angles from rotation matrix
            for i in range(len(ids)):
                rel_r = np.linalg.norm(tvecs_undist[i])
                euler_ang = euler_decom(rvec_undist[i])
                polar_coordinates[i, :] = (spherical_coord[i, 0] / np.pi * 180, rel_r)
                rel_angles[i] = (np.mod(spherical_coord[i, 0] + euler_ang[0], 2 * np.pi) / np.pi * 180).round(4)

            return np.hstack((polar_coordinates, rel_angles))
        else:
            return None

    # def verbose(self, img):
    #     cv2.putText(img, f"Spherical coordinates ("
    #                      f"{(rel_phi / np.pi * 180).round(2)}, "
    #                      f"{(rel_theta / np.pi * 180).round(2)}, "
    #                      f"{(rel_r).round(2)})",
    #                 (25, 25), font, 0.7, (0, 0, 255), 2)
    #     cv2.putText(img, f"tvecs ({tvec.round(2)})",
    #                 (25, 45), font, 0.7, (0, 0, 255), 2)
    #     cv2.putText(img, f"rvecs {euler_ang.round(2)})",
    #                 (25, 65), font, 0.7, (0, 0, 255), 2)
    #     cv2.putText(img, f"rel_ori {np.mod(rel_phi + euler_ang.round(2)[0], 2 * np.pi).round(2)}",
    #                 (25, 85), font, 0.7, (0, 0, 255), 2)


class ArucoFisheyeCube:
    def __init__(self, cube: cv2.aruco_Board, aruco_marker_length: float,
                 camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 dictionary: cv2.aruco_Dictionary, parameters: cv2.aruco_DetectorParameters):
        """
        Fish eye relative position/heading sensor using Aruca
        :param cube: Cube parameters
        :param camera_matrix: Camera_matrix from calibration
        :param dist_coeffs: Distance coefficients from calibration
        :param dictionary: Aruco marker dictionary
        :param parameters: Aruco detection parameters
        """
        self.dictionary = dictionary
        self.parameters = parameters
        self.aruco_marker_length = aruco_marker_length
        self.cube = cube
        self.camera_matrix = camera_matrix
        self.dist_coeffs = np.zeros_like(dist_coeffs)
        self.rad_per_px_h = 2 * np.pi / 1024
        self.rad_per_px_v = 0.6111111 * np.pi / 768
        self.img_size = np.array([[[1024, 384]]], dtype=np.float32)

    def aruco_pose_rel(self, img, fisheye_t):
        # detects aruco aruco_tags and returns list of ids and coords of centre

        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray.copy(), (int(gray.shape[1]), int(gray.shape[0])))
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if_text = False
        rel_angles = np.zeros((1, 1))
        polar_coordinates = np.zeros((1, 2))
        if ids is not None:
            test = fisheye_t.spherical2tangent(np.array(corners).reshape((len(ids), 4, 2)))
            corners_undist = []
            corners_pose = []
            spherical_coord = np.ones((len(ids), 3))
            tangent_px_origin = np.mean(corners, axis=(1, 2), dtype=np.float32, keepdims=True)
            for i in range(len(ids)):
                spherical_coord[i, 0] = tangent_px_origin[i, 0, 0, 0] * self.rad_per_px_h
                spherical_coord[i, 1] = tangent_px_origin[i, 0, 0, 1] * self.rad_per_px_v
                coord_t = fisheye_t.spherical2tangent(corners[i]) + tangent_px_origin[i]
                corners_undist.append(coord_t)
                corners_pose.append(test[i] + self.img_size / 2)
            cam_mat = copy.deepcopy(self.camera_matrix)
            _, rvec, tvec = aruco.estimatePoseBoard(corners_undist, ids, self.cube, cam_mat, self.dist_coeffs, None,
                                                    None)
            # _, _, _ = aruco.estimatePoseBoard(corners_pose, ids, self.cube, cam_mat, self.dist_coeffs, None,
            #                                         None)
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Get Euler angles from rotation matrix
            euler_angles = rotationMatrixToEulerAngles(R)
            # print("Euler angles: ", np.rad2deg(euler_angles))

            img = cv2.drawFrameAxes(img, cam_mat, self.dist_coeffs, rvec, tvec, self.aruco_marker_length)
            img = aruco.drawDetectedMarkers(img, corners_undist, ids)
            if_text = not if_text

            rel_angles = np.zeros((len(ids), 1))
            polar_coordinates = np.zeros((len(ids), 2))
            for i in range(len(ids)):
                rel_r = np.linalg.norm(tvec)
                euler_ang = euler_decom(rvec)
                polar_coordinates[i, :] = (spherical_coord[i, 0] / np.pi * 180, rel_r)
                rel_angles[i] = (np.mod(spherical_coord[i, 0] + euler_ang[0], 2 * np.pi) / np.pi * 180).round(4)

        if (if_text):
            roll = np.rad2deg(euler_angles[0])
            pitch = np.rad2deg(euler_angles[1])
            yaw = np.rad2deg(euler_angles[2])
            distance = np.linalg.norm(tvec)*0.611111
            # Define the text to be printed
            text = 'Euclidian Distance: ' + str(distance.round(2)) + 'm' + ' Roll: ' + str(
                roll.round(2)) + ' Pitch: ' + str(
                pitch.round(2)) + ' Yaw: ' + str(yaw.round(2))

            # Set the font scale, thickness, and color
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (0, 0, 255)

            # Get the size of the text
            size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Calculate the position of the text
            x = int((img.shape[1] - size[0]) / 2)
            y = int((img.shape[0] + size[1]) / 2)

            cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
        return img, np.hstack((polar_coordinates, rel_angles))
