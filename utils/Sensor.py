import numpy as np
import cv2
import copy
from cv2 import aruco

from utils.Utils import euler_decom


class ArucoFisheyePose:
    def __init__(self, aruco_marker_length: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 rad_per_px_h: float, rad_per_px_v: float,
                 dictionary: cv2.aruco_Dictionary, parameters: cv2.aruco_DetectorParameters):
        """
        Fish eye relative position/heading sensor using Aruca
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
        self.dist_coeffs = dist_coeffs
        self.rad_per_px_h = rad_per_px_h
        self.rad_per_px_v = rad_per_px_v
    def aruco_pose_rel(self, img):
        # detects aruco aruco_tags and returns list of ids and coords of centre
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray.copy(), (int(gray.shape[1]), int(gray.shape[0])))
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dictionary,
                                                              parameters=self.parameters)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners,
                                                          self.aruco_marker_length,
                                                          self.camera_matrix,
                                                          self.dist_coeffs)

        if ids is not None:
            rel_angles = np.zeros((len(ids), 1))
            polar_coordinates = np.zeros((len(ids), 2))
            for i in range(len(ids)):
                aruco.drawAxis(img, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.aruco_marker_length)
                aruco.drawDetectedMarkers(img, corners)

                corners_array = corners[i][0]
                angle_h = corners_array[:, 0] * self.rad_per_px_h
                angle_v = 0.5 * np.pi - corners_array[:, 1] * self.rad_per_px_v
                rel_phi = angle_h.mean()
                rel_theta = angle_v.mean()

                corners_undist = copy.deepcopy(corners)
                corners_undist[i][0][:, 0] = np.tan(angle_h - rel_phi)
                corners_undist[i][0][:, 1] = -np.tan(angle_v - rel_theta)
                rvec_undist, tvecs_undist, _ = aruco.estimatePoseSingleMarkers(corners_undist, self.aruco_marker_length,
                                                                               self.camera_matrix, self.dist_coeffs)
                rel_r = np.linalg.norm(tvecs[i])
                euler_ang = euler_decom(rvecs[i], tvecs[i])
                polar_coordinates[i, :] = (rel_phi, rel_r)
                rel_angles[i] = np.mod(rel_phi + euler_ang.round(2)[0], 2 * np.pi)
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