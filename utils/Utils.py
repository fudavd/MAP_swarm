import os

import cv2
import numpy as np


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    return file_list

def euler_decom(rvec):
    rmat = cv2.Rodrigues(rvec)[0]
    proj = np.hstack((rmat, np.array([[1, 1, 1]]).T))
    euler_angles_radians = -cv2.decomposeProjectionMatrix(proj)[6] / 180 * np.pi
    return euler_angles_radians


def find_center(img):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blurr = cv2.GaussianBlur(img_gray, (15, 15), 0)
    threshold = 20
    img_diff = np.logical_xor(img_gray > threshold, blurr > threshold)
    coord = np.argwhere(img_diff)
    x, y = (coord[:, 1], coord[:, 0])
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    x_center = x_m + uc
    y_center = y_m + vc

    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1 = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    radius_est = np.mean(Ri_1)
    residu_1 = int(np.sum((Ri_1 - radius_est) ** 2))
    print(f"Center: {int(x_center), int(y_center)}, radius: {int(radius_est)}\n"
          f"Error sum: {residu_1} | STD: {Ri_1.std()}, N={len(x)}")
    return (int(x_center), int(y_center)), int(radius_est)

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
