from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2


class FisheyeTransform:
    def __init__(self, image_size: Tuple[int, int], polar_center: Tuple[int, int], r_px_length: int):
        """
        Transform fisheye image into mercator coordinates
        :param image_size: dimension of source image (height, width)
        :param polar_center: coordinate of polar center (center_height, center_width)
        :param r_px_length: length of the polar radius
        """
        self.image_size = image_size
        self.r_px_length = r_px_length
        self.polar_center = polar_center
        self.map = self.create_sperical_map()

    def create_sperical_map(self):
        size_y = int(self.image_size[0]/2)
        size_x = self.image_size[1]
        map_x = np.zeros((size_y, size_x), np.float32)
        map_y = np.zeros((size_y, size_x), np.float32)
        rad_per_px_h = 2 * np.pi / size_x
        for y in range(int(size_y)):
            phi = 2*np.arctan((size_y - y) / (1.2222*self.r_px_length))
            # phi = np.abs((size_y-y)/self.r_px_length)
            r_px = self.r_px_length * np.cos(phi)
            for x in range(size_x):
                theta = 2 * np.pi - x * rad_per_px_h
                YY = r_px * np.sin(theta) + self.polar_center[1]
                XX = r_px * np.cos(theta) + self.polar_center[0]
                map_x[y, x] = XX
                map_y[y, x] = YY
        map_x = np.hstack((map_x, map_x[:, :int(size_x/4)]))
        map_y = np.hstack((map_y, map_y[:, :int(size_x/4)]))
        return map_x, map_y

    def transform(self, img):
        img = cv2.remap(img, self.map[0], self.map[1], cv2.INTER_LANCZOS4)
        return img

    def spherical2tangent(self, coord):
        horz_px_coord = coord[:, :, 0]
        vert_px_coord = coord[:, :, 1] + self.image_size[0] / 2
        vert_ang = 0.611111 * np.pi - vert_px_coord / self.image_size[0] * 0.611111 * np.pi
        vert_tangent_origin = np.mean(vert_ang)

        horz_ang = horz_px_coord / self.image_size[1] * 2 * np.pi
        horz_ang = np.unwrap(horz_ang) + 2 * np.pi
        horz_tangent_origin = np.mean(horz_ang)
        r_ratio = np.cos(vert_ang)/2

        x_tangent_coord = np.tan((horz_ang - horz_tangent_origin) * r_ratio) * self.r_px_length
        y_tangent_coord = -np.tan((vert_ang - vert_tangent_origin)) * self.r_px_length
        # print(np.max(x_tangent_coord)-np.min(x_tangent_coord), np.max(y_tangent_coord)-np.min(y_tangent_coord))
        return np.array([x_tangent_coord.T, y_tangent_coord.T]).T

class PolarTransform:
    def __init__(self, image_size: Tuple[int, int], polar_center: Tuple[int, int], r_px_length: int):
        """
        Transform image into polar coordinates
        :param image_size: dimension of source image (height, width)
        :param polar_center: coordinate of polar center (center_height, center_width)
        :param r_px_length: length of the polar radius
        """
        self.image_size = image_size
        self.r_px_length = r_px_length
        self.polar_center = polar_center
        self.map = self.create_polar_map()

    def cv_polar_transform(self, img):
        img = cv2.warpPolar(img,
                            dsize=self.image_size,
                            center=self.polar_center,
                            maxRadius=self.r_px_length, flags=cv2.INTER_LANCZOS4 + cv2.WARP_POLAR_LINEAR)

        img = img[:, int(self.image_size[0] / 2):]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def create_polar_map(self):
        size_y = int(self.image_size[0] / 2)
        size_x = self.image_size[1]
        map_x = np.zeros((size_y, size_x), np.float32)
        map_y = np.zeros((size_y, size_x), np.float32)
        rad_per_px_h = 2 * np.pi / size_x
        for y in range(int(size_y)):
            relative_r = (y + self.image_size[0] / 2) / self.image_size[0]
            r_px = relative_r * self.r_px_length
            for x in range(size_x):
                theta = 2 * np.pi - x * rad_per_px_h
                YY = r_px * np.sin(theta) + self.polar_center[1]
                XX = r_px * np.cos(theta) + self.polar_center[0]
                map_x[y, x] = XX
                map_y[y, x] = YY
        return map_x, map_y

    def transform(self, img):
        img = cv2.remap(img, self.map[0], self.map[1], cv2.INTER_LANCZOS4)
        return img

    def spherical2tangent(self, coord):
        horz_px_coord = coord[:, 0, 0]
        vert_px_coord = coord[:, 0, 1]
        horz_ang = horz_px_coord / self.image_size[1] * 2 * np.pi
        horz_ang = (horz_ang + np.pi) % (2 * np.pi) - np.pi
        vert_ang = 0.5 * np.pi - vert_px_coord / self.image_size[0] * 0.55 * np.pi

        r_ratio = np.cos(vert_ang)

        horz_tangent_origin = np.mean(horz_ang)
        vert_tangent_origin = np.mean(horz_ang)

        x_tangent_coord = self.image_size[0] * (1 + np.tan(horz_ang - horz_tangent_origin) * r_ratio) / 2
        y_tangent_coord = self.image_size[0] * (1 + np.tan(vert_ang - vert_tangent_origin)) / 2
        return


def tan_y_map(img):
    # set up the x and y maps as float32
    img_size = img.shape[:2]
    map_x = np.zeros(img_size, np.float32)
    map_y = np.zeros(img_size, np.float32)
    rad_per_px_v = 0.55 * np.pi / 768
    for y in range(img_size[0]):
        # YY = 384-np.tan(0.55*np.pi-(y+384)*rad_per_px_v)*384
        # YY = int(384-np.tan(0.55*np.pi-(y+384)*rad_per_px_v)*384)
        # YY = -(np.arctan(-(y-384)/384)-0.55*np.pi)/rad_per_px_v-384
        # YY = 768-(0.55*np.pi-np.arctan(384-y))/rad_per_px_v
        YY = (np.arctan((y) / img_size[0]) / 0.25 * 0.55 / rad_per_px_v) * 0.5
        # print(YY)
        # YY = y
        for x in range(img_size[1]):
            XX = x
            map_x[y, x] = XX
            map_y[y, x] = YY

    img = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4)
    return img


# def  create_tan_y_map(img_size):
#     map_x = np.zeros(img_size, np.float32)
#     map_y = np.zeros(img_size, np.float32)
#     rad_per_px_v = 0.55 * np.pi / img_size[0]
#     for y in range(img_size[0]):
#         YY = (np.arctan((y)/img_size[0])/0.25*0.55/rad_per_px_v)*0.5
#         # print(YY)
#         # YY = y
#         for x in range(img_size[1]):
#             XX = x
#             map_x[y, x] = XX
#             map_y[y, x] = YY
#     return map_x, map_y
def create_tan_y_map(img_size):
    size_y = int(img_size[0] / 2)
    size_x = img_size[1]
    map_x = np.zeros((size_y, size_x), np.float32)
    map_y = np.zeros((size_y, size_x), np.float32)
    rad_per_px_v = 0.55 * np.pi / img_size[0]
    for y in range(size_y):
        # phi = (0.5*np.pi-y*rad_per_px_v)
        # YY = np.log(np.tan(0.5*phi+0.25*np.pi))
        YY = np.arctan(np.sinh((y) / 100)) / rad_per_px_v
        # YY = 5
        # print(YY)
        # YY = y
        for x in range(size_x):
            XX = x
            map_x[y, x] = XX
            map_y[y, x] = YY
    print(map_x.min(), map_y.min(),
          map_x.max(), map_y.max())
    return map_x, map_y
