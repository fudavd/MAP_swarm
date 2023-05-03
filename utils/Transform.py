from typing import Tuple

import numpy as np
import cv2
from cv2 import aruco

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

    def transform(self, img):
        img = cv2.warpPolar(img,
                            dsize=self.image_size,
                            center=self.polar_center,
                            maxRadius=self.r_px_length, flags=cv2.INTER_LANCZOS4)[:, int(self.image_size[0] / 2):]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

