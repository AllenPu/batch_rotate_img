from os.path import join

import scipy.io as sp
import os
import numpy as np
from utils.datasets import Datasets
import os
from os import listdir
from math import *
import cv2
import numpy as np
from PIL import Image


#rotate VLCS data from 0 , 15, 30 ,45, 60
class rotate_VLSC(Datasets):
    def __init__(self, root_path, test_split=0.3):
        super(rotate_VLSC, self).__init__(root_path)

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def load_img(self, root,):
        array_of_img = []
        for filename in os.listdir(r"./" + root):
            img = cv2.imread(root + "/" + filename)
            array_of_img.append(img)
        return array_of_img

    def rotate_image(self, root, angle):
        array_of_img = self.load_img(root)
        for image in array_of_img:
            h, w = image.shape[:2]
            newW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
            newH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            M[0, 2] += (newW - w) / 2
            M[1, 2] += (newH - h) / 2
            return cv2.warpAffine(image, M, (newW, newH), borderValue=(255, 255, 255))

    def rotate_img(self, root, angle):
        array_of_img = []
        image_filenames = [x for x in listdir(root) if self.is_image_file(x)]
        for img in image_filenames:
            rot_angle = "ROTATE_" + angle
            img = img.transpose(Image.rot_angle)
            array_of_img.append(img)
        return array_of_img





