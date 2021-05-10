import cv2 as cv
from utils import *
from Odometry import *


class Frame:

    def __init__(self, img, frame_nbr):
        self.frame_nbr = frame_nbr

        # Extract features
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.kp, self.des = orb_extractor(img_grey)

    def nbr_features(self):
        return len(self.kp)
