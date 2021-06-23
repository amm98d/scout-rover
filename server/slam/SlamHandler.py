import os
import numpy as np
import cv2 as cv
import sys

# internal modules
from SLAM import *

class SlamHandler:

    def __init__(self, getFrameFunc):
        self._getFrame = getFrameFunc

    def slamHome(self):
        depthFactor = 1000
        camera_matrix = [[561.93206787, 0, 323.96944442], [ 0, 537.88018799, 249.35236366], [0, 0, 1]]
        dist_coff = [3.64965254e-01, -2.02943943e+00, -1.46113154e-03, 9.97005541e-03, 5.04006892e+00]

        img, depth = self._getFrame()
        self.slamAlgorithm = SLAM(img, depth, depthFactor, camera_matrix, dist_coff)
        i = 1
        while True:
            newImg, newDepth = self._getFrame()

            if np.isscalar(newImg):
                print("BREAKING")
                break

            self.slamAlgorithm.process([img, newImg], [depth, newDepth], i)
            i += 1

            img = newImg
            depth = newDepth