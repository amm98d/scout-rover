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

        frameA, depthA = self._getFrame()
        frameB, depthB = self._getFrame()
        images = [frameA, frameB]
        depths = [depthA, depthB]

        self.slamAlgorithm = SLAM(frameA, depthA, depthFactor, camera_matrix, dist_coff)
        i = 2
        while True:
            # SLAMMING
            if i > 1:
                self.slamAlgorithm.process(images, depths, i)
            i += 1

            # Update Measurements
            frameA = np.copy(frameB)
            depthA = np.copy(depthB)
            frameB, depthB = self._getFrame()
            if np.isscalar(frameB) or i > 500:
                break
            images = [frameA, frameB]
            depths = [depthA, depthB]

        print('Mapping Completed')
