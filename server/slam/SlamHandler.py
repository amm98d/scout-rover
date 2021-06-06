import os
import numpy as np
import cv2 as cv
import sys

# internal modules
# sys.path.append("./")
from SLAM import *

np.random.seed(1)

class SlamHandler:

    def __init__(self, getFrameFunc):
        self._getFrame = getFrameFunc
        self.DATASET = 0
        self.NUM_FRAMES = -1

        self.metadata = {
            0: {
                'directory': os.path.join('datasets', 'carla'),
                'depth': True,
                'associate': False,
                'depth_factor': 1,
                'camera_matrix': [[640.0, 0, 640.0], [0, 480.0, 480.0], [0, 0, 1.0]],
                'dist_coff': None,
            },
            1: {
                'directory': os.path.join('datasets', 'fr1_xyz'),
                'depth': True,
                'associate': True,
                'depth_factor': 5000,
                'camera_matrix': [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]],
                'dist_coff': None, #[0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
            },
            2: {
                'directory': os.path.join('datasets', 'fr1_rpy'),
                'depth': True,
                'associate': True,
                'depth_factor': 5000,
                'camera_matrix': [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]],
                'dist_coff': None, #[0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
            },
            3: {
                'directory': os.path.join('datasets', 'fr2_pslam'),
                'depth': True,
                'associate': True,
                'depth_factor': 5000,
                'camera_matrix': [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]],
                'dist_coff': None, #[0.2312, -0.7849, -0.0033, -0.0001, 0.9172],
            },
            4: {
                'directory': os.path.join('datasets', 'trajectory220'),
                'depth': False,
                'associate': False,
                'depth_factor': 0,
                'camera_matrix': [[827.0, 0, 638.0], [0, 826.0, 347.0], [0, 0, 1.0]],
                'dist_coff': None,
            },
        }

    # def _createFrameGenerator(self):
    #     directory = self.metadata[self.DATASET]['directory']
    #     hasDepth = self.metadata[self.DATASET]['directory']
    #     needsAssociation = self.metadata[self.DATASET]['associate']

    #     rgbDir = os.path.join(directory, 'rgb')
    #     depthDir = os.path.join(directory, 'depth')

    #     if needsAssociation:
    #         associationFile = os.path.join(directory, 'associated.txt')
    #         with open(associationFile, 'r') as inFile:
    #             fileData = inFile.readlines()
    #         for line in fileData:
    #             line = line.split()
    #             rgb = line[1].split('/')[-1]
    #             depth = line[3].split('/')[-1]
    #             rgbFile = os.path.join(rgbDir, rgb)
    #             depthFile = os.path.join(depthDir, depth)
    #             img = cv.imread(rgbFile, cv.IMREAD_UNCHANGED)
    #             depth = cv.imread(depthFile, cv.IMREAD_UNCHANGED)
    #             yield img, depth
    #     else:
    #         rgbFileNames = [os.path.join(rgbDir, fileName)
    #                         for fileName in os.listdir(rgbDir)]
    #         depthFileNames = [None for _ in range(len(rgbFileNames))]
    #         if hasDepth:
    #             depthFileNames = [
    #                 os.path.join(depthDir, fileName) for fileName in os.listdir(depthDir)
    #             ]

    #         for rgbFile, depthFile in zip(rgbFileNames, depthFileNames):
    #             img = cv.imread(rgbFile, cv.IMREAD_UNCHANGED)
    #             if hasDepth:
    #                 depth = np.loadtxt(depthFile, delimiter=",",
    #                                 dtype=np.float64) * 1000.0
    #                 # depth = cv.imread(depthFile, cv.IMREAD_UNCHANGED)
    #                 yield img, depth
    #             else:
    #                 yield img, -1

    # def _saveMap(self):
    #     filePath = os.path.join(os.environ["USERPROFILE"], "Desktop", "Map.png")
    #     print()
    #     print(f'Saving Map at {filePath}')
    #     if cv.imwrite(filePath, self.slamAlgorithm.getMap()):
    #         print(f'Map saved')
    #     else:
    #         print('Map could not be saved')

    # def _getFrame(self, FRAME_GENERATOR):
    #     for i in FRAME_GENERATOR:
    #         return i
    #     return -1, -1

    def slamHome(self):
        # self.FRAME_GENERATOR = self._createFrameGenerator()

        # GLOBAL VARIABLES
        # poseFig, poseAxis = plt.subplots()

        # depthFactor = self.metadata[self.DATASET]['depth_factor']
        # camera_matrix = self.metadata[self.DATASET]['camera_matrix']
        # dist_coff = self.metadata[self.DATASET]['dist_coff']
        depthFactor = 1000
        camera_matrix = [[561.93206787, 0, 323.96944442], [ 0, 537.88018799, 249.35236366], [0, 0, 1]]
        dist_coff = [ 3.64965254e-01 , -2.02943943e+00 , -1.46113154e-03 , 9.97005541e-03 , 5.04006892e+00]
        frameA, depthA = self._getFrame()
        self.slamAlgorithm = SLAM(depthFactor, camera_matrix, dist_coff)

        # frameA, depthA = self._getFrame(self.FRAME_GENERATOR)
        # frameB, depthB = self._getFrame(self.FRAME_GENERATOR)
        frameA, depthA = self._getFrame()
        frameB, depthB = self._getFrame()
        images = [frameA, frameB]
        depths = [depthA, depthB]

        # print("=====================================================================")
        # print(len(frameA))
        # print("=====================================================================")

        i = 2
        while True:
            # SLAMMING
            if i > 1:
                self.slamAlgorithm.process(images, depths, i)
                # cv.waitKey(2000)
            i += 1

            # Update Measurements
            frameA = np.copy(frameB)
            depthA = np.copy(depthB)
            frameB, depthB = self._getFrame()
            # if np.isscalar(frameB) or i > 500:
            #     break
            images = [frameA, frameB]
            depths = [depthA, depthB]

        print('Mapping Completed')

        # trajectory = self.slamAlgorithm.get_trajectory()
        # visualize_data(visualize_trajectory, True, True, "3D", trajectory)

        # poses = self.slamAlgorithm.get_robot_poses()
        # visualize_data(plot_robot_poses, True, True, f"poses", poses)
