# external modules
import socket
import os
from platform import platform
import numpy as np
import cv2 as cv
import urllib.request

# import fpstimer
import select
import sys

# import tty

# import termios
import time
import multiprocessing

# from multiprocessing import Process, Queue, cpu_count
import random
import time
import threading

# internal modules
import sys

sys.path.append("../common/")
sys.path.append("./slam/")
# from NetworkHandler import *
from SLAM import *


np.random.seed(1)

#####################
# READ FRAMES START
#####################
DATASET = 3
NUM_FRAMES = -1

metadata = {
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
        'directory': os.path.join('datasets', 'fr2_p360'),
        'depth': True,
        'associate': True,
        'depth_factor': 5000,
        'camera_matrix': [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]],
        'dist_coff': None, #[0.2312, -0.7849, -0.0033, -0.0001, 0.9172],
    },
    5: {
        'directory': os.path.join('datasets', 'trajectory220'),
        'depth': False,
        'associate': False,
        'depth_factor': 0,
        'camera_matrix': [[827.0, 0, 638.0], [0, 826.0, 347.0], [0, 0, 1.0]],
        'dist_coff': None,
    },
    6: {
        'directory': os.path.join('datasets', 'amm_1'),
        'depth': True,
        'associate': False,
        'depth_factor': 5000,
        'camera_matrix': [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]],
        'dist_coff': None,
    },
}


def createFrameGenerator():

    directory = metadata[DATASET]['directory']
    hasDepth = metadata[DATASET]['directory']
    needsAssociation = metadata[DATASET]['associate']

    rgbDir = os.path.join(directory, 'rgb')
    depthDir = os.path.join(directory, 'depth')

    if needsAssociation:
        associationFile = os.path.join(directory, 'associated.txt')
        with open(associationFile, 'r') as inFile:
            fileData = inFile.readlines()
        for line in fileData:
            line = line.split()
            rgb = line[1].split('/')[-1]
            depth = line[3].split('/')[-1]
            rgbFile = os.path.join(rgbDir, rgb)
            depthFile = os.path.join(depthDir, depth)
            img = cv.imread(rgbFile, cv.IMREAD_UNCHANGED)
            depth = cv.imread(depthFile, cv.IMREAD_UNCHANGED)
            yield img, depth
    else:
        rgbFileNames = [os.path.join(rgbDir, fileName)
                        for fileName in os.listdir(rgbDir)]
        depthFileNames = [None for _ in range(len(rgbFileNames))]
        if hasDepth:
            depthFileNames = [
                os.path.join(depthDir, fileName) for fileName in os.listdir(depthDir)
            ]

        for rgbFile, depthFile in zip(rgbFileNames, depthFileNames):
            img = cv.imread(rgbFile, cv.IMREAD_UNCHANGED)
            if hasDepth:
                if depthFile.endswith('.dat'):
                    depth = np.loadtxt(depthFile, delimiter=",",
                                    dtype=np.float64) * 1000.0
                elif depthFile.endswith('.png'):
                    depth = cv.imread(depthFile, cv.IMREAD_UNCHANGED)
                yield img, depth
            else:
                yield img, -1


FRAME_GENERATOR = createFrameGenerator()


def getFrame():
    for i in FRAME_GENERATOR:
        return i

    return -1, -1


#####################
# READ FRAMES END
#####################

# GLOBAL VARIABLES
# poseFig, poseAxis = plt.subplots()
depthFactor = metadata[DATASET]['depth_factor']
camera_matrix = metadata[DATASET]['camera_matrix']
dist_coff = metadata[DATASET]['dist_coff']

startIdx = 1
i = 0

# Skip initial frames (if needed)
while i < startIdx:
    img, depth = getFrame()
    i += 1

slamAlgorithm = SLAM(img, depth, depthFactor, camera_matrix, dist_coff)
while True:

    img, depth = getFrame()
    # SLAMMING
    slamAlgorithm.process(img, depth, i)
    # cv.waitKey(2000)
    i += 1

    # Update Measurements
    if np.isscalar(img) or i > 1500:
        break

ops = np.array(slamAlgorithm.open_points)
plt.scatter(ops[:, 0], ops[:, 1], c='#00ff00', alpha=1.0, s=1)
mps = np.array(slamAlgorithm.map_points)
plt.scatter(mps[:, 0], mps[:, 1], c='#000000', alpha=1.0, s=1)
ax = plt.gca()
ax.set_aspect('equal', 'box')
xLims = ax.get_xlim()
yLims = ax.get_ylim()
plt.savefig(f"dataviz/map.png")
slamAlgorithm.makeLogProbs(xLims, yLims)

trajectory = slamAlgorithm.get_trajectory()
visualize_data(visualize_trajectory, True, True, "3D", trajectory)

poses = slamAlgorithm.get_robot_poses()
visualize_data(plot_robot_poses, True, True, f"poses", poses)
