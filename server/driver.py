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
DATASET = 0
NUM_FRAMES = -1

metadata = {
    0: {
        'directory': os.path.join('datasets', 'carla'),
        'depth': True,
        'associate': False,
    },
    1: {
        'directory': os.path.join('datasets', 'fr1_xyz'),
        'depth': True,
        'associate': True,
    },
    2: {
        'directory': os.path.join('datasets', 'fr1_rpy'),
        'depth': True,
        'associate': True,
    },
    3: {
        'directory': os.path.join('datasets', 'fr2_pslam'),
        'depth': True,
        'associate': True,
    },
    4: {
        'directory': os.path.join('datasets', 'trajectory220'),
        'depth': False,
        'associate': False,
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
                depth = np.loadtxt(depthFile, delimiter=",",
                                   dtype=np.float64) * 1000.0
                # depth = cv.imread(depthFile, cv.IMREAD_UNCHANGED)
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
slamAlgorithm = SLAM()

frameA, depthA = getFrame()
frameB, depthB = getFrame()
images = [frameA, frameB]
depths = [depthA, depthB]

i = 2
while True:
    # SLAMMING
    slamAlgorithm.process(images, depths, i)
    i += 1

    # plt.cla()
    # trajectory = slamAlgorithm.get_trajectory()
    # visualize_trajectory(trajectory)
    # plt.pause(1e-16)

    # Update Measurements
    frameA = np.copy(frameB)
    depthA = np.copy(depthB)
    frameB, depthB = getFrame()
    if np.isscalar(frameB):
        break
    images = [frameA, frameB]
    depths = [depthA, depthB]

trajectory = slamAlgorithm.get_trajectory()
visualize_data(visualize_trajectory, True, True, "3D", trajectory)

poses = slamAlgorithm.get_robot_poses()
visualize_data(plot_robot_poses, True, True, f"poses", poses)
