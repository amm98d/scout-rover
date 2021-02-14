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


#####################
# READ FRAMES START
#####################
DIR_PATH = "datasets/trajectory220"
NUM_FRAMES = 79

np.random.seed(1)


def createFrameGenerator():

    for i in range(1, NUM_FRAMES + 1):
        zeros = "0" * (5 - len(str(i)))
        fileName = f"{DIR_PATH}/{i}.jpg"
        print(f"Sending {fileName}")
        img = cv.imread(fileName, cv.IMREAD_UNCHANGED)
        # img = cv.flip(img, 1)
        yield img


FRAME_GENERATOR = createFrameGenerator()


def getFrame():
    for i in FRAME_GENERATOR:
        return i

    return -1


#####################
# READ FRAMES END
#####################

## GLOBAL VARIABLES
# poseFig, poseAxis = plt.subplots()
slamAlgorithm = SLAM()

frameA = getFrame()
frameB = getFrame()
images = [frameA, frameB]

i = 2
while True:
    # SLAMMING
    slamAlgorithm.process(images, i)
    i += 1

    # plt.cla()
    # trajectory = slamAlgorithm.get_trajectory()
    # visualize_trajectory(trajectory)
    # plt.pause(1e-16)

    # Update Measurements
    frameA = np.copy(frameB)
    frameB = getFrame()
    if np.isscalar(frameB):
        break
    images = [frameA, frameB]

trajectory = slamAlgorithm.get_trajectory()
visualize_data(visualize_trajectory, True, True, "3D", trajectory)

poses = slamAlgorithm.get_robot_poses()
visualize_data(plot_robot_poses, True, True, f"poses", poses)
