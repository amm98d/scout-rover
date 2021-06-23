import numpy as np
import cv2 as cv
import zlib
import requests
import sys
sys.path.append("../common/")
sys.path.append("./slam/")
from SLAM import *


def get_color():
    response = requests.get('http://192.168.100.113:5000/color')
    return response.content


def get_depth():
    response = requests.get('http://192.168.100.113:5000/rgb')
    return response.content


def getFrame():
    color_bytes = get_color()
    depth_bytes = get_depth()
    
    color_bytes=zlib.decompress(color_bytes)
    depth_bytes=zlib.decompress(depth_bytes)
    
    color_frame = np.reshape(np.frombuffer(color_bytes, dtype=np.uint8), (480, 640))
    depth_frame = np.reshape(np.frombuffer(depth_bytes, dtype=np.uint16), (480, 640))

    return color_frame, depth_frame


def getCameraMatrix():
  fx = 594.21
  fy = 591.04
  a = -0.0030711
  b = 3.3309495
  cx = 339.5
  cy = 242.7
  mat = np.array([[1/fx, 0, -cx/fx],
                  [0, -1/fy, cy/fy],
                  [0,   0, -1]])
  return mat
#   return [[561.93206787, 0, 323.96944442], [ 0, 537.88018799, 249.35236366], [0, 0, 1]]


depthFactor = 1000
camera_matrix = [[561.93206787, 0, 323.96944442], [ 0, 537.88018799, 249.35236366], [0, 0, 1]]
dist_coff = [3.64965254e-01, -2.02943943e+00, -1.46113154e-03, 9.97005541e-03, 5.04006892e+00]


img, depth = getFrame()
slamAlgorithm = SLAM(img, depth, depthFactor, camera_matrix, dist_coff)
i = 1

while True:

    newImg, newDepth = getFrame()
    # SLAMMING
    # cv.imshow('img', newImg)
    # cv.imshow('depth', newDepth.astype(np.uint8))
    # print(np.amin(newDepth), np.amax(newDepth))
    # print(newDepth[240, 320])
    # cv.waitKey(20)
    slamAlgorithm.process([img, newImg], [depth, newDepth], i)
    # cv.waitKey(2000)
    i += 1

    img = newImg
    depth = newDepth

    # Update Measurements
    # if np.isscalar(img) or i > 100:
    #     break
