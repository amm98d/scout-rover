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

    def create_point_cloud(self, img, depth, depth_factor, camera_matrix, tmat):
        points = []
        norm_colors = []
        for u in range(0, img.shape[1], 1):
            for v in range(0, img.shape[0], 1):
                Z = depth[v, u]
                if Z < 4:
                    points.append(point2Dto3D((u, v), Z, camera_matrix, depth_factor))
                    norm_colors.append(img[v, u] / 255)

        self.pc_points = np.array(points)
        self.pc_colors = np.array(norm_colors)

        # Transform point cloud
        points = np.ones((self.pc_points.shape[0], 4))
        points[:,:3] = self.pc_points

        tPoints = tmat @ points.T
        tPoints = tPoints[:3,:].T

        self.pc_points = tPoints