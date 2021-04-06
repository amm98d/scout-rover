from Map import Map
from Landmark import Landmark
from Robot import Robot
from Localization import Localization
from Odometry import *
from utils import *
import glob
import cv2 as cv
import multiprocessing


class SLAM:
    def __init__(self, depthFactor, camera_matrix):
        # GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose

        self.k = np.array(
            camera_matrix, dtype=np.float32,
        )
        self.depthFactor = depthFactor

        self.poses = [[0.0, 0.0, np.pi / 2]]  # initial pose
        self.trajectory = [np.array([0, 0, 0])]  # 3d trajectory
        self.tMats = [
            (np.zeros((3, 3)), np.zeros((3, 1)))
        ]  # Transformation Matrices (rmat, tvec)

    def process(self, images, depths, iterator):
        imgs_grey = [
            cv.cvtColor(images[0], cv.COLOR_BGR2GRAY),
            cv.cvtColor(images[1], cv.COLOR_BGR2GRAY),
        ]

        # Check if this frame should be dropped (blur/same)

        if drop_frame(imgs_grey):
            #print("Dropping the frame")
            print("Sharpening the frame")
            fm = cv.Laplacian(imgs_grey[1], cv.CV_64F).var()
            if fm < 40:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                im = cv.filter2D(imgs_grey[1], -1, kernel)  # sharp
                imgs_grey[1] = im

        # Part I. Features Extraction
        kp_list, des_list = extract_features(imgs_grey)

        # Part II. Feature Matching
        matches = match_features(des_list)
        is_main_filtered_m = True  # Filter matches
        if is_main_filtered_m:
            filtered_matches = filter_matches(matches)
            matches = filtered_matches

        # Removing Same frames
        smatches = sorted(matches, key=lambda x: x.distance)
        sdiff = sum([x.distance for x in smatches[:500]])
        if sdiff < 1000:
            print(f"\t->->Frame Filtered because isSame: {sdiff}")
            return

        # Part III. Trajectory Estimation
        # Essential Matrix or PNP
        # pnp_estimation || essential_matrix_estimation
        self.P, rmat, tvec = estimate_trajectory(
            matches, kp_list, self.k, self.P, depths[1], self.depthFactor)
        # No motion estimation
        if np.isscalar(rmat):
            return
        # Compare same frame
        # prevRmat, prevTvec = self.tMats[-1]
        # if not np.allclose(rmat, prevRmat) and not np.allclose(tvec, prevTvec):
        self.tMats.append((rmat, tvec))
        new_trajectory = self.P[:3, 3]
        self.trajectory.append(new_trajectory)
        self.poses.append(calc_robot_pose(self.P[:3, :3], self.P[:, 3]))

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses
