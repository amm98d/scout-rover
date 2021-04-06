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
        self.MAP_SIZE = 500
        self.ROVER_RADIUS = 5
        self.map = np.ones((self.MAP_SIZE, self.MAP_SIZE, 3))

        self.poses = [[0.0, 0.0, np.pi / 2]]  # initial pose
        self.trajectory = [np.array([0, 0, 0])]  # 3d trajectory
        self.tMats = [
            (np.zeros((3, 3)), np.zeros((3, 1)))
        ]  # Transformation Matrices (rmat, tvec)
        self.trail = []

    def process(self, images, depths, iterator):
        print(f"Processsing frame {iterator}")
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

        # Visualize traj
        OFFSETS = [self.MAP_SIZE // 2, self.MAP_SIZE // 2]
        SCALES = [10, 10]
        curr_pose = self.poses[-1]

        robot_points = self.calc_robot_points(curr_pose, OFFSETS, SCALES)
        self.trail.append((robot_points[0], robot_points[1]))
        self.draw_trail()
        self.draw_robot(robot_points)

        cv.imshow('Map', self.map)
        cv.imshow('Image', images[1])
        cv.waitKey(20)

        self.draw_robot(robot_points, 0)

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses

    def calc_robot_points(self, curr_pose, OFFSETS, SCALES):
        X_OFFSET, Y_OFFSET = OFFSETS
        X_SCALE, Y_SCALE = SCALES

        mapX = int(curr_pose[0] * X_SCALE + X_OFFSET)
        mapY = self.MAP_SIZE - int(curr_pose[1] * Y_SCALE + Y_OFFSET)

        thetaX = int(mapX + self.ROVER_RADIUS * math.cos(-curr_pose[2]))
        thetaY = int(mapY + self.ROVER_RADIUS * math.sin(-curr_pose[2]))

        return [mapX, mapY, thetaX, thetaY]

    def draw_robot(self, robot_points, shouldDraw=1):
        if shouldDraw:
            cv.circle(
                self.map,
                (robot_points[0], robot_points[1]),
                self.ROVER_RADIUS,
                (0, 0, 0),
                -1,
            )
            cv.line(
                self.map,
                (robot_points[0], robot_points[1]),
                (robot_points[2], robot_points[3]),
                (255, 255, 255),
                self.ROVER_RADIUS // 4,
            )
        else:
            cv.circle(
                self.map,
                (robot_points[0], robot_points[1]),
                self.ROVER_RADIUS,
                (255, 255, 255),
                -1,
            )

    def draw_trail(self):
        for pose in self.trail:
            cv.circle(
                self.map,
                pose,
                1,
                (0, 0, 255),
                -1,
            )
