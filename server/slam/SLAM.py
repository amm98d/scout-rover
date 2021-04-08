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
    def __init__(self, depthFactor, camera_matrix, dist_coff):
        # GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose

        self.k = np.array(
            camera_matrix, dtype=np.float32,
        )
        self.dist_coff = np.array(
            dist_coff, dtype=np.float32,
        ) if dist_coff else dist_coff
        self.depthFactor = depthFactor
        self.MAP_SIZE = 1000
        self.ROVER_RADIUS = 15
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
        self.P, rmat, tvec, image1_points, image2_points = estimate_trajectory(
            matches, kp_list, self.k, self.dist_coff, self.P, depths[1], self.depthFactor)
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
        SCALES = [100, 100]
        curr_pose = self.poses[-1]
        angle_diff = self.poses[0][2] - curr_pose[2]

        robot_points = self.calc_robot_points(curr_pose, OFFSETS, SCALES)
        map_points = self.calc_map_points(
            depths[1], angle_diff, robot_points[:2], SCALES)
        self.trail.append((robot_points[0], robot_points[1]))
        self.draw_trail()
        self.draw_robot(robot_points)
        self.draw_map_points(map_points)

        cv.imshow('Map', self.map)
        matches = visualize_camera_movement(
            images[0], image1_points, images[1], image2_points)
        cv.imshow('Image', matches)
        cv.waitKey(20)

        self.draw_robot(robot_points, 0)
        self.draw_map_points(map_points, 0)

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

        FOV = math.radians(57)  # 57 degrees
        VRANGE = 300  # 100px -> 1m
        ang1 = curr_pose[2] - FOV / 2
        ang2 = curr_pose[2] + FOV / 2

        line1X = int(mapX + VRANGE * math.cos(-ang1))
        line1Y = int(mapY + VRANGE * math.sin(-ang1))
        line2X = int(mapX + VRANGE * math.cos(-ang2))
        line2Y = int(mapY + VRANGE * math.sin(-ang2))

        return [mapX, mapY, thetaX, thetaY, line1X, line1Y, line2X, line2Y]

    def draw_robot(self, robot_points, shouldDraw=1):
        FOV_COLOR = (255, 255, 0)
        FOV_WIDTH = 2
        if shouldDraw:
            # FOV LINE 1
            cv.line(
                self.map,
                (robot_points[0], robot_points[1]),
                (robot_points[4], robot_points[5]),
                FOV_COLOR,
                FOV_WIDTH,
            )
            # FOV LINE 2
            cv.line(
                self.map,
                (robot_points[0], robot_points[1]),
                (robot_points[6], robot_points[7]),
                FOV_COLOR,
                FOV_WIDTH,
            )
            # ROVER BASE
            cv.circle(
                self.map,
                (robot_points[0], robot_points[1]),
                self.ROVER_RADIUS,
                (0, 0, 0),
                -1,
            )
            # ROVER DIRECTION
            cv.line(
                self.map,
                (robot_points[0], robot_points[1]),
                (robot_points[2], robot_points[3]),
                (255, 255, 255),
                self.ROVER_RADIUS // 4,
            )
        else:
            # FOV LINE 1
            cv.line(
                self.map,
                (robot_points[0], robot_points[1]),
                (robot_points[4], robot_points[5]),
                (255, 255, 255),
                FOV_WIDTH,
            )
            # FOV LINE 2
            cv.line(
                self.map,
                (robot_points[0], robot_points[1]),
                (robot_points[6], robot_points[7]),
                (255, 255, 255),
                FOV_WIDTH,
            )
            # ROVER BASE
            cv.circle(
                self.map,
                (robot_points[0], robot_points[1]),
                self.ROVER_RADIUS,
                (255, 255, 255),
                -1,
            )

    def draw_trail(self):
        if len(self.trail) < 2:
            return
        i = 0
        while i < len(self.trail) - 1:
            cv.line(
                self.map,
                self.trail[i],
                self.trail[i+1],
                (0, 0, 255),
                1
            )
            # cv.circle(
            #     self.map,
            #     self.trail[i],
            #     2,
            #     (0, 0, 255),
            #     -1,
            # )
            i += 1

    def calc_map_points(self, depth, angle, OFFSETS, SCALES):
        X_OFFSET, Y_OFFSET = OFFSETS
        X_SCALE, Y_SCALE = SCALES
        VALID_RANGE = [0, 200]

        points = []
        for col in range(0, depth.shape[1], 10):
            nonZeroVals = [
                (val, i)
                for i, val in enumerate(depth[VALID_RANGE[0]:VALID_RANGE[1], col])
                if val > 0
            ]
            if len(nonZeroVals):
                Z, row = min(nonZeroVals, key=lambda i: i[0])
                Z /= self.depthFactor
                if Z > 0 and Z < 1000:
                    X = (col - self.k[0][2]) * Z / self.k[0][0]
                    Y = (row - self.k[1][2]) * Z / self.k[1][1]

                    mapX = X * math.cos(angle) + Z * math.sin(angle)
                    mapY = Z * math.cos(angle) - X * math.sin(angle)
                    mapX = mapX * X_SCALE
                    mapY = mapY * Y_SCALE
                    mapX = mapX + X_OFFSET
                    mapY = self.MAP_SIZE - (mapY + Y_OFFSET)

                    points.append((int(mapX), int(mapY)))

        return points

    def draw_map_points(self, points, shouldDraw=1):
        color = (0, 0, 0) if shouldDraw else (255, 255, 255)
        for point in points:
            cv.circle(
                self.map,
                point,
                1,
                color,
                -1,
            )
