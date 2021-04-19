from Map import Map
from Landmark import Landmark
from Robot import Robot
from Localization import Localization
from Odometry import *
from utils import *
import icp
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
        self.MIN_INLIERS = 5
        self.MAP_SIZE = 1000
        self.ROVER_DIMS = [15, 29]
        self.ROVER_RADIUS = 15
        self.MAP_COLOR = {
            'unexplored': (200, 200, 200),
            'open': (255, 255, 255),
            'occupied': (0, 0, 0),
            'trail': (0, 0, 255),
            'fov': (255, 255, 0),
        }
        self.map = np.zeros((self.MAP_SIZE, self.MAP_SIZE, 3), np.uint8)
        self.map[:, :, :] = self.MAP_COLOR['unexplored']

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

        # if drop_frame(imgs_grey):
        #     #print("Dropping the frame")
        #     print("Sharpening the frame")
        #     fm = cv.Laplacian(imgs_grey[1], cv.CV_64F).var()
        #     if fm < 40:
        #         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #         im = cv.filter2D(imgs_grey[1], -1, kernel)  # sharp
        #         imgs_grey[1] = im

        # Part I. Features Extraction
        kp_list, des_list = extract_features(imgs_grey)

        # Part II. Feature Matching and Filtering
        matches = match_features(des_list)
        matches = filter_matches(matches)

        # Part III. Trajectory Estimation
        self.P, rmat, tvec, image1_points, image2_points, cloud1_points, cloud2_points, used_matches, inliersCount = estimate_trajectory(
            matches, kp_list, self.k, self.dist_coff, self.P, depths[1], self.depthFactor)

        # Remove same frames
        sorted_matches = sorted([m.distance for m in used_matches])
        move_mean = sum(sorted_matches) / len(sorted_matches)
        if move_mean < 10:
            print(f"\t->->Frame Filtered because isSame: {move_mean}")
            return

        # No motion estimation
        if np.isscalar(rmat):
            print(f"\t->->Frame Filtered because PnP failed")
            return

        # Not enough inliers
        if inliersCount < self.MIN_INLIERS:
            print(f"\t->->Frame Filtered because low inliers: {inliersCount}")
            return
            # To Do: ICP transformation estimation
        else:
            # Do ICP pose refinement of 3D point clouds
            T, distances, iterations = icp.icp(
                cloud2_points, cloud1_points, tolerance=1e-10)
            rmat = T[:3, :3]
            tvec = T[:3, 3:]

        self.tMats.append((rmat, tvec))
        new_trajectory = self.P[:3, 3]
        self.trajectory.append(new_trajectory)

        curr_pose = calc_robot_pose(self.P[:3, :3], self.P[:, 3])
        curr_pose[2] = self.poses[0][2] + (-1 * curr_pose[2])  # offset
        self.poses.append(curr_pose)

        # Visualize traj
        OFFSETS = [self.MAP_SIZE // 2, self.MAP_SIZE // 2]
        SCALES = [100, 100]

        robot_points = self.calc_robot_points(curr_pose, OFFSETS, SCALES)
        map_points = self.calc_map_points(
            depths[1], curr_pose[2], robot_points[1:3], SCALES)
        self.trail.append((robot_points[1], robot_points[2]))
        self.draw_trail()
        self.draw_map_points(map_points)
        self.draw_robot(robot_points, 0, 1)

        cv.imshow('Map', self.map)
        matches = visualize_camera_movement(
            images[0], image1_points, images[1], image2_points)
        cv.imshow('Image', matches)
        cv.waitKey(20)

        self.draw_robot(robot_points, 0, 0)
        # self.draw_map_points(map_points, 0)

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses

    def calc_robot_points(self, curr_pose, offsets, scales):
        X_OFFSET, Y_OFFSET = offsets
        X_SCALE, Y_SCALE = scales
        X_RADIUS, Y_RADIUS = self.ROVER_DIMS[1] / 2, self.ROVER_DIMS[0] / 2

        originX = int(curr_pose[0] * X_SCALE + X_OFFSET)
        originY = self.MAP_SIZE - int(curr_pose[1] * Y_SCALE + Y_OFFSET)

        thetaX = int(originX + Y_RADIUS * math.cos(-curr_pose[2]))
        thetaY = int(originY + Y_RADIUS * math.sin(-curr_pose[2]))

        FOV = math.radians(60)  # 57 degrees original value
        VRANGE = 300  # 100px -> 1m
        ang1 = curr_pose[2] - FOV / 2
        ang2 = curr_pose[2] + FOV / 2

        line1X = int(originX + VRANGE * math.cos(-ang1))
        line1Y = int(originY + VRANGE * math.sin(-ang1))
        line2X = int(originX + VRANGE * math.cos(-ang2))
        line2Y = int(originY + VRANGE * math.sin(-ang2))

        p1x = originX - X_RADIUS
        p1y = originY - Y_RADIUS
        p2x = originX + X_RADIUS
        p2y = originY - Y_RADIUS
        p3x = originX + X_RADIUS
        p3y = originY + Y_RADIUS
        p4x = originX - X_RADIUS
        p4y = originY + Y_RADIUS

        # Rotate all points
        p1x, p1y = self.rotate_point(
            (p1x, p1y), (originX, originY), curr_pose[2])
        p2x, p2y = self.rotate_point(
            (p2x, p2y), (originX, originY), curr_pose[2])
        p3x, p3y = self.rotate_point(
            (p3x, p3y), (originX, originY), curr_pose[2])
        p4x, p4y = self.rotate_point(
            (p4x, p4y), (originX, originY), curr_pose[2])

        boxPoints = np.array([[p1x, p1y], [p2x, p2y], [p3x, p3y], [
            p4x, p4y]], dtype=np.int32)

        return [boxPoints, originX, originY, thetaX, thetaY, line1X, line1Y, line2X, line2Y]

    def rotate_point(self, point, origin, angle):
        px = point[0] - origin[0]
        py = point[1] - origin[1]
        rotX = px * math.cos(angle) + py * math.sin(angle)
        rotY = py * math.cos(angle) - px * math.sin(angle)
        rotX += origin[0]
        rotY += origin[1]

        return rotX, rotY

    def draw_robot(self, robot_points, addFOV=1, shouldDraw=1, roverType='SQUARE'):
        ROVER_COLOR = self.MAP_COLOR['occupied'] if shouldDraw else self.MAP_COLOR['open']

        # FOV
        if addFOV:
            FOV_WIDTH = 2
            FOV_COLOR = self.MAP_COLOR['fov'] if shouldDraw else self.MAP_COLOR['unexplored']
            cv.line(
                self.map,
                (robot_points[1], robot_points[2]),
                (robot_points[5], robot_points[6]),
                FOV_COLOR,
                FOV_WIDTH,
            )
            cv.line(
                self.map,
                (robot_points[1], robot_points[2]),
                (robot_points[7], robot_points[8]),
                FOV_COLOR,
                FOV_WIDTH,
            )

        if roverType == 'SQUARE':
            rect = cv.minAreaRect(robot_points[0])
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.fillPoly(self.map, [box], ROVER_COLOR)  # filled
            # cv.drawContours(self.map, [box], 0, ROVER_COLOR, 2) #unfilled
        else:
            cv.circle(
                self.map,
                (robot_points[1], robot_points[2]),
                self.ROVER_DIMS[-1],
                ROVER_COLOR,
                -1,
            )

        # ROVER DIRECTION
        if shouldDraw:
            cv.line(
                self.map,
                (robot_points[1], robot_points[2]),
                (robot_points[3], robot_points[4]),
                self.MAP_COLOR['fov'],
                self.ROVER_DIMS[-1] // 4,
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
                self.MAP_COLOR['trail'],
                1
            )
            # cv.circle(
            #     self.map,
            #     self.trail[i],
            #     2,
            #     self.MAP_COLOR['trail'],
            #     -1,
            # )
            i += 1

    def calc_map_points(self, depth, angle, OFFSETS, SCALES):
        X_OFFSET, Y_OFFSET = OFFSETS
        X_SCALE, Y_SCALE = SCALES
        VALID_RANGE = [0, 200]
        angle += np.pi / 2

        points = []
        for col in range(0, depth.shape[1], 10):
            nonZeroVals = [
                (val, i)
                for i, val in enumerate(depth[VALID_RANGE[0]:VALID_RANGE[1], col])
                if val > 0
            ]
            if len(nonZeroVals):
                Z, row = min(nonZeroVals, key=lambda i: i[0])
                X, Y, Z = point2Dto3D((col, row), Z, self.k, self.depthFactor)
                if Z > 0 and Z < 2:
                    X *= -1
                    mapX = X * math.cos(angle) + Z * math.sin(angle)
                    mapY = Z * math.cos(angle) - X * math.sin(angle)
                    mapX = mapX * X_SCALE
                    mapY = mapY * Y_SCALE
                    mapX = mapX + X_OFFSET
                    mapY = mapY + Y_OFFSET

                    points.append((int(mapX), int(mapY)))

        return (tuple(OFFSETS), points)

    def draw_map_points(self, points, shouldDraw=1):
        color = self.MAP_COLOR['occupied'] if shouldDraw else self.MAP_COLOR['unexplored']
        for point in points[1]:
            # cv.line(
            #     self.map,
            #     points[0],
            #     point,
            #     self.MAP_COLOR['open'],
            #     1,
            # )
            cv.circle(
                self.map,
                point,
                1,
                color,
                -1,
            )
