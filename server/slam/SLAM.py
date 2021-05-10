from Map import Map
from Landmark import Landmark
from Robot import Robot
from Localization import Localization
from Odometry import *
from utils import *
from Frame import Frame
from bovw import *
import icp
import glob
import cv2 as cv
import multiprocessing
import grid_map_utils as gmu
# import scipy.io
# import scipy.stats


class SLAM:
    def __init__(self, img, depth, depthFactor, camera_matrix, dist_coff):
        # GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose
        self.k = np.array(
            camera_matrix, dtype=np.float32,
        )
        self.dist_coff = np.array(
            dist_coff, dtype=np.float32,
        ) if dist_coff else dist_coff
        self.depthFactor = depthFactor
        self.VALID_DRANGE = [0, 200]
        self.MAP_SCALE = 100  # 1m
        self.MAP_SIZE = 1000
        self.CELL_SIZE = 1
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

        self.poses = [[0., 0., 0.]]  # initial pose
        self.trajectory = [np.array([0., 0., 0.])]  # 3d trajectory
        self.tMats = [
            (np.zeros((3, 3)), np.zeros((3, 1)))
        ]  # Transformation Matrices (rmat, tvec)
        self.map_points = []
        self.open_points = []

        # NEW MAP
        self.log_prob_map = None  # set all to zero
        # Log-Probabilities to add or remove from the map
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)

        frame = Frame(img, 1)
        self.frames = [frame]
        self.keyframes = [frame]
        cv.imwrite(f'dataviz/kfs/1.png', img)

    def process(self, img, depth, iterator):
        print(f"Processsing frame {iterator}")
        curr_frame = Frame(img, iterator)
        prev_frame = self.frames[-1]

        des_list = [prev_frame.des, curr_frame.des]
        kp_list = [prev_frame.kp, curr_frame.kp]

        # Part I. Feature Matching and Filtering (Nearest Neighbor Distance Ratios)
        matches = match_features(des_list)
        matches = filter_matches(matches)

        # matches, visual_words = build_bovw(des_list, matches)
        # assign_bow_index(visual_words, self.vocabulary)

        # Part II. Trajectory Estimation
        trajRes = estimate_trajectory(
            matches, kp_list, self.k, self.dist_coff, self.P, depth, self.depthFactor)

        if len(trajRes) == 0:
            print(f'\t->->Trajectory estimation failed!')
            return

        self.P, rmat, tvec, image1_points, image2_points, _, _, used_matches, inliersCount = trajRes

        # Remove same frames
        sorted_matches = sorted([m.distance for m in used_matches])
        move_mean = sum(sorted_matches) / len(sorted_matches)
        if move_mean < 10:
            print(f"\t->->Frame filtered because isSame: {move_mean}")
            return

        # Part III. Check Keyframe candidate
        # prev_keyframe = self.keyframes[-1]
        # des_list = [prev_keyframe.des, curr_frame.des]
        # kp_list = [prev_keyframe.kp, curr_frame.kp]
        # matches = match_features(des_list)
        # matches = filter_matches(matches)
        # trajRes = estimate_trajectory(
        #     matches, kp_list, self.k, self.dist_coff, self.P, depth, self.depthFactor)
        # if len(trajRes) == 0 or trajRes[-1] < 5:
        #     print(f'\tKeyframe detected!')
        #     cv.imwrite(f'dataviz/kfs/{iterator}.png', img)
        #     print('\tMatching with all prev keyframes...')
        #     for keyframe in self.keyframes:
        #         des_list = [keyframe.des, curr_frame.des]
        #         kp_list = [keyframe.kp, curr_frame.kp]
        #         matches = match_features(des_list)
        #         matches = filter_matches(matches)
        #         trajRes = estimate_trajectory(
        #             matches, kp_list, self.k, self.dist_coff, self.P, depth, self.depthFactor)
        #         if len(trajRes) > 0:
        #             print(f'\tLoop closure candidate: {keyframe.frame_nbr}')
        #             loopfp = f'dataviz/kfs/{keyframe.frame_nbr}.png'
        #             loopimg = cv.imread(loopfp, cv.IMREAD_UNCHANGED)
        #             visualize_matches(
        #                 loopimg, kp_list[0], img, kp_list[1], matches)
        #     self.keyframes.append(curr_frame)

        self.tMats.append((rmat, tvec))
        new_trajectory = self.P[:3, 3]
        self.trajectory.append(new_trajectory)

        curr_pose = calc_robot_pose(self.P[:3, :3], self.P[:, 3])
        self.poses.append(curr_pose)

        # Visualize traj
        map_points = self.calc_map_points(
            depth, curr_pose[2], curr_pose[0:2], self.MAP_SCALE)
        self.map_points.extend(map_points[1])
        self.open_points.extend(map_points[2])

        # cv.imshow('Log Map', self.log_prob_map)
        # cv.imshow('Log Map', 1.0 - 1./(1.+np.exp(self.log_prob_map)))
        matches = visualize_camera_movement(
            img, image1_points, img, image2_points)
        cv.imshow('Image', matches)
        cv.imshow('Map', self.map)
        cv.waitKey(20)

        # if iterator % 10 == 0:
        #     cv.imwrite(f'dataviz/pic{iterator}.png', images[1])
        #     ops = np.array(self.open_points)
        #     plt.scatter(ops[:, 0], ops[:, 1], c='#00ff00', alpha=1.0, s=1)
        #     mps = np.array(self.map_points)
        #     plt.scatter(mps[:, 0], mps[:, 1], c='#000000', alpha=1.0, s=1)
        #     ax = plt.gca()
        #     ax.set_aspect('equal', 'box')
        #     xLims = ax.get_xlim()
        #     yLims = ax.get_ylim()
        #     plt.savefig(f"dataviz/map{iterator}.png")

        self.frames.append(curr_frame)

    def depth_to_lidar(self, map_points):
        angles = []
        distances = []

        originX, originY = map_points[0]
        for point in map_points[1]:
            X, Y = point
            bearing = math.atan2(Y-originY, X-originX)
            distance = Y

            angles.append(bearing)
            distances.append(distance)

        return np.array(angles), np.array(distances)

    def update_map(self, map_points):
        for cp in map_points[1]:
            px, py = cp
            px -= px % self.CELL_SIZE
            py -= py % self.CELL_SIZE
            self.log_prob_map[py:py+self.CELL_SIZE,
                              px:px+self.CELL_SIZE] += self.l_occ
        for op in map_points[2]:
            px, py = op
            px -= px % self.CELL_SIZE
            py -= py % self.CELL_SIZE
            self.log_prob_map[py:py+self.CELL_SIZE,
                              px:px+self.CELL_SIZE] += self.l_free

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

    def calc_map_points(self, depth, angle, OFFSETS, scale):
        X_OFFSET, Y_OFFSET = OFFSETS

        points = []
        for col in range(0, depth.shape[1], 10):
            nonZeroVals = [
                (val, i)
                for i, val in enumerate(depth[self.VALID_DRANGE[0]:self.VALID_DRANGE[1], col])
                if val > 0
            ]
            if len(nonZeroVals):
                Z, row = min(nonZeroVals, key=lambda i: i[0])
                X, Y, Z = point2Dto3D((col, row), Z, self.k, self.depthFactor)
                if Z > 0 and Z < 20:
                    points.append([X, Y, Z, 1])

        points = np.array(points).T
        tPoints = self.P @ points
        tPoints = np.dot(tPoints, scale)
        tPoints[0, :] += X_OFFSET
        tPoints[2, :] += Y_OFFSET
        tPoints = tPoints[0:3:2].T

        freespaces = []
        for tp in tPoints:
            tp = [int(tp[0]), int(tp[1])]
            sp = [int(X_OFFSET), int(Y_OFFSET)]
            linePoints = gmu.bresenham(sp, tp)
            if linePoints[-1][0] == tp[0] and linePoints[-1][1] == tp[1]:
                freespaces.extend(linePoints[:-1])
            elif linePoints[0][0] == tp[0] and linePoints[0][1] == tp[1]:
                freespaces.extend(linePoints[1:])
            else:
                print('error')
                print(sp, tp, linePoints)
                exit()

        freespaces = np.array(freespaces)

        return (tuple(OFFSETS), tPoints, freespaces)

    def draw_map_points(self, points, showOpenSpaces=0, shouldDraw=1):
        color = self.MAP_COLOR['occupied'] if shouldDraw else self.MAP_COLOR['unexplored']
        opencolor = self.MAP_COLOR['open'] if shouldDraw else self.MAP_COLOR['unexplored']
        # if showOpenSpaces:
        #     for point in points[2]:
        #         px, py = point
        #         px -= px % self.CELL_SIZE
        #         py -= py % self.CELL_SIZE
        #         cv.rectangle(
        #             self.map,
        #             (px, py),
        #             (px+self.CELL_SIZE, py+self.CELL_SIZE),
        #             opencolor,
        #             -1,
        #         )
        for point in points[1]:
            px, py = point
            px -= px % self.CELL_SIZE
            py -= py % self.CELL_SIZE
            # py = self.MAP_SIZE - py
            cv.rectangle(
                self.map,
                (int(px), int(py)),
                (int(px+self.CELL_SIZE), int(py+self.CELL_SIZE)),
                color,
                -1,
            )

    def makeLogProbs(self, xLims, yLims):
        # print(xLims, yLims)
        xLims = (math.floor(xLims[0]), math.ceil(xLims[1]))
        yLims = (math.floor(yLims[0]), math.ceil(yLims[1]))
        # print(xLims, yLims)
        xSize = xLims[1] - xLims[0]
        ySize = yLims[1] - yLims[0]
        # print(xSize, ySize)

        self.log_prob_map = np.zeros(
            (ySize, xSize))
        # print(self.log_prob_map.shape)

        for i, op in enumerate(self.open_points):
            top = [int(op[0]-xLims[0]), int(op[1]-yLims[0])]
            px, py = top
            py = ySize - py
            self.log_prob_map[py, px] += self.l_free

        for i, mp in enumerate(self.map_points):
            tmp = [int(mp[0]-xLims[0]), int(mp[1]-yLims[0])]
            px, py = tmp
            py = ySize - py
            self.log_prob_map[py, px] += self.l_occ

        # for i, mp in enumerate(self.map_points):
        #     tmp = [int(mp[0]-xLims[0]), int(mp[1]-yLims[0])]
        #     px, py = tmp
        #     py = ySize - py
        #     cell_val = self.log_prob_map[py, px]
        #     px -= px % self.CELL_SIZE
        #     py -= py % self.CELL_SIZE
        #     self.log_prob_map[py:py+self.CELL_SIZE,
        #                       px:px+self.CELL_SIZE] = cell_val

        plt.clf()
        plt.imshow(1.0 - 1./(1.+np.exp(self.log_prob_map)),
                   'Greys')  # This is probability
        plt.show()
        # log probabilities (looks really cool)
        # plt.imshow(self.log_prob_map, 'Greys')
        # plt.show()
