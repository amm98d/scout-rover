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
import grid_map_utils as gmu
# import scipy.io
# import scipy.stats


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
        self.VALID_DRANGE = [0, 200]
        self.MAP_SCALE = 100
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

        self.poses = [[0.0, 0.0, np.pi / 2]]  # initial pose
        self.trajectory = [np.array([0, 0, 0])]  # 3d trajectory
        self.tMats = [
            (np.zeros((3, 3)), np.zeros((3, 1)))
        ]  # Transformation Matrices (rmat, tvec)
        self.map_points = []
        self.open_points = []
        self.trail = []

        # NEW MAP
        self.log_prob_map = None  # set all to zero
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)

        self.imagingQ = [self.map]
        self.matchviz = self.map

    def getMap(self):
        if len(self.imagingQ) == 1:
            return self.imagingQ[0]
        return self.imagingQ.pop(0)

    def process(self, images, depths, iterator):
        print(f"Processsing frame {iterator}")
        imgs_grey = [
            cv.cvtColor(images[0], cv.COLOR_BGR2GRAY),
            cv.cvtColor(images[1], cv.COLOR_BGR2GRAY),
        ]

        # Part I. Features Extraction
        kp_list, des_list = extract_features(imgs_grey)

        # Part II. Feature Matching and Filtering
        matches = match_features(des_list)
        matches = filter_matches(matches)

        # Part II. Trajectory Estimation
        trajRes = estimate_trajectory(
            matches, kp_list, self.k, self.dist_coff, self.P, depths[1], self.depthFactor)

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

        self.tMats.append((rmat, tvec))
        new_trajectory = self.P[:3, 3]
        self.trajectory.append(new_trajectory)

        curr_pose = calc_robot_pose(self.P[:3, :3], self.P[:, 3])
        self.poses.append(curr_pose)
        curr_pose[0] += 200
        curr_pose[1] += 200

        # Visualize traj
        map_points = self.calc_map_points(
            depths[1], curr_pose[2], curr_pose[0:2], self.MAP_SCALE)
        self.map_points.extend(map_points[1])
        self.open_points.extend(map_points[2])

        self.trail.append((curr_pose[0], curr_pose[1]))
        self.draw_trail()
        self.draw_map_points(map_points, 0, 1)
        self.imagingQ.append(np.copy(self.map))

        self.matchviz = visualize_camera_movement(
            images[0], image1_points, images[1], image2_points)
        cv.imshow('Image', self.matchviz)
        # cv.imshow('RGB', images[1])
        # cv.imshow('DEPTH', depths[1][self.VALID_DRANGE[0]:self.VALID_DRANGE[1], :])
        cv.imshow('Map', self.map)
        cv.waitKey(20)
        self.draw_map_points(map_points, 0, 0)

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses

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
                if Z > 0 and Z < 4:
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