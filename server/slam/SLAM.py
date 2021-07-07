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
import open3d as o3d
# import scipy.io
# import scipy.stats
import numpy as np

class SLAM:
    def __init__(self, img, depth, depthFactor, camera_matrix, dist_coff):
        # GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose
        self.k = np.array(
            camera_matrix, dtype=np.float32,
        )
        # self.ki = o3d.camera.PinholeCameraIntrinsic()
        # self.ki.set_intrinsics(
        #     640, 480, 525., 525., 319.5, 239.5
        # )
        self.dist_coff = np.array(
            dist_coff, dtype=np.float32,
        ) if dist_coff else dist_coff
        self.depthFactor = depthFactor
        self.VALID_DRANGE = [0, 260]
        self.MAP_SCALE = 100  # 1m
        self.MAP_SIZE = 700
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

        self.last_frame = Frame(img, 1)

        # self.prev_pc = None
        # self.o3vis = o3d.visualization.Visualizer()
        # self.o3vis.create_window()
        self.frontier_points = []

    def process(self, imgs, depths, iterator):
        print(f"Processsing frame {iterator}")
        curr_frame = Frame(imgs[1], iterator)
        prev_frame = self.last_frame

        des_list = [prev_frame.des, curr_frame.des]
        kp_list = [prev_frame.kp, curr_frame.kp]

        # Part I. Feature Matching and Filtering (Nearest Neighbor Distance Ratios)
        matches = match_features(des_list)
        matches = filter_matches(matches)

        # matches, visual_words = build_bovw(des_list, matches)
        # assign_bow_index(visual_words, self.vocabulary)

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

        # Visualize traj
        # map_points = self.calc_map_points(
        #     depths[1], curr_pose[2], curr_pose[0:2], self.MAP_SCALE)
        map_points = self.calc_map_points(
            depths[1], curr_pose[2], [350, 350], 250)
        self.map_points.extend(map_points[1])
        self.open_points.extend(map_points[2])

        # self.draw_dummy_points(depths[1], 1)
        self.draw_map_points(map_points[1], map_points[2])
        self.detect_frontiers()
        for i in self.frontier_points:
            # print(self.map[])
            self.map[i[1][0]][i[1][1]] = [255,0,0]
            # print(self.frontier_points[i])
        # print(self.frontier_points[:5],end='\n')
        # input()

        cv.imshow('map', self.map)
        # cv.imwrite(f"dataviz/map/map-{iterator}.png", self.map)

        # cv.imshow('depth', depths[1][self.VALID_DRANGE[0]:self.VALID_DRANGE[1], :].astype(np.uint8))

        cv.imshow('img', imgs[1][self.VALID_DRANGE[0]:self.VALID_DRANGE[1], :])
        # cv.imwrite(f"dataviz/img/img-{iterator}.png", imgs[1][self.VALID_DRANGE[0]:self.VALID_DRANGE[1], :])
        cv.waitKey(100)

        self.last_frame = curr_frame

    def find_adjacents(self, point, tMap):
        adjacents = []
        pointCoords = point[1]
        adjacents.append(tMap[pointCoords[0]][pointCoords[1]-1]) #up
        adjacents.append(tMap[pointCoords[0]][pointCoords[1]+1]) #down
        adjacents.append(tMap[pointCoords[0]-1][pointCoords[1]]) #left
        adjacents.append(tMap[pointCoords[0]+1][pointCoords[1]]) #right
        return adjacents

    def find_openspace_neighbors(self, point, tMap):
        return [neighbour for neighbour in self.find_adjacents(point, tMap) if (neighbour[0]==[255,255,255]).all()]

    def is_frontier_point(self, point, tMap):
        all_neighbours = self.find_adjacents(point, tMap)
        # criteria for frontier-point: atleast 1 openspace point and atleast 1 unexplored point
        hasUnexploredNeighbour = False
        hasOpenspaceNeighbour = False
        for n in all_neighbours:
            if (n[0]==[255,255,255]).all():
                hasOpenspaceNeighbour = True
            elif (n[0]==[200,200,200]).all():
                hasUnexploredNeighbour = True
        return hasUnexploredNeighbour and hasOpenspaceNeighbour

    def detect_frontiers(self):
        # for i in range(self.map.shape[0]):
        #     if (not (np.unique(self.map[i],axis=0)==[[200,200,200]]).all()):
        #         u, indices = np.unique(self.map[i],axis=0,return_index=True)
        #         for valInd in range(len(u)):
        #             print(u[valInd],indices[valInd])
        #         print("==============================")
        tMap = []
        for row in range(self.map.shape[0]):
            tempRow = []
            for col in range(self.map.shape[1]):
                tempRow.append([self.map[row][col],(row,col),'-'])
            tMap.append(tempRow)
        queueM = []
        pose = tMap[int(len(tMap[0])/2)][int(len(tMap[1])/2)]
        pose[2] = 'Map-Open-List'
        queueM.append(pose)
        while(len(queueM)>0):
            p = queueM.pop()
            if p[2] == 'Map-Close-List':
                continue
            if self.is_frontier_point(p, tMap):
                queueF = []
                p[2] = 'Frontier-Open-List'
                queueF.append(p)
                while (len(queueF)>0):
                    q = queueF.pop()
                    if q[2] == 'Map-Close-List' or q[2] == 'Frontier-CloseList':
                        continue
                    if self.is_frontier_point(q, tMap):
                        self.frontier_points.append(q)
                        for w in self.find_adjacents(q, tMap):
                            if w[2]!='Frontier-OpenList' and w[2]!='Frontier-Close-List' and w[2]!='Map-Close-List':
                                w[2] = 'Frontier-OpenList'
                                queueF.append(w)
                    q[2] = 'Frontier-Close-List'
                for each in self.frontier_points:
                    each[2] = 'Map-Close-List'
            for v in self.find_adjacents(p, tMap):
                if v[2]!='Map-Open-List' and v[2]!='Map-Close-List' and len(self.find_openspace_neighbors(v, tMap))>0:
                    queueM.append(v)
            p[2] = 'Map-Close-List'

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses

    def draw_dummy_points(self, depth, isDraw):
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
                if Z > 0 and Z <= 2:
                    points.append([X, Y, Z, 1])

        points = np.array(points).T
        tPoints = self.P @ points
        # tPoints = np.dot(points, self.MAP_SCALE)
        tPoints[1, :] = tPoints[2, :]
        tPoints[2, :] = 0
        tPoints = tPoints[0:3].T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tPoints)

        if self.prev_pc:
            self.o3vis.remove_geometry(self.prev_pc)
        self.o3vis.add_geometry(pcd)
        self.o3vis.update_geometry(pcd)
        self.o3vis.poll_events()
        self.o3vis.update_renderer()
        self.prev_pc = pcd

    def calc_map_points(self, depth, angle, OFFSETS, scale):
        X_OFFSET, Y_OFFSET = OFFSETS

        points = []
        borders = []
        for col in range(0, depth.shape[1], 10):
            nonZeroVals = [
                (val, i)
                for i, val in enumerate(depth[self.VALID_DRANGE[0]:self.VALID_DRANGE[1], col])
                if val > 0
            ]
            if len(nonZeroVals):
                Z, row = min(nonZeroVals, key=lambda i: i[0])
                X, Y, Z = point2Dto3D((col, row), Z, self.k, self.depthFactor)
                if Z > 0 and Z <= 2:
                    points.append([X, Y, Z, 1])
                elif Z > 2:
                    borders.append([X, Y, 2, 1])

        points = np.array(points).T
        # tPoints = self.P @ points
        tPoints = points
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

        if len(borders):
            borders = np.array(borders).T
            # tPoints = self.P @ points
            tBorders = borders
            tBorders = np.dot(tBorders, scale)
            tBorders[0, :] += X_OFFSET
            tBorders[2, :] += Y_OFFSET
            tBorders = tBorders[0:3:2].T
            for tb in tBorders:
                tb = [int(tb[0]), int(tb[1])]
                sp = [int(X_OFFSET), int(Y_OFFSET)]
                linePoints = gmu.bresenham(sp, tb)
                if linePoints[-1][0] == tb[0] and linePoints[-1][1] == tb[1]:
                    freespaces.extend(linePoints[:-1])
                elif linePoints[0][0] == tb[0] and linePoints[0][1] == tb[1]:
                    freespaces.extend(linePoints[1:])
                else:
                    print('error')
                    print(sp, tb, linePoints)
                    exit()

        freespaces = np.array(freespaces)

        return (tuple(OFFSETS), tPoints, freespaces)

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

    def draw_map_points(self, map_points, open_points):
        self.map[:, :, :] = self.MAP_COLOR['unexplored']
        for mp in map_points:
            cv.circle(
                self.map,
                (int(mp[0]), int(self.MAP_SIZE - mp[1])),
                1,
                self.MAP_COLOR['occupied'],
                -1
            )
        for op in open_points:
            cv.circle(
                self.map,
                (int(op[0]), int(self.MAP_SIZE - op[1])),
                1,
                self.MAP_COLOR['open'],
                -1
            )
