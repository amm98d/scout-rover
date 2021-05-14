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


class SLAM:
    def __init__(self, img, depth, depthFactor, camera_matrix, dist_coff):
        # GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose
        self.k = np.array(
            camera_matrix, dtype=np.float32,
        )
        self.ki = o3d.camera.PinholeCameraIntrinsic()
        self.ki.set_intrinsics(
            640, 480, 525., 525., 319.5, 239.5
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

        self.prev_pc = None
        self.o3vis = o3d.visualization.Visualizer()
        self.o3vis.create_window()

    def process(self, imgs, depths, iterator):
        print(f"Processsing frame {iterator}")
        curr_frame = Frame(imgs[1], iterator)
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

        threshold = 0.02
        trans_init = self.P
        source = self.prev_pc
        target = self.create_pc(depths[1])
        if target:
            if not source:
                source = target
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            self.P = reg_p2p.transformation
            self.draw_registration_result(source, target, reg_p2p.transformation)
            self.prev_pc = target

        self.tMats.append((rmat, tvec))
        new_trajectory = self.P[:3, 3]
        self.trajectory.append(new_trajectory)

        curr_pose = calc_robot_pose(self.P[:3, :3], self.P[:, 3])
        self.poses.append(curr_pose)

        # Visualize traj
        map_points = self.calc_map_points(
            depths[1], curr_pose[2], curr_pose[0:2], self.MAP_SCALE)
        self.map_points.extend(map_points[1])
        self.open_points.extend(map_points[2])

        self.draw_dummy_points(depths[1], 1)
        # cv.imshow('Log Map', self.log_prob_map)
        # cv.imshow('Log Map', 1.0 - 1./(1.+np.exp(self.log_prob_map)))
        # matches = visualize_camera_movement(
        #     img, image1_points, img, image2_points)
        # cv.imshow('Image', matches)
        # cv.imshow('Map', self.map)
        # cv.waitKey(20)

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

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses

    def draw_registration_result(self, source, target, transformation):
        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])
        source.transform(transformation)
        o3d.visualization.draw_geometries([source, target])

    def create_pc(self, depth):
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

        if len(points):
            tPoints = np.array(points).T
            tPoints[1, :] = tPoints[2, :]
            tPoints[2, :] = 0
            tPoints = tPoints[0:3].T

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(tPoints)

            return pcd

        return None

    def intensity_estimation(self, imgs, depths):
        source_color = o3d.geometry.Image(imgs[0])
        source_depth = o3d.geometry.Image(depths[0])
        target_color = o3d.geometry.Image(imgs[1])
        target_depth = o3d.geometry.Image(depths[1])
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth, depth_scale=5000, depth_trunc=4)
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_color, target_depth, depth_scale=5000, depth_trunc=4)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd_image, self.ki)

        option = o3d.pipelines.odometry.OdometryOption()
        odo_init = self.P

        [success_hybrid_term, trans_hybrid_term,
        info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, self.ki, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

        if success_hybrid_term:
            return [trans_hybrid_term]
        else:
            return []

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
                if Z > 0:
                    points.append([X, Y, Z, 1])

        points = np.array(points).T
        tPoints = self.P @ points
        # tPoints = np.dot(points, self.MAP_SCALE)
        tPoints[1, :] = tPoints[2, :]
        tPoints[2, :] = 0
        tPoints = tPoints[0:3].T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tPoints)

        self.o3vis.add_geometry(pcd)
        self.o3vis.update_geometry(pcd)
        self.o3vis.poll_events()
        self.o3vis.update_renderer()

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
