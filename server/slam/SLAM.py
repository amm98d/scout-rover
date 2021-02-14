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
    def __init__(self):
        # GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose

        # self.k = np.array(
        #     [[640.0, 0, 640.0], [0, 480.0, 480.0], [0, 0, 1.0]], dtype=np.float32,
        # )
        # ahmed dataset
        self.k = np.array(
            [[827.0, 0.0, 638.0], [0.0, 826.0, 347], [0.0000, 0.0000, 1.0000]],
            dtype=np.float32,
        )

        self.poses = [[0.0, 0.0, np.pi / 2]]  # initial pose
        self.trajectory = [np.array([0, 0, 0])]  # 3d trajectory
        self.tMats = [
            (np.zeros((3, 3)), np.zeros((3, 1)))
        ]  # Transformation Matrices (rmat, tvec)
        self.env_map = Map(62, 62)
        self._create_map(self.env_map)
        self.particle_filter = Localization(self.env_map)

    def _create_map(self, env_map):
        # store landmarks
        landmark_imgs = glob.glob("./train/*.jpg")
        crd = [(0, 62), (62, 62)]  # coordinates
        for idx, img in enumerate(landmark_imgs):
            img = cv.imread(img)

            croppedImg, _ = applyTranformations(img)
            # gray = cv.cvtColor(croppedImg, cv.COLOR_BGR2GRAY)
            # print(croppedImg.shape)
            kp, des = orb_extractor(croppedImg)
            env_map.add_landmark(
                Landmark(crd[idx][0], crd[idx][1], kp, des, 12))

    def process(self, images, iterator):
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
        kp_list, des_list = extract_features(
            imgs_grey, extract_features_function=extract_features
        )

        # Part II. Feature Matching
        matches = match_features(des_list, match_features)
        is_main_filtered_m = True  # Filter matches
        if is_main_filtered_m:
            filtered_matches = filter_matches(
                threshold_filter, matches)
            matches = filtered_matches

        # Removing Same frames
        smatches = sorted(matches[0], key=lambda x: x.distance)
        sdiff = sum([x.distance for x in smatches[:500]])
        if sdiff < 1000:
            print(f"\t->->Frame Filtered because isSame: {sdiff}")
            return

        # Part III. Trajectory Estimation
        # Essential Matrix or PNP
        # pnp_estimation || essential_matrix_estimation
        self.P, rmat, tvec = estimate_trajectory(
            em_estimation, matches, kp_list, self.k, self.P
        )
        # No motion estimation
        if np.isscalar(rmat):
            return
        # Compare same frame
        # prevRmat, prevTvec = self.tMats[-1]
        # if not np.allclose(rmat, prevRmat) and not np.allclose(tvec, prevTvec):
        self.tMats.append((rmat, tvec))
        new_trajectory = self.P[:3, 3]
        self.trajectory.append(new_trajectory * 2.95)
        self.poses.append(calc_robot_pose(self.P[:3, :3], self.P[:, 3] * 2.95))
        # else:
        #     print(f"\t->Frame Filtered because same TMat")
        #     return

        # Part IV. Localize
        last_pose = self.poses[-1]
        second_last_pose = self.poses[-2]
        print(f"Odometry:\n\t{[second_last_pose, last_pose]}")
        self.particle_filter.motion_update([second_last_pose, last_pose])
        if iterator % 5 == 0:
            print(">>>>> Updating Measurement")
            self.particle_filter.measurement_update(
                images[1], kp_list[1], des_list[1], iterator
            )
            self.particle_filter.sample_particles()

        # Part V. Save Visualization plot
        visualize_data(self.env_map.plot_map, showPlot=False)
        visualize_data(
            self.particle_filter.plot_particles,
            clean_start=False,
            showPlot=False,
            figName=f"frame{iterator}",
        )

        # plt.cla()
        # npTraj = np.array(self.trajectory).T
        # visualize_trajectory(npTraj)
        # plt.savefig(f'dataviz/frame{i}.png')

    def get_trajectory(self):
        return np.array(self.trajectory).T

    def get_robot_poses(self):
        return self.poses
