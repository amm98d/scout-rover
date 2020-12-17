from Map import Map
from Landmark import Landmark
from Robot import Robot
from Localization import Localization
from Odometry import *
from utils import *
import glob

class SLAM:

    def __init__(self):
        ## GLOBAL VARIABLES
        self.P = np.eye(4)  # Pose
        self.k = np.array([[923.0, 0.0, 657.0], [0.0, 923.0, 657.0], [0.0000, 0.0000, 1.0000]], dtype=np.float32)
        self.poses = [[0.0, 0.0, 0.0]]  # initial pose
        ## Create Map
        self.env_map = Map(62, 62)
        ##store landmarks
        getLandmarks = True
        if getLandmarks:
            hamza_images = glob.glob("./train/*.jpg")
            # 0->ecat
            # 1->chemistry
            # 2->physics
            # 3->genis
            crd = [(0, 0), (62, 0), (0, 62), (62, 62)]  # coordinates
            i = 0
            for img in hamza_images:
                img = cv2.imread(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = extract_features(gray)
                self.env_map.add_landmark(Landmark(crd[i][0], crd[i][1], kp, des))
                i += 1
        self.particle_filter = Localization(self.env_map)
        self.trajectory = [np.array([0, 0, 0])]
        self.camPoses = []

    def process(self, images):
        # Part I. Features Extraction
        kp_list, des_list = extract_features_dataset(images, extract_features_function=extract_features)
        # Part II. Feature Matching
        matches = match_features_dataset(des_list, match_features)
        # Set to True if you want to use filtered matches or False otherwise
        is_main_filtered_m = False
        if is_main_filtered_m:
            filtered_matches = filter_matches_dataset(filter_matches_distance, matches)
            matches = filtered_matches
        # Part III. Trajectory Estimation
        self.P, rmat, tvec = estimate_trajectory(estimate_motion, matches, kp_list, self.k, self.P)
        if len(self.camPoses) == 0:
            self.camPoses.append((rmat, tvec))
            new_trajectory = self.P[:3, 3]
            self.trajectory.append(new_trajectory)
            # self.poses.extend(pose)
            # self.trajectory.append(self.P[:3, 3])
        else:
            prevRmat, prevTvec = self.camPoses[-1]
            if not np.allclose(rmat, prevRmat) and not np.allclose(tvec, prevTvec):
                self.camPoses.append((rmat, tvec))
                new_trajectory = self.P[:3, 3]
                self.trajectory.append(new_trajectory)
                # self.poses.extend(pose)
                # self.trajectory.append(self.P[:3, 3])
            else:
                print("frame filtered...")
        # Part IV. Localize
        # last_pose = self.poses[-1]
        # second_last_pose = self.poses[-2]
        # self.particle_filter.motion_update([second_last_pose, last_pose])
        # self.particle_filter.measurement_update(images[1])
        # self.particle_filter.sample_particles()
        # Part V. Save Visualization plot
        # self.particle_filter.plot_data(i)
        # Save visualization frames
        plt.cla()
        npTraj = np.array(self.trajectory).T
        visualize_trajectory(npTraj)
        plt.savefig(f'dataviz/frame{i}.png')
