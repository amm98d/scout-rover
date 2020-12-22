import numpy as np
import cv2
from Odometry import *
from utils import *


class Robot:
    def __init__(self, pose=[0, 0, 0]):

        # Robot Pose
        self.x = round(pose[0], 5)
        self.y = round(pose[1], 5)
        self.theta = round(pose[2], 5)

        # Noises
        ## Motion noises
        # self.alpha1 = 0.25  # rotation
        # self.alpha2 = 0.175  # translation
        # self.alpha3 = 0.175  # translation
        # self.alpha4 = 0.27  # rotation
        self.alpha1 = 0.2  # rotation
        self.alpha2 = 0.03  # translation
        self.alpha3 = 0.3  # translation
        self.alpha4 = 0.2  # rotation
        ## Measurement noises
        # self.alpha5 = 0.0  # bearing
        self.alpha6 = 10  # range

        self.fov = np.pi / 2

    def move(self, motion_control):
        """This function returns a pose in world coordinates by using internal odometry information and current pose of the robot.
        Arguments:
        ----------
            motion_control: 2d array of shape (2,3) which contains xbarT and xbarT_1 which are robot poses at two consecutive timesteps in internal odometry.
        Returns:
        --------
            final_pose: Robot pose at current timestep in the global coordinate system.
        """
        # Seperate each coordinate from parameters
        xbarT_1, ybarT_1, thetabarT_1 = motion_control[0]
        xbarT, ybarT, thetabarT = motion_control[1]
        xT_1, yT_1, thetaT_1 = self.x, self.y, self.theta

        rot1 = np.arctan2(ybarT - ybarT_1, xbarT - xbarT_1) - thetabarT_1
        trans = np.sqrt(np.square(xbarT_1 - xbarT) + np.square(ybarT_1 - ybarT))
        rot2 = thetabarT - thetabarT_1 - rot1

        rothat1 = rot1 - np.random.normal(
            scale=np.square(self.alpha1 * rot1 + self.alpha2 * trans)
        )
        transhat = trans - np.random.normal(
            scale=np.square(self.alpha3 * trans + self.alpha4 * (rot1 + rot2))
        )
        rothat2 = rot2 - np.random.normal(
            scale=np.square(self.alpha1 * rot2 + self.alpha2 * trans)
        )

        # Calculate final pose
        xT = xT_1 + transhat * np.cos(thetaT_1 + rothat1)
        yT = yT_1 + transhat * np.sin(thetaT_1 + rothat1)
        thetaT = thetaT_1 + rothat1 + rothat2

        final_pose = [round(xT, 5), round(yT, 5), round(thetaT, 5)]
        self.x = final_pose[0]
        self.y = final_pose[1]
        self.theta = final_pose[2]

        return final_pose

    def measure(self, img1, landmarks, kp1, img_des, iterator):
        # Match features of landmark with img
        goodFeatures = []
        idx = -1
        MIN_MATCH_COUNT = 10  ##HYPER-PARAMETER
        # print("Len of landmark", len(landmarks))

        crop, height = applyTranformations(img1, str(iterator))
        
        
    

        if not isinstance(crop, int):

            Nkp1, Ndes1 = extract_features(crop)
            if len(Nkp1) <= 1:
                return -1, -1

            for k in range(len(landmarks)):
                # printFrame(crop, "Detect landmark")
                # print(len(Ndes1))
                # print(landmarks[k].des)

                good = match_features(landmarks[k].des, Ndes1)
                good = filter_matches_distance(good, 0.7)  ##HYPER-PARAMETER

        
                if len(good) > MIN_MATCH_COUNT:
                    goodFeatures.append((k, len(good)))

            if len(goodFeatures) == 0:
                idx = -1
            else:
                # print("GOOD Features tuple list", goodFeatures)
                tmpI, tmpCount = 0, 0
                for i, count in goodFeatures:
                    if count > tmpCount:
                        tmpI = i
                        tmpCount = count
                idx = tmpI

            if idx != -1:
                return idx, height
            else:
                return -1, -1
        else:
            return -1, -1

    def Gaussian(self, mu, sigma, x):

        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        # return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * pi * (sigma ** 2))
        return np.exp(-(np.square(mu - x) / np.square(sigma) / 2.0)) / np.sqrt(
            2.0 * np.pi * np.square(sigma)
        )

    # def measurement_prob(self, landmark, l_dist):
    #     lx = landmark.x
    #     ly = landmark.y

    #     rx = self.x
    #     ry = self.y
    #     rtheta = self.theta

    #     landmark_range = np.sqrt(np.square(lx - rx) + np.square(ly - ry))
    #     # landmark_bearing = np.arctan2(ly - ry, lx - rx) - rtheta

    #     # Add noise
    #     # landmark_range = np.random.normal(landmark_range, self.alpha6)
    #     # landmark_bearing = np.random.normal(landmark_bearing, self.alpha5)

    #     prob = self.Gaussian(landmark_range, self.alpha6, l_dist)

    #     # prob = landmark_range * landmark_bearing
    #     # if prob < 0:
    #     #     print(f"range: {landmark_range}, bearing: {landmark_bearing}")
    #     #     print(prob)
    #     #     prob = prob * (-1)

    #     return prob

    def measurement_prob(self, landmarks, calc_dist):
        # Rover Pose
        rx = self.x
        ry = self.y
        rtheta = self.theta

        prob = 1.0
        bearings = []
        for landmark in landmarks:
            lx = landmark.x
            ly = landmark.y

            landmark_bearing = np.arctan2(ly - ry, lx - rx) - rtheta
            if landmark_bearing < 0:
                landmark_bearing *= -1
            if landmark_bearing < self.fov / 2:
                bearings.append((landmark, landmark_bearing))
            # print(f"\tBearing for {(lx,ly)}: {landmark_bearing}")

        if len(bearings) > 0:
            closest = sorted(bearings, key=lambda x: x[1])[0]
            lx, ly = closest[0].x, closest[0].y
            # print(f"\tClosest: {(lx, ly)} -> {closest[1]}")
            real_dist = np.sqrt(np.square(lx - rx) + np.square(ly - ry))
            prob *= self.Gaussian(real_dist, self.alpha6, calc_dist)

            return prob

        return 0

    def __repr__(self):
        return f"({round(self.x, 2)}, {round(self.y, 2)}, {round(self.theta, 2)})"

    def get_pose(self):
        return [self.x, self.y, self.theta]
