import numpy as np
import cv2
from Odometry import extract_features


class Robot:
    def __init__(self, pose=[0, 0, 0]):

        # Robot Pose
        self.x = round(pose[0], 5)
        self.y = round(pose[1], 5)
        self.theta = round(pose[2], 5)

        # Noises
        ## Motion noises
        self.alpha1 = 0.0  # rotation
        self.alpha2 = 0.0  # translation
        self.alpha3 = 0.0  # translation
        self.alpha4 = 0.0  # rotation
        ## Measurement noises
        self.alpha5 = 0.0  # bearing
        self.alpha6 = 0.0  # range

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

    def measure(self, img1, landmarks):
        goodFeatures = []
        kp2, des2 = extract_features(img1)
        MIN_MATCH_COUNT = 10
        for k in range(len(landmarks)):
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(landmarks[k].des, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for i, result in enumerate(matches):
                if len(result) == 2:
                    m, n = result
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                # Visualizing
                goodFeatures.append((k, len(good)))

        if len(goodFeatures) == 0:
            return -1
        else:
            tmpI, tmpCount = 0, 0
            for i, count in goodFeatures:
                if count > tmpCount:
                    tmpI = i
                    tmpCount = count
            return tmpI

    def measurement_prob(self, landmark):
        lx = landmark.x
        ly = landmark.y

        rx = self.x
        ry = self.y
        rtheta = self.theta

        landmark_range = np.sqrt(np.square(lx - rx) + np.square(ly - ry))
        landmark_bearing = np.arctan2(ly - ry, lx - rx) - rtheta

        # Add noise
        landmark_range = np.random.normal(landmark_range, self.alpha6)
        landmark_bearing = np.random.normal(landmark_bearing, self.alpha5)

        prob = landmark_range * landmark_bearing
        if prob < 0:
            print(f"range: {landmark_range}, bearing: {landmark_bearing}")
            print(prob)
            prob = prob * (-1)

        return prob
