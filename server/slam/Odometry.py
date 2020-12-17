import numpy as np
import cv2
from utils import *


def extract_features(image):

    orb = cv2.ORB_create(nfeatures=5000, WTA_K=4)
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    return kp, des


def extract_features_dataset(images, extract_features_function):
    kp_list = []
    des_list = []

    ### START CODE HERE ###
    for img in images:
        kp, des = extract_features(img)
        kp_list.append(kp)
        des_list.append(des)

    ### END CODE HERE ###

    return kp_list, des_list


def visualize_features(image, kp):
    display = cv2.drawKeypoints(image, kp, None)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)


def match_features(des1, des2):

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    match = flann.knnMatch(des1, des2, k=2)

    return match


def match_features_dataset(des_list, match_features):
    matches = []

    ### START CODE HERE ###
    for i in range(len(des_list) - 1):
        descriptor1 = des_list[i]
        descriptor2 = des_list[i + 1]
        match = match_features(descriptor1, descriptor2)
        matches.append(match)

    ### END CODE HERE ###

    return matches


def filter_matches_distance(match, dist_threshold):
    filtered_match = []
    for i, result in enumerate(match):
        if len(result) == 2:
            m, n = result
            if m.distance < dist_threshold * n.distance:
                filtered_match.append(m)

    return filtered_match


def filter_matches_dataset(filter_matches_distance, matches):
    filtered_matches = []
    dist_threshold = 0.6
    ### START CODE HERE ###
    for m in matches:
        new_match = filter_matches_distance(m, dist_threshold)
        filtered_matches.append(new_match)

    ### END CODE HERE ###

    return filtered_matches


def visualize_matches(image1, kp1, image2, kp2, match):

    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def estimate_motion(match, kp1, kp2, k):

    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    for m in match:
        #         m = m[0]
        train_idx = m.trainIdx
        query_idx = m.queryIdx

        p1x, p1y = kp1[query_idx].pt
        image1_points.append([p1x, p1y])

        p2x, p2y = kp2[train_idx].pt
        image2_points.append([p2x, p2y])

    E, mask = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), k)

    retval, rmat, tvec, mask = cv2.recoverPose(
        E, np.array(image1_points), np.array(image2_points), k
    )
    ### END CODE HERE ###

    return rmat, tvec, image1_points, image2_points


###uska function
def estimate_trajectory(estimate_motion, matches, kp_list, k, P, depth_maps=[]):
    R = np.diag([1, 1, 1])
    T = np.zeros([3, 1])
    # RT = np.hstack([R, T])
    # RT = np.vstack([RT, np.zeros([1, 4])])
    # RT[-1, -1] = 1

    for i in range(len(matches)):
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i + 1]

        rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k)
        rt_mtx = np.hstack([rmat, tvec])
        rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
        rt_mtx[-1, -1] = 1

        rt_mtx_inv = np.linalg.inv(rt_mtx)

        # RT = np.dot(RT, rt_mtx_inv)
        P = np.dot(P, rt_mtx_inv)

    return P, rmat, tvec