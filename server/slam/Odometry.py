import numpy as np
import cv2 as cv
from utils import *


def orb_extractor(image):

    orb = cv.ORB_create(nfeatures=5000, WTA_K=4)
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    return kp, des


def extract_features(images, extract_features_function=orb_extractor):

    kp_list = []
    des_list = []

    # Make sure only 2 images are provided
    assert len(
        images) == 2, f"{len(images)} images provided to extract_features. Required = 2"

    img1, img2 = images

    # Extract keypoints and descriptors of img1
    kp1, des1 = extract_features_function(img1)
    kp_list.append(kp1)
    des_list.append(des1)

    # Extract keypoints and descriptors of img2
    kp2, des2 = extract_features_function(img2)
    kp_list.append(kp2)
    des_list.append(des2)

    return kp_list, des_list


def visualize_features(image, kp):

    display = cv.drawKeypoints(image, kp, None)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)


def flann_matcher(des1, des2):

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)
    match = flann.knnMatch(des1, des2, k=2)

    return match


def match_features(des_list, match_features_function=flann_matcher):

    # Make sure only 2 descriptors are provided
    assert len(
        des_list) == 2, f"{len(des_list)} descriptors provided to match_features. Required = 2"

    # Match descriptors
    des1, des2 = des_list
    matches = match_features_function(des1, des2)

    return matches


def filter_matches(matches):

    filtered_matches = []
    dist_threshold = 0.6

    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < dist_threshold * n.distance:
                filtered_matches.append(m)

    return filtered_matches


def visualize_matches(image1, kp1, image2, kp2, match):

    image_matches = cv.drawMatches(
        image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def pnp_estimation(match, kp1, kp2, k,  depth_map=[]):

    image1_points = []
    image2_points = []
    object_points = []

    for m in match:
        query_idx = m.queryIdx
        train_idx = m.trainIdx

        # get first img matched keypoints
        p1_x, p1_y = kp1[query_idx].pt

        # get second img matched keypoints
        p2_x, p2_y = kp2[train_idx].pt

        p1_z = depth_map[int(p1_y), int(p1_x)]
        if p1_z != 0:
            image1_points.append([p1_x, p1_y])
            image2_points.append([p2_x, p2_y])
            # Convert to object points
            p1_z /= 5000
            w1_x = (p1_x - k[0][2]) * p1_z / k[0][0]
            w1_y = (p1_y - k[1][2]) * p1_z / k[1][1]
            object_points.append([w1_x, w1_y, p1_z])

    object_points = np.array(object_points).astype('double')
    image2_points = np.array(image2_points).astype('double')
    _, rvec, tvec, _ = cv.solvePnPRansac(
        object_points, image2_points, k, None, flags=cv.SOLVEPNP_EPNP)

    rmat, _ = cv.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points


def em_estimation(match, kp1, kp2, k):

    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    for m in match:
        train_idx = m.trainIdx
        query_idx = m.queryIdx

        p1x, p1y = kp1[query_idx].pt
        image1_points.append([p1x, p1y])

        p2x, p2y = kp2[train_idx].pt
        image2_points.append([p2x, p2y])

    if len(image2_points) < 5:
        print(f"motion estimation: IMAGE POINTS LESS THAN 5")
        return -1, -1, -1, -1
    E, _ = cv.findEssentialMat(
        np.array(image1_points), np.array(image2_points), k)

    _, rmat, tvec, _ = cv.recoverPose(
        E, np.array(image1_points), np.array(image2_points), k
    )

    return rmat, tvec, image1_points, image2_points


# def estimate_trajectory(estimate_motion, matches, kp_list, k, P, depth_map=[]):

#     for i in range(len(matches)):
#         match = matches[i]
#         kp1 = kp_list[i]
#         kp2 = kp_list[i + 1]

#         rmat, tvec, _, _ = estimate_motion(
#             match, kp1, kp2, k)
#         if np.isscalar(rmat):
#             print("estimate trajectory: NO RMAT, TVEC")
#             return P, -1, -1
#         rt_mtx = np.hstack([rmat, tvec])
#         rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
#         rt_mtx[-1, -1] = 1

#         rt_mtx_inv = np.linalg.inv(rt_mtx)

#         P = np.dot(P, rt_mtx_inv)

#     return P, rmat, tvec

def estimate_trajectory(matches, kp_list, k, P, depth_map):

    kp1 = kp_list[0]
    kp2 = kp_list[1]

    if depth_map:
        rmat, tvec, _, _ = pnp_estimation(
            matches, kp1, kp2, k, depth_map)
    else:
        rmat, tvec, _, _ = em_estimation(
            matches, kp1, kp2, k)

    if np.isscalar(rmat):
        print("estimate trajectory: NO RMAT, TVEC")
        return P, -1, -1

    rt_mtx = np.hstack([rmat, tvec])
    rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
    rt_mtx[-1, -1] = 1

    rt_mtx_inv = np.linalg.inv(rt_mtx)

    P = np.dot(P, rt_mtx_inv)

    return P, rmat, tvec
