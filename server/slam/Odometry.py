import numpy as np
import cv2 as cv
from utils import *


def orb_extractor(image):

    orb = cv.ORB_create(nfeatures=1000, WTA_K=4)
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
    dist_threshold = 0.7

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


def visualize_camera_movement(
    image1, image1_points, image2, image2_points, is_show_img_after_move=False
):
    image1 = image1.copy()
    image2 = image2.copy()

    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv.circle(image1, p1, 5, (0, 255, 0), 1)
        cv.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv.circle(image2, p2, 5, (255, 0, 0), 1)

    if is_show_img_after_move:
        return image2
    else:
        return image1


def pnp_estimation(match, kp1, kp2, k, dist_coff, depth_map, depth_factor):

    image1_points = []
    image2_points = []
    cloud1_points = []
    cloud2_points = []
    used_matches = []

    for m in match:
        query_idx = m.queryIdx
        train_idx = m.trainIdx

        # get first img matched keypoints
        p1_x, p1_y = kp1[query_idx].pt

        # get second img matched keypoints
        p2_x, p2_y = kp2[train_idx].pt

        p1_z = depth_map[int(p1_y), int(p1_x)]
        p2_z = depth_map[int(p2_y), int(p2_x)]
        w1_x, w1_y, w1_z = point2Dto3D((p1_x, p1_y), p1_z, k, depth_factor)
        w2_x, w2_y, w2_z = point2Dto3D((p2_x, p2_y), p2_z, k, depth_factor)
        if w1_z > 0 and w1_z < 1000:
            used_matches.append(m)
            image1_points.append([p1_x, p1_y])
            image2_points.append([p2_x, p2_y])
            cloud1_points.append([w1_x, w1_y, w1_z])
            cloud2_points.append([w2_x, w2_y, w2_z])

    cloud1_points = np.array(cloud1_points).astype('double')
    cloud2_points = np.array(cloud2_points).astype('double')
    image1_points = np.array(image1_points).astype('double')
    image2_points = np.array(image2_points).astype('double')

    _, rvec, tvec, inliers = cv.solvePnPRansac(
        cloud1_points, image2_points, k, dist_coff, flags=cv.SOLVEPNP_EPNP)

    inliers = inliers.flatten()
    used_matches = np.take(used_matches, inliers, 0)
    image1_points = np.take(image1_points, inliers, 0)
    image2_points = np.take(image2_points, inliers, 0)
    cloud1_points = np.take(cloud1_points, inliers, 0)
    cloud2_points = np.take(cloud2_points, inliers, 0)

    rmat, _ = cv.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points, cloud1_points, cloud2_points, used_matches, len(inliers)


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

def estimate_trajectory(matches, kp_list, k, dist_coff, P, depth_map, depth_factor):

    kp1 = kp_list[0]
    kp2 = kp_list[1]

    if not np.isscalar(depth_map):
        rmat, tvec, image1_points, image2_points, cloud1_points, cloud2_points, used_matches, inliersCount = pnp_estimation(
            matches, kp1, kp2, k, dist_coff, depth_map, depth_factor)
    else:
        rmat, tvec, image1_points, image2_points = em_estimation(
            matches, kp1, kp2, k)

    if np.isscalar(rmat):
        print("estimate trajectory: NO RMAT, TVEC")
        return P, -1, -1

    rt_mtx = np.hstack([rmat, tvec])
    rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
    rt_mtx[-1, -1] = 1

    rt_mtx_inv = np.linalg.inv(rt_mtx)

    P = np.dot(P, rt_mtx_inv)

    return P, rmat, tvec, image1_points, image2_points, cloud1_points, cloud2_points, used_matches, inliersCount
