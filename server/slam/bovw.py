import cv2 as cv
import numpy as np
from Odometry import *


def build_bovw(des_list, matches, dist_threshold=0.7):
    filtered_matches = []
    nbr_words = len(matches)
    visual_words = []

    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < dist_threshold * n.distance:
                filtered_matches.append(m)

                d1 = des_list[1][m.queryIdx]
                d2 = des_list[0][m.trainIdx]
                centroid = calc_centroid(d1, d2)
                visual_words.append(centroid)

    return filtered_matches, visual_words


def calc_centroid(d1, d2):
    desc_len = len(d1)

    new_desc = []
    for i in range(desc_len):
        ith_val = d1[i] | d2[i]
        new_desc.append(ith_val)

    return np.array(new_desc)


def assign_bow_index(vwords, vocabulary, dist_threshold=0.7):
    matches = match_features([vwords, vocabulary])

    indices = set()

    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < dist_threshold * n.distance:
                idx = m.queryIdx
                indices.add(idx)
