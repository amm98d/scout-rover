import math
import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os


def calc_robot_pose(rmat, tvec, showAllDOF=False):
    """This function encodes the given rotation matrix and translational vector into a robot pose [x, y, theta].
    Arguments:
    ----------
        rmat: 2d array of shape (3, 3) which is the rotation matrix
        tvec: 2d array of shape (3, 1) which is the translational vector
    Returns:
    --------
        robot_pose: Robot pose corresponding to the rmat and tvec
    """
    # Decomposing coordinates from tvec
    x = tvec[0]
    y = tvec[1]  # Ignored in 3 DOF
    z = tvec[2]

    # Calculating Eular angles from rmat
    thetaX = np.arctan2(rmat[2][1], rmat[2][2])  # Ignored in 3 DOF
    thetaY = np.arctan2(
        -rmat[2][0], np.sqrt(np.square(rmat[2][1]) + np.square(rmat[2][2]))
    )
    thetaZ = np.arctan2(rmat[1][0], rmat[0][0])  # Ignored in 3 DOF

    # if showAllDOF:
    #     print(f"RMat:\n\t{rmat}")
    #     print(f"Coordinates:\n\t{[round(x, 5), round(y, 5), round(z, 5)]}")
    #     print(f"Angles:\n\t{[round(thetaX, 5), round(thetaY, 5), round(thetaZ, 5)]}")
    return [int(x), int(z), round(thetaY, 5)]


# def extract_poses(particles):
#     """
#     From an array of robot objects (particles), this function extracts their poses and returns in the format [x, y, theta] for each particle in a list
#     """
#     poses = []

#     for particle in particles:
#         pose = [particle.x, particle.y, particle.theta]
#         poses.append(pose)

#     return poses


def plot_robot_poses(poses):
    plt.xlim([-10, 70])
    plt.ylim([-10, 70])
    for [x, y, t] in poses:
        x2 = x + round(np.cos(t), 2)
        y2 = y + round(np.sin(t), 2)
        plt.plot(x, y, "bo", alpha=0.3)
        plt.plot([x, x2], [y, y2], "k")


def visualize_data(
    viz_ftn, clean_start=True, showPlot=True, figName="", *args, **kwargs
):
    fig = plt.gcf()  # Copy plot for showing and saving simultaneously
    if clean_start:
        plt.cla()
    viz_ftn(*args)
    if showPlot:
        plt.show()
    if figName != "":
        fig.savefig(f"dataviz/{figName}.png")


def visualize_trajectory(trajectory):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    max = -math.inf
    min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]

        locX.append(current_pos.item(0))
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        if np.amax(current_pos) > max:
            max = np.amax(current_pos)
        if np.amin(current_pos) < min:
            min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    if max > 0 and min > 0:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2
    elif max < 0 and min < 0:
        minY = auxY_line + (min - max) / 2
        maxY = auxY_line - (min - max) / 2
    else:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection="3d")

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(
        locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4
    )
    traj_main_plt.set_xlabel("Z")
    # traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot(
        [locZ[0], locZ[-1]],
        [locX[0], locX[-1]],
        "--",
        label="Auxiliary line",
        zorder=0,
        linewidth=1,
    )
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red",
                          label="Start location", zorder=2)
    traj_main_plt.set_xlim([min, max])
    traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(
        loc=1, title="Legend", borderaxespad=0.0, fontsize="medium", frameon=True
    )

    # Plot ZY
    # ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.plot(
        [locZ[0], locZ[-1]],
        [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2],
        "--",
        linewidth=1,
        zorder=1,
    )
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min, max])
    ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    # YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.plot(
        [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2],
        [locX[0], locX[-1]],
        "--",
        linewidth=1,
        zorder=1,
    )
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min, max])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=toffset)
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min, max)
    D3_plt.set_ylim3d(min, max)
    D3_plt.set_zlim3d(min, max)
    D3_plt.tick_params(direction="out", pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)

    # plt.axis('equal')
    D3_plt.view_init(45, azim=30)
    plt.tight_layout()


#####################
# LANDMARK UTILS
#####################
def applyTranformations(src_img, frameNumber="no"):

    destPath = "landmarksDetected"

    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
    src_img_hsv = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
    org = src_img_hsv
    yellow_lower = np.array([80, 100, 100])
    yellow_upper = np.array([100, 255, 255])

    mask = cv.inRange(src_img_hsv, yellow_lower, yellow_upper)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    kernel1 = np.ones((2, 2), np.uint8)
    dilation = cv.dilate(closing, kernel1, iterations=1)

    # OPENCV-VERSIONS DIFFERENCE
    # _, contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cross-version approach (https://stackoverflow.com/a/48292371/11395861)
    contours, _ = cv.findContours(
        dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]

    if len(contours) < 1:
        if frameNumber != "no":
            des = "./" + destPath + "/" + frameNumber + ".jpg"
            cv.imwrite(des, cv.cvtColor(org, cv.COLOR_HSV2RGB))
        return -1, -1  # no rectangle exist

    c1 = max(contours, key=cv.contourArea)

    rect = cv.minAreaRect(c1)
    X, Y, W, H = cv.boundingRect(c1)
    cropped = org[Y: Y + H, X: X + W]

    h, s, v1 = cv.split(cropped)

    box = cv.boxPoints(rect)
    box = np.int0(box)
    bdg_rect = cv.drawContours(org, [box], 0, (0, 255, 255), 7)
    tmpBdgRect = cv.cvtColor(bdg_rect, cv.COLOR_HSV2RGB)
    # have to get the frame number here
    if frameNumber != "no":
        des = "./" + destPath + "/" + frameNumber + ".jpg"
        cv.imwrite(des, cv.cvtColor(org, cv.COLOR_HSV2RGB))

    return v1, H


def is_frame_blur(image, threshold):
    fm = cv.Laplacian(image, cv.CV_64F).var()
    if fm < threshold:
        return True
    return False


def drop_frame(grey_imgs):
    prevImg, currImg = grey_imgs

    isBlur = is_frame_blur(currImg, 100)

    # sharpening images
    # if(isBlue==True):
    #     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #     im = cv.filter2D(img, -1, kernel)

    if isBlur:
        print(f"\t->Frame Filtered because isBlur: {isBlur}")
        return True
    return False


def printFrame(img, name):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def point2Dto3D(point, depth, k, depth_factor):
    px, py = point
    depth /= depth_factor
    wx = (px - k[0][2]) * depth / k[0][0]
    wy = (py - k[1][2]) * depth / k[1][1]
    return (wx, wy, depth)
