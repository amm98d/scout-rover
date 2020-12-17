# external modules
import socket
import threading
from time import sleep
import os
from platform import platform
import numpy as np
import cv2
import urllib.request
import fpstimer
import glob

# internal modules
import sys
sys.path.append("../common/")
from NetworkHandler import *

# from Map import Map
# from Landmark import Landmark
# from Robot import Robot
# from Localization import Localization
# from Odometry import *
# from utils import *

class Server:
    """
        A Class for handling communication with the Raspberry Pi
        physically mounted on the Rover.
    """

    def start(self, port=6909):
        """
            Kickstarts the whole server.
            - Binds the socket
            - Listens for connection
            - Once connected, performs the following in parallel:
                - calls handle_incoming_messages()
                - calls mainMenu()
        """
        try:
            # self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            # self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # self.server_socket.setblocking(True)
            # self.server_socket.bind(('',port))
            # self.server_socket.listen(1)
            # self.connection, self.client_address = self.server_socket.accept()
            self.mainMenu()
        finally:
            pass
            # self.cleanUp()

    def _getFrame(self):
        imgResp = urllib.request.urlopen('http://192.168.100.45:8080/shot.jpg')
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        return img

    def localize(self):
        # initialize frames
        frame_A = self._getFrame()
        frame_B = self._getFrame()
        images = [frame_A, frame_B]

        # initialize localization variables
        ## GLOBAL VARIABLES
        P = np.eye(4)  # Pose
        k = np.array([[923.0, 0.0, 657.0],
                [0.0, 923.0, 657.0],
                [0.0000,   0.0000,   1.0000]], dtype=np.float32)
        
        poses = [[0.0, 0.0, 0.0]]  # initial pose
        ## Create Map
        env_map = Map(62, 62)
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
                env_map.add_landmark(Landmark(crd[i][0], crd[i][1], kp, des))
                i += 1
        particle_filter = Localization(env_map)

        # Setting loop for 10 FPS
        # timer = fpstimer.FPSTimer(60)

        i = 0
        trajectory = [np.array([0, 0, 0])]
        camPoses = []
        # main loop
        while True:
            # Part I. Features Extraction
            kp_list, des_list = extract_features_dataset(images, extract_features)
            # Part II. Feature Matching
            matches = match_features_dataset(des_list, match_features)
            # Set to True if you want to use filtered matches or False otherwise
            is_main_filtered_m = False
            if is_main_filtered_m:
                filtered_matches = filter_matches_dataset(filter_matches_distance, matches)
                matches = filtered_matches
            # Part III. Trajectory Estimation
            P, rmat, tvec = estimate_trajectory(estimate_motion, matches, kp_list, k, P)
            if len(camPoses) == 0:
                camPoses.append((rmat, tvec))
                new_trajectory = P[:3, 3]
                trajectory.append(new_trajectory)
                # poses.extend(pose)
                # trajectory.append(P[:3, 3])
            else:
                prevRmat, prevTvec = camPoses[-1]

                if not np.allclose(rmat, prevRmat) and not np.allclose(tvec, prevTvec):
                    camPoses.append((rmat, tvec))
                    new_trajectory = P[:3, 3]
                    trajectory.append(new_trajectory)
                    # poses.extend(pose)
                    # trajectory.append(P[:3, 3])
                else:
                    print("frame filtered...")
            # Part IV. Localize
            # last_pose = poses[-1]
            # second_last_pose = poses[-2]
            # particle_filter.motion_update([second_last_pose, last_pose])
            # particle_filter.measurement_update(images[1])
            # particle_filter.sample_particles()
            # Part V. Save Visualization plot
            # particle_filter.plot_data(i)
            i += 1
            # Get New Frames
            frame_A = np.copy(frame_B)
            frame_B = self._getFrame()
            images = [frame_A, frame_B]
            # cv2.imshow('RGB',cv2.resize(frame_A, (640, 480)))

            # FPS Settings
            # timer.sleep()

            # Temporary
            q = cv2.waitKey(1)
            if q == ord("q"):
                break
            # Save visualization frames
            plt.cla()
            npTraj = np.array(trajectory).T
            visualize_trajectory(npTraj)
            plt.savefig(f'dataviz/frame{i}.png')
        cv2.destroyAllWindows()
        # npSocket = NumpySocket()
        # npSocket.startClient(9999)
        # for i in range(10):
        #     print(i)
        #     frame = npSocket.recieveNumpy()
        #     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #     cv2.imshow('RGB',frame)
        #     print("received\n")
        # try:
        #     npSocket.endServer()
        # except OSError as err:
        #     print("error",err)

    def _clearScreen(self):
        print(platform())
        if platform() == "Windows":
            os.system("cls")
        elif platform() == "Linux-5.4.0-58-generic-x86_64-with-Ubuntu-18.04-bionic":
            os.system('clear')

    def initiateExploration(self):
        self._clearScreen()
        print("===================================================================")
        print("                           Driving Mode                            ")
        print("===================================================================")
        print("Instructions:")
        print(" ==> Use the Arrow Keys to drive the rover around.")
        print(" ==> Use Spacebar to break.")
        print(" ==> Press Esc Key to exit.")
        print("-------------------------------------------------------------------")
        import sys
        import select
        import tty
        import termios
        def isData():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                # for manual non-blocking user controlled movement of rover
                if isData():
                    k = sys.stdin.read(1)
                    if ord(k) == 119:
                        # NetworkHandler().send(b'w',self.connection)
                        print("up")
                    elif ord(k) == 115:
                        # NetworkHandler().send(b's',self.connection)
                        print("down")
                    elif ord(k) == 100:
                        # NetworkHandler().send(b'd',self.connection)
                        print("right")
                    elif ord(k) == 97:
                        # NetworkHandler().send(b'a',self.connection)
                        print("left")
                    elif ord(k) == 32:
                        # NetworkHandler().send(b' ',self.connection)
                        print('brakes')
                    elif ord(k) == 27:
                        print("Esc")
                        return
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def mainMenu(self):
        while (True):
            self._clearScreen()
            print("=========================================")
            print("                Main Menu                ")
            print("=========================================")
            print("1. Initiate Exploration.")
            print("0. Exit.")
            print("-----------------------------------------")
            choice = input("Your Choice: ")
            if (choice=="1"):
                self.initiateExploration()
            elif (choice=="0"):
                self._clearScreen()
                print()
                print("Scout-Rover signing off.")
                print()
                return

    def cleanUp(self):
        """
            Closes Connections.
        """
        print("Cleaning Up.")
        if self.connection is not None:
            self.connection.close()
        if self.server_socket is not None:
            self.server_socket.close()

server = Server()
server.start()
