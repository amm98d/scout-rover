# external modules
import socket
import os
from platform import platform
from time import sleep
import select
import zlib
import msvcrt
import threading
import select
import zlib
import requests
import sys
import cv2

# internal modules
import sys
sys.path.append("../common/")
sys.path.append("./slam/")
from NetworkHandler import *
from SLAM import *

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
            self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setblocking(True)
            self.server_socket.bind(('',port))
            self.server_socket.listen(1)
            self.connection, self.client_address = self.server_socket.accept()
            self.measurementsQueue = []
            self.escaped = False
            self.measurementsHandlerThread = threading.Thread(target=self._startMeasuring)
            self.controlHandlerThread = threading.Thread(target=self._initiateExploration)
            self.controlHandlerThread.daemon = True
            self.measurementsHandlerThread.daemon = True
            self._mainMenu()
        finally:
            self._cleanUp()

    def _clearScreen(self):
        """
            Clears the console. Intelligently recongizes host os between Windows and Linux.
        """
        if platform() == "Windows-10-10.0.19041-SP0":
            os.system("cls")
        else:
            os.system('clear')

    def _cleanUp(self):
        """
            Closes Connections.
        """
        try:
            if self.connection is not None:
                self.connection.close()
            if self.server_socket is not None:
                self.server_socket.close()
        except Exception as e:
            print(e)

    def _mainMenu(self):
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
                self._slamHome()
            elif (choice=="0"):
                self._clearScreen()
                print()
                print("Scout-Rover signing off.")
                print()
                return

    def _check_interrupts(self):
        """Non-Blocking Console Input for three different Server Environments:
            - 'LocalTemp': Blocking Input for development purposes
            - 'Windows': Non-blocking windows input using msvcrt
            - 'Linux': Non-blocking linux input using termios
        Returns:
            bool: False if Esc Key Pressed
        """
        isInput = False
        if msvcrt.kbhit():
            k = msvcrt.getch()
            isInput = True
        if isInput:
            if ord(k) == 119:
                NetworkHandler().send(b'w',self.connection)
                print("up")
            elif ord(k) == 115:
                NetworkHandler().send(b's',self.connection)
                print("down")
            elif ord(k) == 100:
                NetworkHandler().send(b'd',self.connection)
                print("right")
            elif ord(k) == 97:
                NetworkHandler().send(b'a',self.connection)
                print("left")
            elif ord(k) == 32:
                NetworkHandler().send(b' ',self.connection)
                print('brakes')
            elif ord(k) == 27:
                print("Esc")
                return False
        return True

    def _oneMeasurement(self):
        rgb_byte_array = requests.get('http://192.168.100.113:5000/color').content
        depth_byte_array = requests.get('http://192.168.100.113:5000/rgb').content
        # rgb_byte_array = requests.get('http://192.168.43.193:5000/color').content
        # depth_byte_array = requests.get('http://192.168.43.193:5000/rgb').content
        rgb = zlib.decompress(rgb_byte_array)
        depth = zlib.decompress(depth_byte_array)
        rgb = np.reshape(np.frombuffer(rgb, dtype=np.uint8), (480, 640))
        depth = np.reshape(np.frombuffer(depth, dtype=np.uint16), (480, 640))
        return (rgb,depth)

    def _startMeasuring(self):
        while(self.escaped==False):
            if (len(self.measurementsQueue)<100):
                frame_A = self._oneMeasurement()
                self.measurementsQueue.append(frame_A)
                frame_B = self._oneMeasurement()
                self.measurementsQueue.append(frame_B)

    def _getFrame(self):
        if len(self.measurementsQueue) == 1:
            return self.measurementsQueue[0]
        return self.measurementsQueue.pop(0)

    def _initiateExploration(self):
        while(True):
            # checking for user interrupts
            if self._check_interrupts() == False:
                self.escaped = True
                self.measurementsQueue.append((1,1))
                break
            # sleep(0.1)

    def adjust_heading(self):
        print('centroid_x:',self.slamAlgorithm.centroid_x, 'diff:', abs(self.slamAlgorithm.centroid_x - 350))
        for i in range(abs(self.slamAlgorithm.centroid_x - 350)):
            if 350 - self.slamAlgorithm.centroid_x > 0:
                NetworkHandler().send(b'd',self.connection)
            elif 350 - self.slamAlgorithm.centroid_x < 0:
                NetworkHandler().send(b'a',self.connection)

    def move(self):
        print('centroid_y:',self.slamAlgorithm.centroid_y, 'diff:', int(abs(self.slamAlgorithm.centroid_y - 350))*0.1)
        for i in range(int(abs(self.slamAlgorithm.centroid_y - 350)*0.1)):
            NetworkHandler().send(b'w',self.connection)
            sleep(0.1)

    def _slamHome(self):
        self._clearScreen()
        print("===================================================================")
        print("                         Exploration Mode                          ")
        print("===================================================================")
        print("Instructions:")
        print(" ==> Use the Arrow Keys to drive the rover around.")
        print(" ==> Use Spacebar to break.")
        print(" ==> Press Esc Key to exit.")
        print("-------------------------------------------------------------------")

        self.measurementsHandlerThread.start()

        # buffering measurements queue
        while (len(self.measurementsQueue)<2):
            pass

        depthFactor = 1000
        camera_matrix = [[561.93206787, 0, 323.96944442], [ 0, 537.88018799, 249.35236366], [0, 0, 1]]
        dist_coff = [3.64965254e-01, -2.02943943e+00, -1.46113154e-03, 9.97005541e-03, 5.04006892e+00]

        # for i in range(120):
        #     NetworkHandler().send(b'a',self.connection)
        #     print("right")

        img, depth = self._getFrame()
        newImg, newDepth = self._getFrame()
        self.slamAlgorithm = SLAM(img, depth, depthFactor, camera_matrix, dist_coff)
        # self.slamAlgorithm.process([img, newImg], [depth, newDepth], i)

        # self.controlHandlerThread.start()
        # self.controlHandlerThread.join()

        i = 1
        while True:
            newImg, newDepth = self._getFrame()

            if np.isscalar(newImg):
                print("BREAKING")
                break

            self.slamAlgorithm.process([img, newImg], [depth, newDepth], i)
            i += 1

            self.adjust_heading()
            self.move()

            sleep(1)

            img = newImg
            depth = newDepth

        cv2.destroyAllWindows()

server = Server()
server.start()