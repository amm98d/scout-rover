# external modules
import socket
from time import sleep
import os
from platform import platform
import numpy as np
import cv2
import urllib.request
import fpstimer
import select
import sys
import tty
import termios

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
            self.slamAlgorithm = SLAM()
            self.mainMenu()
        finally:
            self.cleanUp()

    # for non-blocking user controlled movement of rover
    def _check_interrupts(self):
        k = input("Command")
        if k == 'w':
            NetworkHandler().send(b'w',self.connection)
            print("up")
        elif k == 's':
            NetworkHandler().send(b's',self.connection)
            print("down")
        elif k == 'd':
            NetworkHandler().send(b'd',self.connection)
            print("right")
        elif k == 'a':
            NetworkHandler().send(b'a',self.connection)
            print("left")
        elif k == 'x':
            NetworkHandler().send(b' ',self.connection)
            print('brakes')
        if k == 'q':
            print("Esc")
            return False
        else:
            return True
        # if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        #     k = sys.stdin.read(1)
        #     if ord(k) == 119:
        #         NetworkHandler().send(b'w',self.connection)
        #         print("up")
        #     elif ord(k) == 115:
        #         NetworkHandler().send(b's',self.connection)
        #         print("down")
        #     elif ord(k) == 100:
        #         NetworkHandler().send(b'd',self.connection)
        #         print("right")
        #     elif ord(k) == 97:
        #         NetworkHandler().send(b'a',self.connection)
        #         print("left")
        #     elif ord(k) == 32:
        #         NetworkHandler().send(b' ',self.connection)
        #         print('brakes')
        #     elif ord(k) == 27:
        #         print("Esc")
        #         return False
        # return True

    def _takeMeasurements(self):
        imgResp = urllib.request.urlopen('http://192.168.100.45:8080/shot.jpg')
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        return img

    def initiateExploration(self):
        self._clearScreen()
        print("===================================================================")
        print("                         Exploration Mode                          ")
        print("===================================================================")
        print("Instructions:")
        print(" ==> Use the Arrow Keys to drive the rover around.")
        print(" ==> Use Spacebar to break.")
        print(" ==> Press Esc Key to exit.")
        print("-------------------------------------------------------------------")

        # old_settings = termios.tcgetattr(sys.stdin)
        try:
            # tty.setcbreak(sys.stdin.fileno())

            # Take Initialize Measurements
            # frame_A = self._takeMeasurements()
            # frame_B = self._takeMeasurements()
            # images = [frame_A, frame_B]

            # timer = fpstimer.FPSTimer(60)
            print("here")
            while True:

                # checking for user interrupts
                if self._check_interrupts() == False:
                    break

                # SLAMMING
                self.slamAlgorithm.process(images)

                # Update Measurements
                # frame_A = np.copy(frame_B)
                # frame_B = self._takeMeasurements()
                # images = [frame_A, frame_B]

                # FPS Settings
                # timer.sleep()

        finally:
            cv2.destroyAllWindows()
            # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _clearScreen(self):
        print(platform())
        if platform() == "Windows":
            os.system("cls")
        else:
            os.system('clear')

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
        try:
            if self.connection is not None:
                self.connection.close()
            if self.server_socket is not None:
                self.server_socket.close()
        except Exception as e:
            print(e)

server = Server()
server.start()
