# external modules
import socket
from time import sleep
import os
from platform import platform
import urllib.request
# import fpstimer
import select
import sys
# import tty
# import termios
import numpy as np
import cv2
import io
import zlib
import msvcrt
import threading

# internal modules
import sys
sys.path.append("../common/")
sys.path.append("./slam/")
from NetworkHandler import *
from SLAM import *

# global variables
SERVER_ENV = 'Windows'

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
            # self.slamAlgorithm = SLAM()
            self.mainMenu()
        finally:
            self.cleanUp()

    # for non-blocking user controlled movement of rover
    def _check_interrupts(self):
        """Non-Blocking Console Input for three different Server Environments:
            - 'LocalTemp': Blocking Input for development purposes
            - 'Windows': Non-blocking windows input using msvcrt
            - 'Linux': Non-blocking linux input using termios
        Returns:
            bool: False if Esc Key Pressed
        """
        isInput = False
        if SERVER_ENV=='LocalTemp':
            k = input("Command")
            isInput = True
        elif SERVER_ENV=='Linux':
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                k = sys.stdin.read(1)
                isInput = True
        elif SERVER_ENV=='Windows':
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

    def _uncompress_nparr(self, bytestring):
        return np.load(io.BytesIO(zlib.decompress(bytestring)))

    def _takeMeasurements(self):
        rgb_byte_array = urllib.request.urlopen('http://192.168.100.113:5000/rgb').read()
        depth_byte_array = urllib.request.urlopen('http://192.168.100.113:5000/depth').read()
        rgb = self._uncompress_nparr(rgb_byte_array)
        depth = self._uncompress_nparr(depth_byte_array)
        return (rgb,depth)

    def _startMeasuring(self):
        while(True):
            if (len(self.measurementsQueue)<100):
                frame_A = self._takeMeasurements()
                self.measurementsQueue.append(frame_A)
                frame_B = self._takeMeasurements()
                self.measurementsQueue.append(frame_B)

    def _getFrame(self):
        if len(self.measurementsQueue) == 1:
            return self.measurementsQueue[0]
        return self.measurementsQueue.pop(0)

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

        if SERVER_ENV=='Linux':
            old_settings = termios.tcgetattr(sys.stdin)
        try:
            if SERVER_ENV=='Linux':
                tty.setcbreak(sys.stdin.fileno())

            # Take Initialize Measurements
            # frame_A = self._takeMeasurements()
            # frame_B = self._takeMeasurements()
            # images = [frame_A, frame_B]

            self.measurementsHandlerThread = threading.Thread(target=self._startMeasuring)
            self.measurementsHandlerThread.daemon = True
            self.measurementsHandlerThread.start()

            # wait until first two images are loaded atleast
            while (len(self.measurementsQueue)<2):
                pass

            print("IMAGES BUFFERED. STARTING!")

            # timer = fpstimer.FPSTimer(60)
            while(True):
                frame = self._getFrame()
                print(type(frame[0]))
                print(len(frame[0]))
                cv2.imshow('rgb', frame[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # gray = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
                # cv2.imshow('frame', gray)

                # checking for user interrupts
                if SERVER_ENV=='Linux':
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if self._check_interrupts() == False:
                    break

                ## SLAMMING
                # self.slamAlgorithm.process(images)

                # Update Measurements
                # frame_A = np.copy(frame_B)
                # frame_B = self._takeMeasurements()
                # images = [frame_A, frame_B]

                # FPS Settings
                # timer.sleep()

            # self.messagesListenerThread.join()

        finally:
            pass
            cv2.destroyAllWindows()
            if SERVER_ENV=='Linux':
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _clearScreen(self):
        """
            Clears the console. Intelligently recongizes host os between Windows and Linux.
        """
        if platform() == "Windows-10-10.0.19041-SP0":
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