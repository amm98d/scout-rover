# external modules
import socket
from time import sleep
import os
from platform import platform
import urllib.request
import select
import sys
import numpy as np
import cv2
import io
import zlib
import msvcrt
import threading
import select

# internal modules
import sys
sys.path.append("../common/")
sys.path.append("./slam/")
from NetworkHandler import *
from SlamHandler import *

# global initializations
SERVER_ENV = 'Windows'
np.random.seed(1)

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
            self.slamHandler = SlamHandler(self._getFrame)
            self.measurementsHandlerThread = threading.Thread(target=self._startMeasuring)
            self.slamHandlerThread = threading.Thread(target=self.slamHandler.slamHome)
            self.measurementsHandlerThread.daemon = True
            self.slamHandlerThread.daemon = True
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
                self._initiateExploration()
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

    def _oneMeasurement(self):
        def _uncompress_nparr(bytestring):
            return np.load(io.BytesIO(zlib.decompress(bytestring)))
        rgb_byte_array = urllib.request.urlopen('http://192.168.100.113:5000/rgb').read()
        depth_byte_array = urllib.request.urlopen('http://192.168.100.113:5000/depth').read()
        rgb = _uncompress_nparr(rgb_byte_array)
        depth = _uncompress_nparr(depth_byte_array)
        return (rgb,depth)

    def _startMeasuring(self):
        while(True):
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
        self._clearScreen()
        print("===================================================================")
        print("                         Exploration Mode                          ")
        print("===================================================================")
        print("Instructions:")
        print(" ==> Use the Arrow Keys to drive the rover around.")
        print(" ==> Use Spacebar to break.")
        print(" ==> Press Esc Key to exit.")
        print("-------------------------------------------------------------------")

        try:
            self.measurementsHandlerThread.start()

            # buffering measurements queue
            while (len(self.measurementsQueue)<10):
                pass

            self.slamHandlerThread.start()

            while(True):
                frame = self._getFrame()
                cv2.imshow('rgb', frame[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # checking for user interrupts
                if self._check_interrupts() == False:
                    break
        finally:
            cv2.destroyAllWindows()

server = Server()
server.start()