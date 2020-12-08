# external modules
import socket
import threading
from time import sleep
import os
from platform import platform
import numpy as np
import cv2

# internal modules
import sys
sys.path.append("../common/")
from MovementMessage import *
from NetworkHandler import *
from NumpySocket import NumpySocket

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

            self.messagesListenerThread = threading.Thread(target=self.handle_incoming_messages)
            self.receiveStreamThread = threading.Thread(target=self.receiveStream)
            self.mainMenuThread = threading.Thread(target=self.mainMenu)

            self.messagesListenerThread.start()
            self.receiveStreamThread.start()
            self.mainMenuThread.start()

            self.mainMenuThread.join()
            self.messagesListenerThread.join()
            self.receiveStreamThread.join()
        finally:
            self.cleanUp()

    def receiveStream(self):
        npSocket = NumpySocket()
        npSocket.startClient(9999)
        while True:
            frame = npSocket.recieveNumpy()
            print(len(frame))
            #cv2.imshow("The display window" , frame)
        try:
            npSocket.endServer()
        except OSError as err:
            print("error",err)

    def handle_incoming_messages(self):
        """
            Listens for messages from the Rover.
        """
        while True:
            message = NetworkHandler().receive(self.connection)
            if message:
                print(message)
            else:
                return

    def clearScreen(self):
        if platform() == "Windows":
            os.system("cls")
        elif platform() == "Linux-5.4.0-52-generic-x86_64-with-Ubuntu-18.04-bionic":
            os.system('clear')

    def drivingMode(self):
        def get_raw_char():
            ch = ''
            if os.name == 'nt':
                import msvcrt
                ch = msvcrt.getch()
            else:
                import tty, termios, sys
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        self.clearScreen()
        print("===================================================================")
        print("                           Driving Mode                            ")
        print("===================================================================")
        print("Instructions:")
        print("Use the arrow keys to drive the rover around. Press Esc Key to exit.")
        print("-------------------------------------------------------------------")
        while (True):
            k = get_raw_char()
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
                return

    def mainMenu(self):
        while (True):
            self.clearScreen()
            print("=========================================")
            print("                Main Menu                ")
            print("=========================================")
            print("1. Driving Mode.")
            print("0. Exit.")
            print("-----------------------------------------")
            choice = input("Your Choice: ")
            if (choice=="1"):
                self.drivingMode()
            elif (choice=="0"):
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