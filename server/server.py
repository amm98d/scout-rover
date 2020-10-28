# external modules
import socket
import threading
from time import sleep
import os
from platform import platform

# internal modules
import sys
sys.path.append("../common/")
from MovementMessage import *
from NetworkHandler import *

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
            self.messagesListenerThread.start()
            self.mainMenuThread = threading.Thread(target=self.mainMenu)
            self.mainMenuThread.start()
            self.mainMenuThread.join()
            self.messagesListenerThread.join()
        finally:
            self.cleanUp()

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
                    ch = sys.stdin.read(3)
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
            if k == "\x1b[A":
                NetworkHandler().send(b'w',self.connection)
                print("up")
            elif k == '\x1b[B':
                NetworkHandler().send(b's',self.connection)
                print("down")
            elif k == '\x1b[C':
                NetworkHandler().send(b'd',self.connection)
                print("right")
            elif k == '\x1b[D':
                NetworkHandler().send(b'a',self.connection)
                print("left")
            # if k == b'\x1b':
            #     print("Esc")

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