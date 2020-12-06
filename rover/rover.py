# external modules
import socket
import sys
import time
import serial
import os

# internal modules
import sys
sys.path.append("/etc/scout/scout-rover/common/")
from MovementMessage import *
from NetworkHandler import *

class Rover:
    """
        A Class for handling communication with the Server operating
        externally.
    """

    def start(self, ip='192.168.100.14', port=6909):
        """
            Kickstarts the rover.
            - Creates socket
            - Sends connection request to Server
            - Once connected, performs the following in parallel:
                - calls listen()
        """
        try:
            self.bluetooth_port = serial.Serial('/dev/rfcomm0',9600)
            self.rover_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.rover_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rover_socket.connect((ip,port))
            print("sent the request")
            # NetworkHandler().send('Hi !',self.rover_socket)
            self.handle_incoming_messages()
        finally:
            self.cleanUp()

    def handle_incoming_messages(self):
        """
            Listens for messages from the Rover.
        """
        while True:
            message = NetworkHandler().receive(self.rover_socket)
            if message:
                print(message)
                self.bluetooth_port.write(message)
                # s.write(b'w')
                # s.write(b' ')
            else:
                return

    def cleanUp(self):
        """
            Closes Connections.
        """
        print("Cleaning Up")
        if self.rover_socket is not None:
            self.rover_socket.close()
        if self.bluetooth_port is not None:
            self.bluetooth_port.close()

rover = Rover()
rover.start()

