# external modules
import socket

# internal modules
import sys
sys.path.append("../common/")
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
            self.rover_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.rover_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rover_socket.connect((ip,port))
            NetworkHandler().send('Hi !',self.rover_socket)
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
            else:
                return

    def cleanUp(self):
        """
            Closes Connections.
        """
        print("Cleaning Up")
        if self.rover_socket is not None:
            self.rover_socket.close()

rover = Rover()
rover.start()

# import sys
# import time
# import serial
# import os
# s = serial.Serial('/dev/rfcomm0',9600)
# print("writing")
# for i in range(5):
#     s.write(b'w')
#     time.sleep(0.5)
#     s.write(b' ')
#     time.sleep(0.5)
# s.close()
# print("done")