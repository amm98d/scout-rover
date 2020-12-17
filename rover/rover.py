# external modules
import socket
import sys
import time
import serial
import os
import time
import numpy as np
import threading

# internal modules
import sys
sys.path.append("/etc/scout/scout-rover/common/")
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
            self.rover_socket.connect((ip,port))

            self.messagesListenerThread = threading.Thread(target=self.handle_incoming_messages)
            self.messagesListenerThread.start()
            self.messagesListenerThread.join()
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
            else:
                return

    def cleanUp(self):
        """
            Closes Connections.
        """
        try:
            if self.rover_socket is not None:
                self.rover_socket.close()
            if self.bluetooth_port is not None:
                self.bluetooth_port.close()
        except Exception as e:
            print(e)

rover = Rover()
rover.start()

