# external modules
import socket
import sys
import time
import serial
import os
import time
import numpy as np
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import threading

# internal modules
import sys
sys.path.append("/etc/scout/scout-rover/common/")
from MovementMessage import *
from NetworkHandler import *
from NumpySocket import NumpySocket

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
            self.sendStreamThread = threading.Thread(target=self.sendStream)

            self.messagesListenerThread.start()
            self.sendStreamThread.start()

            self.messagesListenerThread.join()
            self.sendStreamThread.join()
        finally:
            self.cleanUp()

    def sendStream(self):
        host_ip = '192.168.100.14'
        npSocket = NumpySocket()
        npSocket.startServer(host_ip, 9999)
        while True:
            frame, _ = get_video()
            time.sleep(0.016)
            npSocket.sendNumpy(frame)
        try:
            npSocket.endServer()
        except OSError as err:
            print("error")

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
        print("Cleaning Up")
        if self.rover_socket is not None:
            self.rover_socket.close()
        if self.bluetooth_port is not None:
            self.bluetooth_port.close()

rover = Rover()
rover.start()

