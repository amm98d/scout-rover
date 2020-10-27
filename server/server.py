# external modules
import socket
import threading
from time import sleep

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
            for i in range(5):
                abc = "hi, I'm server"+str(i)
                NetworkHandler().send(abc,self.connection)
                sleep(0.1)
            self.messagesListenerThread.join()
            # self.mainMenuThread = threading.Thread(target=self.mainMenu)
            # self.mainMenuThread.start()
            # self.mainMenuThread.join()
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

    def mainMenu(self):
        while (True):
            abc = input("CHOICE")

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