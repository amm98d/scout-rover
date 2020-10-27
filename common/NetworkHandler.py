import pickle

class NetworkHandler:
    """
        Common Static Class for Network Communication between Server and Rover.
    """

    def send(self, message_object, socket):
        """
            Generic method for sending messages. Serializes the message_object by using Pickle.

            Args:
                message_object (object): any object-type can be sent
                socket (Socket): socket for sending
        """
        socket.sendall(pickle.dumps(message_object))

    def receive(self, socket):
        """
            Generic method for receiving messages. Deserializes the received by using Pickle.

            Args:
                socket (Socket): socket for receiving

            Returns:
                object: deserialized object
        """
        BUFF_SIZE = 4096
        data = b''
        while True:
            part = socket.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                break
        return pickle.loads(data) if len(data) is not 0 else None

