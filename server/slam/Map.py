class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.landmarks = []

    def add_landmark(self, landmark):
        self.landmarks.append(landmark)
