from matplotlib import pyplot as plt


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.landmarks = []

    def add_landmark(self, landmark):
        self.landmarks.append(landmark)

    def plot_map(self):
        # Plot boundaries
        plt.plot([0, 0], [0, self.height], "k--", linewidth=2)  # Left wall
        plt.plot(
            [self.width, self.width], [0, self.height], "k--", linewidth=2
        )  # Right wall
        plt.plot([0, self.width], [0, 0], "k--", linewidth=2)  # Bottom wall
        plt.plot(
            [0, self.width], [self.height, self.height], "k--", linewidth=2
        )  # Top wall

        # Plot landmarks
        for l in self.landmarks:
            plt.plot(l.x, l.y, "r*", markersize=16)

