import numpy as np
from Robot import Robot
from utils import *


class Localization:
    def __init__(self, env_map, num_particles=1000):
        self.env_map = env_map
        self.num_particles = num_particles
        self.generate_particles()

    def generate_particles(self):
        self.particles = []
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        for _ in range(self.num_particles):
            # Generate random pose
            rx = np.random.random() * self.env_map.width
            ry = np.random.random() * self.env_map.height
            rtheta = np.random.random() * 2.0 * np.pi

            # Add robot with generated pose
            r = Robot([rx, ry, rtheta])
            self.particles.append(r)

    def motion_update(self, odom_data):
        for pid, particle in enumerate(self.particles):
            new_pose = particle.move(odom_data)
            x, y, _ = new_pose

            # Remove Particles that go outside the box
            if x < 0 or x >= self.env_map.width or y < 0 or y >= self.env_map.height:
                self.weights[pid] = 0

    def measurement_update(self, frame):
        # Detect Landmark
        l_idx = self.particles[0].measure(frame, self.env_map.landmarks)

        # If landmark exists
        if l_idx > -1:
            landmark = self.env_map.landmarks[l_idx]
            # print(f"Landmark {l_idx}: ({landmark.x}, {landmark.y})")
            for i in range(self.num_particles):
                self.weights[i] = self.particles[i].measurement_prob(landmark)

        # Avoid all-zero weights
        if np.sum(self.weights) == 0:
            self.weights = np.ones_like(self.weights)
        self.weights /= np.sum(self.weights)

    def sample_particles(self):
        # Resample according to importance weights
        self.particles = np.random.choice(
            self.particles, self.num_particles, True, self.weights
        )

    def plot_data(self, frameIdx=-1):
        plt.cla()
        poses = extract_poses(self.particles)
        plot_poses(poses)
        # Plot boundaries
        plt.plot([0, 0], [0, 62])
        plt.plot([62, 62], [0, 62])
        plt.plot([0, 62], [0, 0])
        plt.plot([0, 62], [62, 62])
        plt.savefig(f"dataviz/frame{frameIdx}.png")
        # plt.show()


"""
Flow of Events:
    1. Generate Robot Pose
    2. Generate Particle Filters
    3. Make a Move
    4. Estimate Motion (estimate_motionEM)
    5. Sample Motion (sample_motion)
    6. Detect Landmarks (sense_measurements)
    7. Check measurment prob (measurement_prob)
    8. Sample Particles
    9. Repeat from Step 3
"""
