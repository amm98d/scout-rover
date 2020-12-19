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
            print(f"=> Detected Landmark {l_idx} at ({landmark.x}, {landmark.y})")
            for i in range(self.num_particles):
                if self.weights[i] > 0:
                    self.weights[i] = self.particles[i].measurement_prob(landmark)

        # Avoid all-zero weights
        if np.sum(self.weights) == 0:
            print("All particles are bad! Resampling...")
            # self.weights = np.ones_like(self.weights)
            self.generate_particles()
        # else:
        #     self.weights /= np.sum(self.weights)

    def sample_particles(self):
        # Resample according to importance weights
        new_idexes = np.random.choice(
            len(self.particles), len(self.particles), True, self.weights
        )

        # Update new particles
        self.particles = [self.particles[i] for i in new_idexes]
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def plot_particles(self):
        # x = [p.x for p in self.particles]
        # y = [p.y for p in self.particles]
        # weights = [w * 100 * self.num_particles for w in self.weights]
        # plt.scatter(x, y, s=weights)
        for pid, p in enumerate(self.particles):
            x = p.x
            y = p.y
            t = p.theta
            plt.plot(x, y, "bo", alpha=0.3)
            plt.arrow(x, y, np.cos(t), np.sin(t))
