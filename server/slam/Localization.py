import numpy as np
from Robot import Robot
from utils import *


class Localization:
    def __init__(self, env_map, num_particles=500):
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

    def measurement_update(self, frame, img_kp, img_des, iterator):
        # Detect Landmark
        l_idx, l_px = self.particles[0].measure(
            frame, self.env_map.landmarks, img_kp, img_des, iterator
        )
        # If landmark exists
        if l_idx > -1:
            landmark = self.env_map.landmarks[l_idx]
            # Calc distance from landmark
            focal_length = 827.0
            dist = landmark.height * focal_length / l_px
            print(
                f"=> Detected Landmark {l_idx}: ({landmark.x}, {landmark.y})\tDist: {dist}"
            )

            # Send all landmarks with calculated distance
            for i in range(self.num_particles):
                if self.weights[i] > 0:
                    # print(f"For Robot: {self.particles[i].get_pose()}")
                    self.weights[i] = self.particles[i].measurement_prob(
                        self.env_map.landmarks, dist
                    )

    def sample_particles(self):
        # Avoid all-zero weights
        if np.sum(self.weights) == 0:
            print("All particles are bad! Resampling...")
            self.generate_particles()

        # Normalize weights
        self.weights /= sum(self.weights)

        # Resample according to importance weights
        new_indices = np.random.choice(
            len(self.particles), len(self.particles), True, self.weights
        )

        # Update new particles
        # print(f"Sampled: {new_indices}")
        new_particles = []
        for i in new_indices:
            r_pose = self.particles[i].get_pose()
            new_particles.append(Robot(r_pose))
        self.particles = new_particles[:]
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
            x2 = x + round(np.cos(t), 2)
            y2 = y + round(np.sin(t), 2)
            plt.plot(x, y, "bo", alpha=0.3)
            plt.plot([x, x2], [y, y2], "k")
