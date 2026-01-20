import numpy as np
import matplotlib.pyplot as plt


#parameters.py 

#Physical Parameters from Helbing et al. 2000)

mass = 80.0                # kg
tau = 0.005                  # relaxation time (s)
a = 2000.0                # social force strength (N)
b = 0.08                   # social force range (m)

#The paper introduces a body compression (repulsive) force that prevents pedestrians from occupying the same space
#There is a body compression when people are pressed against each other
k_body = 1.2e5             # body force constant 

#This variable introduces a tangential friction between bodies when they touch: prevents people
#sliding past each other easily
k_friction = 2.4e5         # sliding friction constant

# Desired speed
v_desired = 50            # m/s (baseline, non-panic)

# Geometry
pedestrian_radius_range = (0.25, 0.35)  # meters

# Time stepping
dt = 0.001
total_time = 60

ROOM_WIDTH = 15
ROOM_HEIGHT = 15

EXIT_X = 15.0
EXIT_Y_MIN = 6.5
EXIT_Y_MAX = 8.5


class Pedestrian:
    def __init__(self, position, velocity, desired_location, radius):
        self.position = position #tuple
        self.velocity = velocity #tuple
        self.desired_direction = [0.0, 0.0]
        self.desired_location = desired_location
        self.radius = radius
        self.exited = False
        self.force = np.zeros(2)


    def step(self, force):
        # Update velocity and position based on the force
        acceleration = force / mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def desired_force(self):
        desired_velocity = v_desired * self.desired_direction
        #print("Desired Velocity:", desired_velocity)
        return (desired_velocity - self.velocity) / tau

    def pedestrian_force(self, p_j):

        # Create a distance vector between pedestrians
        r_ij = self.position - p_j.position
        d_ij = np.linalg.norm(r_ij)
        n_ij = r_ij / d_ij
        overlap = self.radius + p_j.radius - d_ij

    # Social repulsion (always active)
        force = a * np.exp(overlap / b) * n_ij

        if overlap > 0:
            # Body force
            force += k_body * overlap * n_ij

            # Sliding friction
            t_ij = np.array([-n_ij[1], n_ij[0]])
            delta_v = np.dot(p_j.velocity - self.velocity, t_ij)
            force += k_friction * overlap * delta_v * t_ij

        return force
    
    def wall_force(self, wall_start, wall_end):
        force = np.zeros(2)
        # --- Left wall (x = 0)
        d = self.position[0] - self.radius
        if d < 0:
            n = np.array([1.0, 0.0])
            force += a * np.exp(-d / b) * n
            force += k_body * (-d) * n
            force -= k_friction * (-d) * self.velocity

        # --- Right wall (x = ROOM_WIDTH), except exit
        if not (EXIT_Y_MIN <= self.position[1] <= EXIT_Y_MAX):
            d = ROOM_WIDTH - self.position[0] - self.radius
            if d < 0:
                n = np.array([-1.0, 0.0])
                force += a * np.exp(-d / b) * n
                force += k_body * (-d) * n
                force -= k_friction * (-d) * self.velocity

        # --- Bottom wall (y = 0)
        d = self.position[1] - self.radius
        if d < 0:
            n = np.array([0.0, 1.0])
            force += a * np.exp(-d / b) * n
            force += k_body * (-d) * n
            force -= k_friction * (-d) * self.velocity

        # --- Top wall (y = ROOM_HEIGHT)
        d = ROOM_HEIGHT - self.position[1] - self.radius
        if d < 0:
            n = np.array([0.0, -1.0])
            force += a * np.exp(-d / b) * n
            force += k_body * (-d) * n
            force -= k_friction * (-d) * self.velocity

        return force
    
    def location_updater(self):
        self.desired_direction[0] = self.desired_location[0] - self.position[0]
        self.desired_direction[1] = self.desired_location[1] - self.position[1]
        norm = np.linalg.norm(self.desired_direction)
        if norm > 0:
            self.desired_direction /= norm

    def force_summer(self, pedestrians):
        total_force = self.desired_force()

        for other in pedestrians:
            if other != self and not other.exited:
                total_force += self.pedestrian_force(other)

        total_force += self.wall_force(None, None)

        #print("Total Force on Pedestrian at", self.position, ":", total_force)
        self.force = total_force


class Simulator:
    def __init__(self, num_pedestrians):
        self.pedestrians = []
        self.time = 0.0
        self.num_pedestrians = num_pedestrians
    
    def initialize_pedestrians(self):
        for _ in range(self.num_pedestrians):
            x = np.random.uniform(1.0, ROOM_WIDTH - 1.0)
            y = np.random.uniform(1.0, ROOM_HEIGHT - 1.0)
            position = np.array([x, y])
            velocity = np.array([0.0, 0.0])
            desired_location = np.array([EXIT_X, np.random.uniform(EXIT_Y_MIN, EXIT_Y_MAX)])
            radius = np.random.uniform(*pedestrian_radius_range)
            ped = Pedestrian(position, velocity, desired_location, radius)
            self.pedestrians.append(ped)

    def frame_step(self):
        for ped in self.pedestrians:
            if not ped.exited:
                ped.location_updater()
                ped.force_summer(self.pedestrians)
                ped.step(ped.force)

                # Check for exit
                if ped.position[0] >= EXIT_X and EXIT_Y_MIN <= ped.position[1] <= EXIT_Y_MAX:
                    ped.exited = True

        self.time += dt

    def animate(self):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)

        while self.time < total_time:
            ax.clear()
            ax.set_xlim(0, ROOM_WIDTH)
            ax.set_ylim(0, ROOM_HEIGHT)

            # Draw exit
            ax.plot([EXIT_X, EXIT_X], [EXIT_Y_MIN, EXIT_Y_MAX], color='green', linewidth=5)

            for ped in self.pedestrians:
                if not ped.exited:
                    circle = plt.Circle((ped.position[0], ped.position[1]), ped.radius, color='blue')
                    ax.add_artist(circle)

            plt.pause(0.01)
            self.frame_step()

        plt.ioff()
        plt.show()

    def run(self):
        self.initialize_pedestrians()
        self.animate()

if __name__ == "__main__":
    sim = Simulator(num_pedestrians=30)
    sim.run()

