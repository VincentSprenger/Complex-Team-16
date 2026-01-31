import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import EllipseCollection 
from tqdm import tqdm

# ==========================================
# --- CONFIGURATION & CONSTANTS ---
# ==========================================

# Default Simulation Environment
DEFAULT_ROOM_SIZE = (12.0, 12.0)
EXIT_WIDTH = 1.15         
N_INDIVIDUALS = 100
RADIUS = 0.25
STD_RADIUS = 0.0
MASS = 80.0

# BEHAVIOR PARAMETERS
MAX_SPEED = 12.0       
TAU = 0.5             # Relaxation Time

# HELBING FORCE PARAMETERS 
A = 2000.0            # Social Force Strength (Newton)
B = 0.08              # Social Force Falloff (meters)
K1 = 120000.0         # Body Compression Force (Newton/m)
K2 = 1000.0           # Friction Force (Newton/m)

# TIMESTEP / TIMEOUT / ANTI-JAMMING PARAMETERS
DT = 0.02             # Time Step (seconds)
SUBSTEPS = 15         # Physics Substeps per frame for stability
MAX_STEPS = 5000      # Max steps before simulation is considered jammed
PENALTY_TIME = MAX_STEPS * DT 

class CrowdSimulation:
    def __init__(self, room_size=DEFAULT_ROOM_SIZE, desired_velocity=6.0, noise_strength=3.0):
        """
        Initialize the simulation with specific parameters.
        """
        self.room_size = room_size
        self.dt = DT
        self.n_individuals = N_INDIVIDUALS
        self.noise_strength = noise_strength

        # --- AGENT INITIALIZATION ---
        self.pos = np.random.rand(self.n_individuals, 2)
        # Spawn agents on the left side of the room
        self.pos[:, 0] = self.pos[:, 0] * (self.room_size[0] - 3) + 0.5 
        self.pos[:, 1] = self.pos[:, 1] * (self.room_size[1] - 1) + 0.5 
        
        # Heterogeneous population: 85% small, 15% big
        n_small = int(self.n_individuals * 0.85)  
        n_big = self.n_individuals - n_small     
        r_small = np.ones(n_small) * 0.07  
        r_big   = np.ones(n_big)   * 0.45 
        self.radius = np.concatenate([r_small, r_big])
        np.random.shuffle(self.radius)
      
        self.vel = np.zeros((self.n_individuals, 2))
        
        # Approximate mass based on area
        self.mass = (self.radius ** 2) * MASS / RADIUS**2  
        self.desired_velocity = np.ones(self.n_individuals) * desired_velocity
        # Small agents move slightly faster
        self.desired_velocity[self.radius < 0.2] *= 1.3
        
        # State tracking
        self.time = 0.0
        self.evacuation_times = np.full(self.n_individuals, 9999.0) 
        self.exited_mask = np.zeros(self.n_individuals, dtype=bool)

    def get_forces(self):
        """
        Calculate Social Forces and Physical Forces (Helbing Model).
        """
        total_force = np.zeros((self.n_individuals, 2))

        # 1. DESIRED FORCE (Target: Center of the exit)
        target = np.array([self.room_size[0], self.room_size[1]/2])      
        direction = target - self.pos
        escaped_mask = self.pos[:, 0] > self.room_size[0]

        dist_to_target = np.linalg.norm(direction, axis=1, keepdims=True)
        e_desired = direction / (dist_to_target + 1e-6)
        
        # Keep moving right after escaping
        e_desired[escaped_mask] = np.array([1.0, 0.0])
                
        f_desired = self.mass[:, None] * (self.desired_velocity[:, None] * e_desired - self.vel) / TAU
        total_force += f_desired 

        # 2. AGENT-AGENT INTERACTIONS
        diff_matrix = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :] 
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        n_matrix = diff_matrix / (dist_matrix[:, :, np.newaxis] + 1e-6)
        t_matrix = np.zeros_like(n_matrix)
        t_matrix[:, :, 0] = -n_matrix[:, :, 1]
        t_matrix[:, :, 1] = n_matrix[:, :, 0]
        
        sum_radii = self.radius[:, np.newaxis] + self.radius[np.newaxis, :]
        overlap = sum_radii - dist_matrix
        
        # A) Social repulsive force
        f_social = A * np.exp(overlap / B)[:, :, np.newaxis] * n_matrix
        
        # B) Physical forces (Body compression and Friction)
        mask_touch = overlap > 0
        f_phys = np.zeros_like(diff_matrix)
        
        if np.any(mask_touch):
            f_body_vec = (K1 * overlap[mask_touch])[:, None] * n_matrix[mask_touch]
            vel_diff = self.vel[np.newaxis, :, :] - self.vel[:, np.newaxis, :] 
            v_dot_t = np.sum(vel_diff * t_matrix, axis=2) 
            f_fric_vec = (K2 * overlap[mask_touch] * v_dot_t[mask_touch])[:, None] * t_matrix[mask_touch]
            f_phys[mask_touch] = f_body_vec + f_fric_vec

        total_force += np.sum(f_social + f_phys, axis=1)

        # 3. WALL INTERACTIONS
        wall_normals = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]])
        wall_tangents = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        W, H = self.room_size
        r = self.radius
        x, y = self.pos[:, 0], self.pos[:, 1]

        dists = np.stack([x - r, y - r, (H - r) - y, (W - r) - x], axis=1)

        # Door logic
        door_half_width = EXIT_WIDTH / 2.0
        in_door_gap = np.abs(y - H/2) < door_half_width
        ignore_right_wall = escaped_mask | in_door_gap
        dists[ignore_right_wall, 3] = np.inf

        overlap_wall = -dists 
        mask_wall_touch = overlap_wall > 0
        
        f_soc_mag = A * np.exp(overlap_wall / B)
        f_body_mag = K1 * np.maximum(overlap_wall, 0)
        f_normal_total = f_soc_mag + f_body_mag         
        
        v_tan_mag = self.vel @ wall_tangents.T 
        f_friction_mag = np.zeros_like(overlap_wall)
        if np.any(mask_wall_touch):
            f_friction_mag[mask_wall_touch] = -K2 * overlap_wall[mask_wall_touch] * v_tan_mag[mask_wall_touch]        

        total_force += (f_normal_total @ wall_normals) + (f_friction_mag @ wall_tangents)

        return total_force

    def update(self, dt):
        """
        Physics Integration Step.
        """
        forces = self.get_forces()
        acc = forces / self.mass[:, None]
        self.vel += acc * dt

        # Speed limit
        speed = np.linalg.norm(self.vel, axis=1)
        speed_safe = np.where(speed < 1e-6, 1e-6, speed)        
        self.vel = np.where(speed[:, None] > MAX_SPEED, self.vel / speed_safe[:, None] * MAX_SPEED, self.vel)
        
        # Position update
        new_pos = self.pos + self.vel * dt
        
        # Boundary constraints
        new_pos[:, 1] = np.clip(new_pos[:, 1], self.radius, self.room_size[1] - self.radius)        
        new_pos[:, 0] = np.maximum(new_pos[:, 0], self.radius)
        
        # Exit gate physics
        crossing_right = (new_pos[:, 0] > (self.room_size[0] - self.radius)) & (self.pos[:, 0] < self.room_size[0])
        in_door = np.abs(new_pos[:, 1] - self.room_size[1]/2) < (EXIT_WIDTH/2 - self.radius/2)
        blocked = crossing_right & (~in_door)
        
        new_pos[blocked, 0] = self.room_size[0] - self.radius[blocked]
        self.vel[blocked, 0] = 0
        self.pos = new_pos 

        # Log evacuation times
        escaped_now = self.pos[:, 0] > self.room_size[0]
        new_exits = escaped_now & (~self.exited_mask)
        self.evacuation_times[new_exits] = self.time
        self.exited_mask |= escaped_now

    def animate(self):
        """
        Visualize the simulation.
        """
        self.fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax1.set_xlim(0, self.room_size[0] + 4)
        ax1.set_ylim(0, self.room_size[1])
        ax1.set_aspect('equal')
        
        # Unified title for the segregation study
        ax1.set_title("Crowd Dynamics: Size Segregation & Evacuation Flow")

        # Draw Environment
        ax1.plot([self.room_size[0], self.room_size[0]], [(self.room_size[1] - EXIT_WIDTH) / 2, (self.room_size[1] + EXIT_WIDTH) / 2], color='green', linewidth=5, label='Exit')
        ax1.plot([0, self.room_size[0]], [0, 0], 'k'); ax1.plot([0, self.room_size[0]], [self.room_size[1], self.room_size[1]], 'k')
        ax1.plot([0, 0], [0, self.room_size[1]], 'k'); ax1.plot([self.room_size[0], self.room_size[0]], [0, (self.room_size[1] - EXIT_WIDTH) / 2], 'k')
        ax1.plot([self.room_size[0], self.room_size[0]], [(self.room_size[1] + EXIT_WIDTH) / 2, self.room_size[1]], 'k')  

        diameters = 2 * self.radius
        self.collection = EllipseCollection(
            widths=diameters, heights=diameters, angles=0, units='xy',
            offsets=self.pos, transOffset=ax1.transData,
            edgecolors='black', linewidths=0.5
        )
        # Color by radius: Blue (small), Red (big)
        self.collection.set_array(self.radius)
        self.collection.set_cmap('bwr') 
        self.collection.set_clim(vmin=self.radius.min(), vmax=self.radius.max())
        ax1.add_collection(self.collection)

        def animate_frame(frame):
            for _ in range(SUBSTEPS): 
                self.update(self.dt/SUBSTEPS)
                self.time += (self.dt/SUBSTEPS)
            self.collection.set_offsets(self.pos)
            return self.collection
        
        anim = FuncAnimation(self.fig, animate_frame, frames=600, interval=30, blit=False)
        plt.show()

    def plot_segregation_results(self):
        """
        Boxplot to compare evacuation times: Small vs Large agents.
        """
        threshold = 0.20 
        small_times = self.evacuation_times[(self.radius < threshold) & (self.evacuation_times < 9000)]
        big_times = self.evacuation_times[(self.radius >= threshold) & (self.evacuation_times < 9000)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        data_to_plot = [small_times, big_times]
        labels = ['Small Agents (Fluid)', 'Large Agents (Obstacles)']
        colors = ['lightblue', 'salmon']
        
        bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=labels, medianprops=dict(color='black'))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_title('Granular Segregation: Evacuation Time by Size')
        ax.set_ylabel('Time (s)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        stats_text = f"Escaped Small: {len(small_times)}\nEscaped Large: {len(big_times)}"
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, ha='right', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.show()



if __name__ == "__main__":
    print("\n--- STARTING COMPLEXITY STUDY: GRANULAR FLOW EFFECT ---")
    sim = CrowdSimulation(noise_strength=0.0, desired_velocity=4.0)
    sim.animate()    
    sim.plot_segregation_results()