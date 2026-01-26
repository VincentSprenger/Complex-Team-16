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
EXIT_WIDTH = 1.0          
N_INDIVIDUALS = 100
RADIUS = 0.25
STD_RADIUS = 0.03
MASS = 80.0

# BEHAVIOR PARAMETERS
MAX_SPEED = 12.0       
TAU = 0.5             # Relaxation Time

# HELBING FORCE PARAMETERS
A = 2000.0            # Social Force Strength (Newton) --> in the paper was 2000
B = 0.08              # Social Force Falloff (meters)
K1 = 120000.0          # Body Compression Force (Newton/m)
K2 = 240000.0          # Friction Force (Newton/m)

# TIMESTEP / TIMEOUT / ANTI-JAMMING PARAMETERS
DT=0.02              # Time Step (seconds)
SUBSTEPS = 15        # Physics Substeps per frame for stability
MAX_STEPS = 5000  # If simulation exceeds this, it is considered jammed
PENALTY_TIME = MAX_STEPS*DT # approx 1800 seconds




class CrowdSimulation:
    def __init__(self, room_size=DEFAULT_ROOM_SIZE, desired_velocity=5.0, noise_strength=0.0, n_individuals=N_INDIVIDUALS):
        """
        Initialize the simulation with specific parameters.
        """
        self.room_size = room_size
        self.dt = DT
        self.n_individuals = n_individuals
        self.desired_velocity = desired_velocity
        self.noise_strength = noise_strength

        # --- AGENT INITIALIZATION ---
        self.pos = np.random.rand(self.n_individuals, 2)
        # Spawn agents on the left side of the room to force them to cross it
        self.pos[:, 0] = self.pos[:, 0] * (self.room_size[0] - 3) + 0.5 
        self.pos[:, 1] = self.pos[:, 1] * (self.room_size[1] - 1) + 0.5 
        
        # Random radii (Normal distribution around RADIUS)
        self.radius = np.random.normal(RADIUS, STD_RADIUS, size=self.n_individuals)        
        self.vel = np.zeros((self.n_individuals, 2))
        self.forces_magnitude = np.zeros(self.n_individuals)
        
        # State tracking
        self.time = 0.0
        self.exit_times = [] 
        self.exited_mask = np.zeros(self.n_individuals, dtype=bool)

    def get_forces(self):
        """
        Calculate Social Forces and Physical Forces (Helbing Model).
        """
        total_force = np.zeros((self.n_individuals, 2))

        # 1. DESIRED FORCE (Goal Direction)
        target = np.array([self.room_size[0], self.room_size[1]/2])      
        direction = target - self.pos
        escaped_mask = self.pos[:, 0] > self.room_size[0]

        dist_to_target = np.linalg.norm(direction, axis=1, keepdims=True)
        e_desired = direction / (dist_to_target + 1e-6)
        
        # If agent has escaped, keep running to the right (to clear the buffer)
        e_desired[escaped_mask] = np.array([1.0, 0.0])
        
        f_desired = MASS * (self.desired_velocity * e_desired - self.vel) / TAU
        total_force += f_desired 

        # 2. Individual-Individual
        diff_matrix = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :] # Shape: (N, N, 2) with broadcasting
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        

        n_matrix = diff_matrix / (dist_matrix[:, :, np.newaxis] + 1e-6)
        t_matrix = np.zeros_like(n_matrix)
        t_matrix[:, :, 0] = -n_matrix[:, :, 1]
        t_matrix[:, :, 1] = n_matrix[:, :, 0]
        
        
        sum_radii = self.radius[:, np.newaxis] + self.radius[np.newaxis, :]
        overlap = sum_radii - dist_matrix
        
        # A) Social
        f_social = A * np.exp(overlap / B)[:, :, np.newaxis] * n_matrix
        
        # B) Physical
        mask_touch = overlap > 0
        f_phys = np.zeros_like(diff_matrix)
        
        if np.any(mask_touch):
            # Body Force
            f_body_vec = (K1 * overlap[mask_touch])[:, None] * n_matrix[mask_touch]
            
            # Friction
            vel_diff = self.vel[np.newaxis, :, :] - self.vel[:, np.newaxis, :] # Shape: (N, N, 2)

            v_dot_t = np.sum(vel_diff * t_matrix, axis=2) # Shape: (N, N)
            f_fric_vec = (K2 * overlap[mask_touch] * v_dot_t[mask_touch])[:, None] * t_matrix[mask_touch]
            
            f_phys[mask_touch] = f_body_vec + f_fric_vec

        total_force += np.sum(f_social + f_phys, axis=1)

        # -------------------------------------------------
        # 3. WALL INTERACTIONS (VECTORIZED - NO FOR LOOP)
        # -------------------------------------------------
        
        # --- A. Wall Geometry Setup ---
        # Order: [Left, Bottom, Top, Right]
        wall_normals = np.array([
            [ 1.0,  0.0], # Left Wall 
            [ 0.0,  1.0], # Bottom Wall
            [ 0.0, -1.0], # Top Wall   
            [-1.0,  0.0]  # Right Wall 
        ])
        
        # Tangents (4, 2): Rotated by 90 degrees (-ny, nx) for friction
        wall_tangents = np.array([
            [ 0.0,  1.0], # Left   Y
            [-1.0,  0.0], # Bottom -X
            [ 1.0,  0.0], # Top    X
            [ 0.0,  1.0]  # Right  Y
        ])

        
        W, H = self.room_size
        r = self.radius
        x, y = self.pos[:, 0], self.pos[:, 1]

        # dists[i, j] = distance of agent i from wall j
        dists = np.stack([
            x - r,            
            y - r,            
            (H - r) - y,      
            (W - r) - x       
        ], axis=1)

        # individuals aligned with the door in the right wall 
        door_half_width = EXIT_WIDTH / 2.0
        dist_from_door_center = np.abs(y - H/2)
        in_door_gap = dist_from_door_center < door_half_width
        
        ignore_right_wall = escaped_mask | in_door_gap
        dists[ignore_right_wall, 3] = np.inf

        overlap = -dists         
        mask_touch = overlap > 0
        
        f_soc_mag = A * np.exp(overlap / B)
        
        f_body_mag = K1 * np.maximum(overlap, 0)
        
        f_normal_total = f_soc_mag + f_body_mag         
        
        v_tan_mag = self.vel @ wall_tangents.T  # v_tan_mag[i, j] = tangential velocity of pedestrian i against wall j
        
        f_friction_mag = np.zeros_like(overlap)
        if np.any(mask_touch):
                    f_friction_mag[mask_touch] = -K2 * overlap[mask_touch] * v_tan_mag[mask_touch]        


        F_wall_normal = f_normal_total @ wall_normals 
        
        F_wall_friction = f_friction_mag @ wall_tangents
        
        total_force += F_wall_normal + F_wall_friction

        return total_force

    def update(self, dt):
        """
        Physics Integration Step.
        """
        forces = self.get_forces()
        acc = forces / MASS
        self.vel += acc * dt
        
        # --- NOISE (Random Walk) ---
        if self.noise_strength > 0:
            noise = self.noise_strength * np.sqrt(dt) * np.random.randn(self.n_individuals, 2)
            self.vel += noise

        # --- SPEED LIMITING ---
        speed = np.linalg.norm(self.vel, axis=1)
        speed_safe = np.where(speed < 1e-6, 1e-6, speed)        
        self.vel = np.where(speed[:, None] > MAX_SPEED, self.vel / speed_safe[:, None] * MAX_SPEED, self.vel)
        
        # # --- OPTIONAL: PHYSICAL BLOCKING (Real Jamming) ---
        # # Uncomment below to simulate agents getting stuck due to high pressure
        # CRITICAL_PRESSURE = 4000.0
        # stuck_mask = self.forces_magnitude > CRITICAL_PRESSURE
        # self.vel[stuck_mask] = 0.0

        # Update Position
        new_pos = self.pos + self.vel * dt
        
        # --- BOUNDARY CONDITIONS ---
        # 1. Floor and Ceiling
        new_pos[:, 1] = np.clip(new_pos[:, 1], self.radius, self.room_size[1] - self.radius)        
        # 2. Left Wall
        new_pos[:, 0] = np.maximum(new_pos[:, 0], self.radius)
        
        # 3. Right Wall (Hard Obstacle with Door Gap)
        crossing_wall = (new_pos[:, 0] > (self.room_size[0] - self.radius)) & (self.pos[:, 0] < self.room_size[0])
        in_door = np.abs(new_pos[:, 1] - self.room_size[1]/2) < (EXIT_WIDTH/2 - self.radius/2)
        
        # Block agents hitting the wall, let those in the door pass
        blocked_indices = crossing_wall & (~in_door)
        
        new_pos[blocked_indices, 0] = self.room_size[0] - self.radius[blocked_indices]
        self.vel[blocked_indices, 0] = 0
        self.pos = new_pos 

        # --- EXIT LOGGING ---
        escaped_now = self.pos[:, 0] > self.room_size[0]
        new_exits = escaped_now & (~self.exited_mask)
        exit_indices = np.where(new_exits)[0]
        for _ in exit_indices:
            self.exit_times.append(self.time)
        self.exited_mask |= escaped_now

    def animate(self):
        """
        Visualize the simulation using Matplotlib.
        Uses EllipseCollection for physically accurate circles.
        """
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # --- PLOT 1: ROOM ---
        ax1.set_xlim(0, self.room_size[0] + 4)
        ax1.set_ylim(0, self.room_size[1])
        ax1.set_aspect('equal') # CRITICAL: Ensures circles look like circles, not ovals
        ax1.set_title(f"Crowd Evacuation (N={self.n_individuals}, V0={self.desired_velocity})")

        # Draw Walls
        # Door gap (Green)
        ax1.plot([self.room_size[0], self.room_size[0]], [(self.room_size[1] - EXIT_WIDTH) / 2, (self.room_size[1] + EXIT_WIDTH) / 2], color='green', linewidth=5)
        # Black walls
        ax1.plot([0, self.room_size[0]], [0, 0], 'k'); ax1.plot([0, self.room_size[0]], [self.room_size[1], self.room_size[1]], 'k')
        ax1.plot([0, 0], [0, self.room_size[1]], 'k'); ax1.plot([self.room_size[0], self.room_size[0]], [0, (self.room_size[1] - EXIT_WIDTH) / 2], 'k')
        ax1.plot([self.room_size[0], self.room_size[0]], [(self.room_size[1] + EXIT_WIDTH) / 2, self.room_size[1]], 'k')  

        # --- AGENT RENDERING (EllipseCollection) ---
        self.norm = mcolors.Normalize(vmin=0, vmax=3000)
        self.cmap = cm.jet

        # We use EllipseCollection to render circles defined by PHYSICAL units (meters)
        # units='xy' tells matplotlib that width/height are in data coordinates.
        diameters = 2 * self.radius
        
        self.collection = EllipseCollection(
            widths=diameters, 
            heights=diameters, 
            angles=0, 
            units='xy',           # <--- KEY: Matches agent size to room size
            offsets=self.pos,
            transOffset=ax1.transData,
            cmap=self.cmap,
            norm=self.norm,
            edgecolors='black',
            linewidths=0.5
        )
        self.collection.set_array(self.forces_magnitude)
        ax1.add_collection(self.collection)

        # # Colorbar
        # self.sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        # self.sm.set_array([])
        # self.cbar = self.fig.colorbar(self.sm, ax=ax1, orientation='horizontal', fraction=0.04, pad=0.1, shrink=0.7)
        # self.cbar.set_label('Pressure Force (Newton)', labelpad=5)

        # # --- PLOT 2: HISTOGRAM ---
        # ax2.set_xlim(50, 6000)
        # ax2.set_ylim(0, 40)
        # ax2.set_title("Danger Zone: People under Pressure (>50N)")
        # self.bins = np.linspace(50, 5000, 8)
        # self.bars = ax2.bar(self.bins[:-1], np.zeros(len(self.bins)-1), width=np.diff(self.bins), align='edge', color='red', alpha=0.7, edgecolor='black')

        def animate_frame(frame):
            # Update physics twice per frame for stability
            for _ in range(SUBSTEPS): 
                self.update(self.dt/SUBSTEPS)
            
            # Update visuals
            self.collection.set_offsets(self.pos)
            self.collection.set_array(self.forces_magnitude)
            
            # # Update Histogram
            # counts, _ = np.histogram(self.forces_magnitude, bins=self.bins)
            # for bar, count in zip(self.bars, counts): bar.set_height(count)
            
            return self.collection #, list(self.bars)
        
        anim = FuncAnimation(self.fig, animate_frame, frames=400, interval=30, blit=False)
        plt.show()

    def run(self):
        """
        Runs the simulation without graphics (Headless).
        Returns:
            - float: Time taken to evacuate.
            - None: If simulation exceeds max_steps (Jamming detected).
        """

        stuck_counter = 0
        last_exited_count = 0
        
        for step in range(MAX_STEPS):
            for _ in range(SUBSTEPS):
                self.update(self.dt/SUBSTEPS)
            self.time += self.dt
            # Check if everyone has escaped
            if np.min(self.pos[:, 0]) > self.room_size[0]:
                return self.time
            
            current_exited_count = np.sum(self.pos[:, 0] > self.room_size[0])
            
            if current_exited_count == last_exited_count:
                stuck_counter += 1
            else:
                stuck_counter = 0 
                last_exited_count = current_exited_count
            
            if stuck_counter > 500: # stuck for 10 seconds
                return None 
            
        return None # Timeout / Jamming


# ==========================================
# --- EXPERIMENTS ---
# ==========================================


# ==========================================
# --- 3. VELOCITY EFFECT (Complexity Study) ---
# ==========================================
def EscapeTimeVSVelocity():
    # Define velocity range (from normal walking to sprinting/panic)
    velocities = np.arange(1.5, MAX_SPEED, (MAX_SPEED-1.5)/10.) 
    # velocities=velocities[::-1]
    avg_times = []
    std_times = []
    
    runs = 3          # Number of runs per velocity
    
    # Penalty Parameters
    # If they don't exit within this limit, we mark it as "Jammed"
    
    print("\n--- STARTING COMPLEXITY STUDY: FASTER-IS-SLOWER EFFECT ---")
    print("Hypothesis: Higher desired velocity leads to coordination breakdown and jamming.")
    
    for v in tqdm(velocities):
        run_times = []
        
        for _ in range(runs):
            # Create simulation with specific desired velocity
            sim = CrowdSimulation(desired_velocity=v, n_individuals=N_INDIVIDUALS)
            
            # Run ONCE without retries. We want to detect the failure.
            val = sim.run()
            
            if val is not None:
                # Case A: Successful Evacuation
                run_times.append(val)
            else:
                # Case B: JAMMING (Emergence)
                # The system failed to clear. We assign the max penalty time.
                print(f"[!] Jamming detected at V0={v:.2f} ({_+1}/{runs})")
                # run_times.append(PENALTY_TIME)
        
        # Calculate statistics
        if run_times != []:
            avg_times.append(np.mean(run_times))
            std_times.append(np.std(run_times))
        else:
            avg_times.append(PENALTY_TIME)
            std_times.append(0.0)

    # --- PLOTTING ---
    plt.figure(figsize=(10,6))
    
    # Main Curve
    plt.errorbar(velocities, avg_times, yerr=std_times, fmt='-o', color='blue', capsize=5, label='Evacuation Time')
    
    # Jamming Threshold Line
    plt.axhline(y=PENALTY_TIME, color='red', linestyle='--', label='Jamming Threshold (Timeout)')
    
    plt.title("Complexity: The 'Faster-is-Slower' Effect")
    plt.xlabel("Desired Velocity (m/s)")    
    plt.ylabel("Evacuation Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    plt.show()
    

# ==========================================
# --- 4. NOISE EFFECT (Complexity Study) ---
# ==========================================
def NoiseEffectExperiment(noise_values, vel, runs=5):
    print("\n--- STARTING COMPLEXITY STUDY: NOISE EFFECT ---")
    print("Hypothesis: Noise (randomness) might paradoxically IMPROVE flow by breaking jams.")
    
    avg_times = []
    std_times = []
    
    for sigma in tqdm(noise_values):
        run_times = []
        success_rate = []
        for _ in range(runs):
            sim = CrowdSimulation(desired_velocity=vel, noise_strength=sigma)
            val = sim.run()
            
            if val is not None:
                run_times.append(val)
            else:
                # JAMMING DETECTED: Assign maximum penalty time.
                # This represents the "infinite" time of a blocked system.
                print(f"[!] Jamming detected at Noise={sigma:.3f} ({_+1}/{runs})")
                run_times.append(PENALTY_TIME)
        success_rate.append(sum(1 for t in run_times if t < PENALTY_TIME) / runs)
        avg_times.append(np.mean(run_times))
        std_times.append(np.std(run_times))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(noise_values, avg_times, yerr=std_times, fmt='-s', color='green', capsize=5)
    
    # Plot the Jamming Threshold line
    plt.axhline(y=PENALTY_TIME, color='r', linestyle='--', label='Jamming Threshold (Timeout)')
    
    plt.title("Complexity: Can Noise Solve Jamming?")
    plt.xlabel("Noise Strength (Randomness)")
    plt.ylabel("Evacuation Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()    

    plt.figure(figsize=(10, 6))
    plt.plot(noise_values, success_rate, '-o', color='orange')
    plt.title("Success Rate vs Noise Strength")
    plt.xlabel("Noise Strength (Randomness)")
    plt.ylabel("Success Rate")
    plt.show()
        
# ==========================================
# --- 5. ASPECT RATIO (Complexity Study) ---
# ==========================================
# def AspectRatio(ratio_list, runs=5):

# ==========================================
# --- MAIN MENU ---
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n===========================")
        print(" CROWD SIMULATION SUITE")
        print("===========================")
        print("1. Run Standard Animation (REAL CIRCLES)")
        print("2. Run Animation with NOISE")
        print("3. Velocity Effect")
        print("4. Noise Effect")
        print("5. Aspect Ratio Effect")
        print("0. Exit")
        
        choice = input("\nEnter selection: ")
        
        if choice == "1":
            sim = CrowdSimulation(desired_velocity=10.5)
            sim.animate()
        elif choice == "2":
            sim = CrowdSimulation(desired_velocity=5.0, noise_strength=0.5)
            sim.animate()
        elif choice == "3":
            EscapeTimeVSVelocity()
        elif choice == "4":
            # Test noise from 0 (jamming) to high (chaos)
            noise_vals = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2]
            NoiseEffectExperiment(noise_vals, vel=9.0, runs=5)
        elif choice == "5":
            # Test various room shapes
            # AspectRatio([0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
            print("Aspect Ratio study not implemented yet.")
        elif choice == "0":
            break
        else:
            print("Invalid selection.")