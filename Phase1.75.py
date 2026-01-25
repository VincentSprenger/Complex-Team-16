import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm

# --- CONFIGURATION ---
room_size=(12.0,12.0)
exit_width = 0.45
n_individuals = 250
radius = 0.30
mass = 80.0

# BEHAVIOR PARAMETERS
desired_velocity = 5.0  # Panic speed
tau = 0.3              # Relaxation Time

# FORCE PARAMETERS
A = 2000.0            # Social Force
B = 0.08
K = 120000.0          # Body Force (Compression)

# Noise parameter for random walk noise experiment
noise_strength = 0.0 

class CrowdSimulation:
    def __init__(self, room_size=room_size, timestep=0.04, n_individuals=n_individuals):
        # Spawn agents
        self.room_size = room_size
        self.dt = timestep
        self.n_individuals = n_individuals
        self.pos = np.random.rand(n_individuals, 2)
        self.pos[:, 0] = self.pos[:, 0] * (self.room_size[0] - 3) + 0.5 
        self.pos[:, 1] = self.pos[:, 1] * (self.room_size[1] - 1) + 0.5 
        self.radius= np.random.normal(loc=radius, scale=0.05, size=n_individuals)
        self.vel = np.zeros((n_individuals, 2))
        self.forces_magnitude = np.zeros(n_individuals)
        self.exited = 0
        self.time = 0.0
        self.exit_times = [] #use for flow
        self.exited_mask = np.zeros(self.n_individuals, dtype=bool)

        
        # print("Initialized positions:", self.pos)
    
    def get_forces(self):
        total_force = np.zeros((n_individuals, 2))
        pressure_force = np.zeros(n_individuals)
        
        # 1. Goal Direction
        target = np.array([self.room_size[0], self.room_size[1]/2])
        direction = target - self.pos
        escaped_mask = self.pos[:, 0] > self.room_size[0]

        dist_to_target = np.linalg.norm(direction, axis=1, keepdims=True)
        e_desired = direction / (dist_to_target + 1e-6) # versor
        e_desired[escaped_mask] = np.array([1.0, 0.0]) # Run right if escaped
        
        f_desired = mass * (desired_velocity * e_desired - self.vel) / tau # equation in the paper
        total_force += f_desired

        # 2. Individual-Individual Repulsion
        for i in range(n_individuals):
            d_vec = self.pos[i] - self.pos[i+1:] 
            dist = np.linalg.norm(d_vec, axis=1)
            safe_dist = np.where(dist < 1e-6, 1e-6, dist) 
            n_vec = d_vec / safe_dist[:, None] # versor
            
            overlap_factor = 0.75 #allow some overlap for more realistic behavior
            
            sum_radii = (self.radius[i] + self.radius[i+1:]) * overlap_factor   

            overlap = sum_radii - dist
            
            f_social = A * np.exp(overlap / B)[:, None] * n_vec
            
            f_body = np.zeros_like(d_vec)
            mask_touch = overlap > 0
            
            if np.any(mask_touch):
                f_body[mask_touch] = (K * overlap[mask_touch])[:, None] * n_vec[mask_touch]
                body_mag = K * overlap[mask_touch]
                pressure_force[i] += np.sum(body_mag)
                j_indices = np.arange(i + 1, n_individuals)[mask_touch]
                np.add.at(pressure_force, j_indices, body_mag)

            force = f_social + f_body
            total_force[i] += np.sum(force, axis=0)
            total_force[i+1:] -= force

        self.forces_magnitude = pressure_force
        return total_force

    def update(self,dt):
        forces=self.get_forces()
        acc=forces/mass
        
        self.vel+=acc*dt
        
        # Random walk (stochastic motion)
        if noise_strength > 0:
            noise = noise_strength * np.sqrt(dt) * np.random.randn(self.n_individuals, 2)
            self.vel += noise

        #clipping speed
        max_speed=8.0
        speed = np.linalg.norm(self.vel, axis=1)
        speed_safe = np.where(speed < 1e-6, 1e-6, speed)        
        self.vel = np.where(speed[:, None] > max_speed, self.vel / speed_safe[:, None] * max_speed, self.vel)
        
        new_pos=self.pos+self.vel*dt
       
        #clipping position (stay inside room)
        new_pos[:, 1] = np.clip(new_pos[:, 1], self.radius, self.room_size[1] - self.radius)        
        new_pos[:, 0] = np.maximum(new_pos[:, 0], self.radius)
        
        
        #handle wall collisions (except door)
        crossing_wall = (new_pos[:, 0] > (self.room_size[0] - self.radius)) & (self.pos[:, 0] < self.room_size[0])
        in_door = np.abs(new_pos[:, 1] - self.room_size[1]/2) < (exit_width/2 - self.radius/2)
        
        blocked_indices = crossing_wall & (~in_door)
        
        new_pos[blocked_indices, 0] = self.room_size[0] - self.radius[blocked_indices]
        self.vel[blocked_indices, 0] = 0
        self.pos = new_pos 

        ##Exit detection and logging
        escaped_now = self.pos[:, 0] > self.room_size[0]

        new_exits = escaped_now & (~self.exited_mask)

        exit_indices = np.where(new_exits)[0]
        for _ in exit_indices:
            self.exit_times.append(self.time)

        self.exited_mask |= escaped_now

    def animate(self, histogram=True):
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # --- PLOT 1: ROOM ---
        ax1.set_xlim(0, self.room_size[0] + 4)
        ax1.set_ylim(0, self.room_size[1])
        ax1.set_aspect('equal')
        ax1.set_title(f"Crowd Evacuation (N={n_individuals})")

        # WALLS
        ax1.plot([self.room_size[0], self.room_size[0]], 
                [(self.room_size[1] - exit_width) / 2, (self.room_size[1] + exit_width) / 2], 
                color='green', linewidth=5)
        ax1.plot([0, self.room_size[0]], [0, 0], color='black')
        ax1.plot([0, self.room_size[0]], [self.room_size[1], self.room_size[1]], color='black')
        ax1.plot([0, 0], [0, self.room_size[1]], color='black')
        ax1.plot([self.room_size[0], self.room_size[0]], [0, (self.room_size[1] - exit_width) / 2], color='black')
        ax1.plot([self.room_size[0], self.room_size[0]], [(self.room_size[1] + exit_width) / 2, self.room_size[1]], color='black')  


        # PEOPLE
        self.scatter = ax1.scatter(self.pos[:, 0], self.pos[:, 1], s=400*self.radius, c='b', edgecolors='black', linewidth=0.5)

        # COLORBAR
        self.norm = mcolors.Normalize(vmin=0, vmax=3000)
        self.cmap = cm.jet
        self.sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        self.sm.set_array([])
        self.cbar = self.fig.colorbar(self.sm, ax=ax1, orientation='horizontal', fraction=0.04, pad=0.1, shrink=0.7)
        self.cbar.set_label('Pressure Force (Newton)', labelpad=5)

        # --- PLOT 2: HISTOGRAM (Filtered) ---
        # We start bins at 50 to ignore people with 0 force (Safe)
        # We use fewer bins (15) for wider bars
        ax2.set_xlim(50, 6000)
        ax2.set_ylim(0, 40) # Lower Y limit since we removed the "0-force" majority
        ax2.set_title("Danger Zone: People under Pressure (>50N)")
        ax2.set_xlabel("Crushing Force (N)")
        ax2.set_ylabel("Number of People")

        self.bins = np.linspace(50, 5000, 8)
        self.bars = ax2.bar(self.bins[:-1], np.zeros(len(self.bins)-1), width=np.diff(self.bins), align='edge', color='red', alpha=0.7, edgecolor='black')

    def animate2(self, frame):
        for _ in range(2): 
            self.update(self.dt/2)

        self.scatter.set_offsets(self.pos)

        pressures = self.forces_magnitude
        colors = self.cmap(self.norm(pressures))
        self.scatter.set_color(colors)
        
        # Histogram Update
        counts, _ = np.histogram(self.forces_magnitude, bins=self.bins)
        for bar, count in zip(self.bars, counts):
            bar.set_height(count)

        return self.scatter, list(self.bars)

    def animate_run(self):
        anim = FuncAnimation(self.fig, self.animate2, frames=400, interval=30, blit=False)
        plt.show()
    
    def run(self, total_steps=1000):
        for _ in range(total_steps):
            self.update(self.dt)
            self.time += self.dt
            if min(self.pos[:, 0]) > self.room_size[0]:
                return self.time
                #print("All individuals have exited the room.", self.time)
                #break
        return np.nan


def compute_flow(exit_times, dt):
    """
    Convert exit timestamps into a flow time series
    (number of exits per timestep).
    """
    if len(exit_times) == 0:
        return np.array([])

    exit_times = np.array(exit_times)
    t_max = exit_times.max()
    bins = np.arange(0, t_max + dt, dt)

    flow, _ = np.histogram(exit_times, bins=bins)
    return flow
    
#Return time when x-th individual exits, if fewer than x exits occured return np.nan
def time_until_x_exits(exit_times, x):
    if len(exit_times) < x:
        return np.nan
    return exit_times[x - 1]
#sim=CrowdSimulation((48, 3), timestep=0.04, n_individuals=150)
#sim.animate()
#sim.animate_run()

#sim.run(total_steps=10000)


def AspectRatio(ratio : list[float], runs: int = 20):
    results = []
    mean_res = []
    std_res = []
    for i in ratio:
        room_size = (12 * i, 12 / i)
        runtime = []
        for _ in tqdm(range(runs)):
            sim=CrowdSimulation(room_size, timestep=0.04, n_individuals=150)
            runtime_res = sim.run(total_steps=10000)
            runtime.append(runtime_res)

        mean_res.append(np.nanmean(runtime))
        std_res.append(np.nanstd(runtime))

    plt.errorbar(ratio, mean_res, yerr=std_res, marker='o')
    plt.xlabel("Aspect ratio (width / height)")
    plt.ylabel("Mean evacuation time (s)")
    plt.title("Evacuation time vs room aspect ratio")
    plt.show()

#AspectRatio([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])

#AspectRatio([1])

### Noise Random Walk Experiment
#measure time until x individuals exits as function of noise strength

def NoiseExperiment(noise_values, runs, x_exits):
    global noise_strength

    max_time = 60.0 #seconds , stop condition

    mean_times = []
    std_times = []
    
    for sigma in noise_values:
        noise_strength = sigma
        times = []

        for _ in tqdm(range(runs)):
            sim = CrowdSimulation(room_size=(12.0,12.0),
                                  timestep=0.04,
                                  n_individuals=250)
            
            max_steps = int(max_time / sim.dt)
            
            sim.run(total_steps=max_steps)
            
            t_x = time_until_x_exits(sim.exit_times, x_exits)
            times.append(t_x)
        
        success_rate = np.sum(~np.isnan(times)) / len(times)

        print(f"noise={sigma}, times={times}, std={np.nanstd(times)}")
        print(f"noise={sigma}, success_rate={success_rate}")

        mean_times.append(np.nanmean(times))
        std_times.append(np.nanstd(times))

    plt.errorbar(noise_values, mean_times, yerr=std_times, marker='o')
    plt.xlabel("Noise strength")
    plt.ylabel(f"Time until {x_exits} exits (s)")
    plt.title("Noise effect on partial evacuation time")
    plt.show()

noise_vals = [0.0, 0.05, 0.08, 0.1, 0.15, 0.25, 0.4, 0.6]
NoiseExperiment(noise_vals,runs = 5, x_exits=80)

##Measure flow variance (exits per timestep) as a function of noise strength.

def NoiseFlowVarianceExperiment(noise_values, runs):

    global noise_strength

    max_time = 60.0 #seconds , stop condition

    flow_variances = []
    flow_variances_std = []

    for sigma in noise_values:
        noise_strength = sigma
        run_variances = []

        for _ in tqdm(range(runs)):
            sim = CrowdSimulation(
                room_size=(12.0, 12.0),
                timestep=0.04,
                n_individuals=250
            )

            max_steps = int(max_time / sim.dt)
            
            sim.run(total_steps=max_steps)
            
            flow = compute_flow(sim.exit_times, sim.dt)

            if len(flow) > 1:
                run_variances.append(np.var(flow))

        flow_variances.append(np.nanmean(run_variances))
        flow_variances_std.append(np.nanstd(run_variances))

    # plot
    plt.errorbar(
        noise_values,
        flow_variances,
        yerr=flow_variances_std,
        marker='o',
        capsize=4
    )
    plt.xlabel("Noise strength")
    plt.ylabel("Flow variance (exits per timestep)")
    plt.title("Noise-induced flow intermittency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return flow_variances

#NoiseFlowVarianceExperiment(noise_vals, runs=5)
