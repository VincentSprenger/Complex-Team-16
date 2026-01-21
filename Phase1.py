import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
room_size=(12.0,12.0)
exit_width = 1.2
n_individuals = 150
radius = 0.30
mass = 80.0

# BEHAVIOR PARAMETERS
desired_velocity = 5.0  # Panic speed
tau = 0.5               # Relaxation Time

# FORCE PARAMETERS
A = 2000.0            # Social Force
B = 0.08
K = 120000.0          # Body Force (Compression)

class CrowdSimulation:
    def __init__(self):
        # Spawn agents
        self.pos = np.random.rand(n_individuals, 2)
        self.pos[:, 0] = self.pos[:, 0] * (room_size[0] - 3) + 0.5 
        self.pos[:, 1] = self.pos[:, 1] * (room_size[1] - 1) + 0.5 
        self.radius= np.random.normal(loc=radius, scale=0.05, size=n_individuals)
        self.vel = np.zeros((n_individuals, 2))
        self.forces_magnitude = np.zeros(n_individuals)
        # print("Initialized positions:", self.pos)
    
    def get_forces(self):
        total_force = np.zeros((n_individuals, 2))
        pressure_force = np.zeros(n_individuals)
        
        # 1. Goal Direction
        target = np.array([room_size[0], room_size[1]/2])
        direction = target - self.pos
        escaped_mask = self.pos[:, 0] > room_size[0]
        
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
        
        #clipping speed
        max_speed=8.0
        speed = np.linalg.norm(self.vel, axis=1)
        speed_safe = np.where(speed < 1e-6, 1e-6, speed)        
        self.vel = np.where(speed[:, None] > max_speed, self.vel / speed_safe[:, None] * max_speed, self.vel)
        
        new_pos=self.pos+self.vel*dt
       
        #clipping position (stay inside room)
        new_pos[:, 1] = np.clip(new_pos[:, 1], self.radius, room_size[1] - self.radius)        
        new_pos[:, 0] = np.maximum(new_pos[:, 0], self.radius)
        
        
        #handle wall collisions (except door)
        crossing_wall = (new_pos[:, 0] > (room_size[0] - self.radius)) & (self.pos[:, 0] < room_size[0])
        in_door = np.abs(new_pos[:, 1] - room_size[1]/2) < (exit_width/2 - self.radius/2)
        
        blocked_indices = crossing_wall & (~in_door)
        
        new_pos[blocked_indices, 0] = room_size[0] - self.radius[blocked_indices]
        self.vel[blocked_indices, 0] = 0
        self.pos = new_pos    

sim=CrowdSimulation()
dt = 0.04

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- PLOT 1: ROOM ---
ax1.set_xlim(0, room_size[0] + 4)
ax1.set_ylim(0, room_size[1])
ax1.set_aspect('equal')
ax1.set_title(f"Crowd Evacuation (N={n_individuals})")

# WALLS
ax1.plot([room_size[0], room_size[0]], 
         [(room_size[1] - exit_width) / 2, (room_size[1] + exit_width) / 2], 
         color='green', linewidth=5)
ax1.plot([0, room_size[0]], [0, 0], color='black')
ax1.plot([0, room_size[0]], [room_size[1], room_size[1]], color='black')
ax1.plot([0, 0], [0, room_size[1]], color='black')
ax1.plot([room_size[0], room_size[0]], [0, (room_size[1] - exit_width) / 2], color='black')
ax1.plot([room_size[0], room_size[0]], [(room_size[1] + exit_width) / 2, room_size[1]], color='black')  


# PEOPLE
scatter = ax1.scatter(sim.pos[:, 0], sim.pos[:, 1], s=400*sim.radius, c='b', edgecolors='black', linewidth=0.5)

# COLORBAR
norm = mcolors.Normalize(vmin=0, vmax=3000)
cmap = cm.jet
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', fraction=0.04, pad=0.1, shrink=0.7)
cbar.set_label('Pressure Force (Newton)', labelpad=5)

# --- PLOT 2: HISTOGRAM (Filtered) ---
# We start bins at 50 to ignore people with 0 force (Safe)
# We use fewer bins (15) for wider bars
ax2.set_xlim(50, 6000)
ax2.set_ylim(0, 40) # Lower Y limit since we removed the "0-force" majority
ax2.set_title("Danger Zone: People under Pressure (>50N)")
ax2.set_xlabel("Crushing Force (N)")
ax2.set_ylabel("Number of People")

bins = np.linspace(50, 5000, 8)
bars = ax2.bar(bins[:-1], np.zeros(len(bins)-1), width=np.diff(bins), align='edge', color='red', alpha=0.7, edgecolor='black')


def animate(frame):
    for _ in range(2): 
        sim.update(dt/2)
    
    scatter.set_offsets(sim.pos)
    
    pressures = sim.forces_magnitude
    colors = cmap(norm(pressures))
    scatter.set_color(colors)
    
    # Histogram Update
    counts, _ = np.histogram(sim.forces_magnitude, bins=bins)
    for bar, count in zip(bars, counts):
        bar.set_height(count)
        
    return scatter, list(bars)

anim = FuncAnimation(fig, animate, frames=400, interval=30, blit=False)
plt.show()
