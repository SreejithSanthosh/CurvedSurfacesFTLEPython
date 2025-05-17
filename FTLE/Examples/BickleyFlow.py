import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Adjust for relative import

from FTLE.Flat.FlatSurfaceMain import run_FTLE_2d # Primary FTLE computation
import h5py
import numpy as np

# --- Load Bickley flow data ---
file_path = os.path.join(os.path.dirname(__file__), 'bickley_flow_data.h5')
with h5py.File(file_path, 'r') as f:
    velocity_points = f['points'][:]                 # shape (M, 2)
    velocity_vectors = f['vectors'][:]               # shape (M, 2, T)
    time_steps = f['time_steps'][:]                  # shape (T,)

# --- Define 2D grid for particle seeding (must be inside domain) ---
x = np.linspace(0, 10, 120)
y = np.linspace(-3, 3, 120)
X, Y = np.meshgrid(x, y)

# Swap the rows and columns
X = X.T
Y = Y.T

# --- FTLE parameters ---
initial_time = time_steps[0]
final_time = 3
dt = 0.2

# --- Run FTLE computation ---
ftle, traj, iso, bftle, btraj, biso = run_FTLE_2d(
    velocity_points,
    velocity_vectors,
    X,
    Y,
    dt,
    initial_time,
    final_time,
    time_steps,
    plot_ftle=True,
    save_plot_path=None  # or provide a path like 'bickley_ftle_output.png'
)



import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_trajectories_2d(trajectories, save_path="trajectories", interval=100):
    """
    Creates and saves an animation of 2D particle trajectories over time.

    Parameters:
        trajectories (ndarray): shape (N, 2, T), forward particle trajectories.
        save_path (str): path to save the .mp4 animation.
        interval (int): delay between frames in milliseconds.
    """
    num_particles, _, num_frames = trajectories.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=10, c='blue')

    x_min, x_max = np.min(trajectories[:, 0, :]), np.max(trajectories[:, 0, :])
    y_min, y_max = np.min(trajectories[:, 1, :]), np.max(trajectories[:, 1, :])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Particle Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')

    def update(frame):
        scat.set_offsets(trajectories[:, :, frame])
        ax.set_title(f"Time Step {frame}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    ani.save(save_path, writer='ffmpeg', dpi=200)
    plt.close(fig)
    print(f"Saved trajectory animation to {save_path}")


animate_trajectories_2d(traj, save_path="bickley_trajectories.gif")

