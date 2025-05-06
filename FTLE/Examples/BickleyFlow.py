import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from FTLE.Flat.FlatSurfaceMain import run_FTLE_2d
import h5py
import numpy as np

# --- Load Bickley flow data ---
file_path = os.path.join(os.path.dirname(__file__), 'bickley_flow_data.h5')
with h5py.File(file_path, 'r') as f:
    velocity_points = f['points'][:]                 # shape (M, 2)
    velocity_vectors = f['vectors'][:]               # shape (M, 2, T)
    time_steps = f['time_steps'][:]                  # shape (T,)

# --- Define 2D grid for particle seeding (must be inside domain) ---
x = np.linspace(0, 20, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)

# --- FTLE parameters ---
initial_time = time_steps[0]
final_time = time_steps[-1]
dt = 0.5

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
