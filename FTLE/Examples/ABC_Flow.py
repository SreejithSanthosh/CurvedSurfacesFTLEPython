import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Adjust for relative import

from FTLE.Flat.FlatSurfaceMain import run_FTLE_3d  # Primary FTLE computation
import h5py
import numpy as np

# --- Load the ABC flow data ---
file_path = os.path.join(os.path.dirname(__file__), 'abc_flow_data.h5')
with h5py.File(file_path, 'r') as f:
    velocity_points = f['points'][:]                 # shape (M, 3)
    velocity_vectors = f['vectors'][:]               # shape (M, 3, T)
    time_steps = f['time_steps'][:]                  # shape (T,)

# --- Define 3D grid for initial particle positions (must be inside the domain) ---
x = np.linspace(0, 2*np.pi, 2)
y = np.linspace(0, 2*np.pi, 2)
z = np.linspace(0, 2*np.pi, 2)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

X = X.T
Y = Y.T
Z = Z.T

# --- FTLE parameters ---
initial_time = time_steps[0]
final_time = time_steps[-1]
dt = 1

# --- Run FTLE computation (with plotting enabled) ---
ftle, traj, iso, bftle, btraj, biso = run_FTLE_3d(
    velocity_points,
    velocity_vectors,
    X,
    Y,
    Z,
    dt,
    initial_time,
    final_time,
    time_steps,
    plot_ftle=True,
    save_plot_path=None  # or a path like 'abc_ftle_output.png'
)
