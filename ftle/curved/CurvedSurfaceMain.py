from scipy.spatial import cKDTree
from numba import njit
import numpy as np
from scipy.spatial import cKDTree
from itertools import combinations
from .advection import RK4_particle_advection
from .FTLECompute import FTLE_compute
from .utilities import plot_FTLE_mesh






def FTLE_mesh(
    node_connections,           # List[np.ndarray], each (M_t, 3)
    node_positions,             # List[np.ndarray], each (N_t, 3)
    node_velocities,            # List[np.ndarray], each (N_t, 3)
    particle_positions,         # (P, 3)
    initial_time,               # int
    final_time,                 # int
    time_steps,                 # (T,)
    direction="forward",        # "forward" or "backward"
    plot_ftle=False,            # If True, calls plot_ftle_mesh
    save_path = None,
    camera_setup = None,
    neighborhood=15,            # For FTLE computation
    lam=1e-10                   # Regularization
):
    """
    Run a particle advection and FTLE computation on a triangulated surface (staggered mesh compatible).
    """
    old_it = initial_time
    old_ft = final_time
    time_length = len(time_steps)


    direction = direction.lower()

    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Initial and final times must be in `time_steps`.")
    if initial_time == final_time:
        raise ValueError("Initial and final time steps must differ.")

    # Reverse time indexing if backward
    if direction == "backward":
        if initial_time < final_time:
            raise ValueError("Backward advection: initial_time must be > final_time")

        # Reverse data
        node_connections = node_connections[::-1]
        node_positions = node_positions[::-1]
        node_velocities = node_velocities[::-1] # reverse vector field direction
        time_steps = time_steps[::-1]
        
        for i in range(len(node_velocities)):
            for j in range(len(node_velocities[i])):
                for k in range(len(node_velocities[i][j])):
                    node_velocities[i][j][k] *= -1
                    
        # Update to reflect reversed time axis
        initial_time = time_length - initial_time -1
        final_time = time_length - final_time -1

    else:
        if initial_time > final_time:
            raise ValueError("Forward advection: final_time must be > initial_time")
            
    # Run RK4 advection
    x_traj, y_traj, z_traj, centroids = RK4_particle_advection(
        node_connections,
        node_positions,
        node_velocities,
        particle_positions,
        initial_time,
        final_time
    )

    if x_traj is None or y_traj is None or z_traj is None:
        raise RuntimeError("Trajectory computation returned None")

    final_positions = np.vstack([x_traj[:, -1], y_traj[:, -1], z_traj[:, -1]]).T

    # Compute FTLE



    ftle = FTLE_compute(
        node_connections,
        node_positions,
        centroids,
        particle_positions,
        final_positions,
        initial_time,
        final_time,
        neighborhood,
        lam
    )

    if ftle is None:
        raise RuntimeError("FTLE computation returned None")

    if plot_ftle:
        plot_FTLE_mesh(node_connections, node_positions, old_it, old_ft, ftle, direction, save_path, camera_setup)

    return ftle, np.stack([x_traj, y_traj, z_traj], axis=-1)
