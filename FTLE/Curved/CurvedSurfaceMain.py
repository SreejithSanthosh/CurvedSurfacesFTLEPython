from scipy.spatial import cKDTree
from numba import njit
import numpy as np
from scipy.spatial import cKDTree
from itertools import combinations
from FTLE.Curved.advection import RK4_particle_advection
from FTLE.Curved.FTLECompute import FTLE_compute
from FTLE.Curved.utilities import plot_FTLE_mesh, plot_FTLE_mesh4






def FTLE_mesh(
    node_connections,           # List[np.ndarray], each (M_t, 3)
    node_positions,             # List[np.ndarray], each (N_t, 3)
    node_velocities,            # List[np.ndarray], each (N_t, 3)
    particle_positions,         # (P, 3)
    initial_time,               # int
    final_time,                 # int
    time_steps,                 # (T)
    plot_ftle=False,            # If True, calls plot_ftle_mesh
    save_path = None,
    camera_setup = None,
    neighborhood=15,            # For FTLE computation
    lam=1e-10                   # Regularization
):
    """
    Run a particle advection and FTLE computation on a triangulated surface (staggered mesh compatible).
    """
    initial_time
    final_time

    if initial_time >= final_time:
        raise ValueError("Initial time must be less than final time")

    time_length = len(time_steps)


    direction = direction.lower()

    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Initial and final times must be in `time_steps`.")
    if initial_time == final_time:
        raise ValueError("Initial and final time steps must differ.")

    # Find time indexes
    initial_time_index = time_steps.index(initial_time)
    final_time_index = time_steps.index(final_time)

    # Run forward RK4 advection
    x_traj, y_traj, z_traj, centroids = RK4_particle_advection(
        node_connections,
        node_positions,
        node_velocities,
        particle_positions,
        initial_time_index,
        final_time_index
    )
   

    # Reverse data for backward computation
    back_node_connections = node_connections[::-1]
    back_node_positions = node_positions[::-1]
    back_node_velocities = node_velocities[::-1] # reverse vector field direction(this is the same as reversing the RK4 scheme directly)
    back_time_steps = time_steps[::-1]
    
    for i in range(len(node_velocities)):
        for j in range(len(node_velocities[i])):
            for k in range(len(node_velocities[i][j])):
                back_node_velocities[i][j][k] *= -1
                
    # Update to reflect reversed time axis
    initial_time_index = time_length - initial_time_index -1
    final_time_index = time_length - final_time_index -1


     # Run forward RK4 advection
    back_x_traj, back_y_traj, back_z_traj, back_centroids = RK4_particle_advection(
        back_node_connections,
        back_node_positions,
        back_node_velocities,
        particle_positions,
        initial_time,
        final_time
    )
    if x_traj is None or y_traj is None or z_traj is None:
        raise RuntimeError("Trajectory computation returned None")

    # Fetch the final positions of the advection for ftle computation
    final_positions = np.vstack([x_traj[:, -1], y_traj[:, -1], z_traj[:, -1]]).T
    back_final_positions = np.vstack([back_x_traj[:, -1], back_y_traj[:, -1], back_z_traj[:, -1]]).T
    
    # Compute FTLE
    ftle, isotropy = FTLE_compute(
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

    back_ftle, back_isotropy = FTLE_compute(
        back_node_connections,
        back_node_positions,
        back_centroids,
        particle_positions,
        back_final_positions,
        initial_time,
        final_time,
        neighborhood,
        lam
    )

    if ftle is None:
        raise RuntimeError("FTLE computation returned None")

    if plot_ftle:
        plot_FTLE_mesh4(node_connections,
        node_positions,
        ftle,
        isotropy,
        back_node_connections,
        back_node_positions,
        back_ftle,
        back_isotropy,
        initial_time,
        final_time, save_path, camera_setup)


    return ftle, np.stack([x_traj, y_traj, z_traj], axis=-1), isotropy, back_ftle, np.stack([back_x_traj, back_y_traj, back_z_traj], axis=-1), back_isotropy
