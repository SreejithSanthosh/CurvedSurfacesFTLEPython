from scipy.spatial import cKDTree
from numba import njit
import numpy as np
from scipy.spatial import cKDTree
from itertools import combinations
from .advection import RK4_particle_advection
from .utilities import plot_FTLE_mesh




@njit
def local_tangent_project(A, B, C, position):
    """
    Project the current position to the plane formed by points A, B, C

    Parameters:
    A, B, C: Points in R^3
    position: The point to project onto the ABC plane, assuming it's already on the plane

    Returns:
    x_local: The coordinates of the position in the local tangent plane
    """
    vec1 = B - A
    vec2 = C - A

    # Normalize the vectors to form an orthonormal basis (this assumes vec1 and vec2 are not collinear)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 - np.dot(vec2, vec1) * vec1  # Orthogonalize vec2 with respect to vec1
    vec2 = vec2 / np.linalg.norm(vec2)

    # Calculate the vector from A to the position
    vec3 = position - A

    # Project vec3 onto the vectors
    return np.array([np.dot(vec3, vec1), np.dot(vec3, vec2)])




def FTLE_compute(node_connections, node_positions, centroids, initial_positions, final_positions, initial_time, final_time, neighborhood=15, lam=1e-10):
    """
    Computes FTLE for staggered mesh data. This version supports time-dependent, non-uniform triangulations and vertex positions.
    
    Parameters:
        node_connections (list of ndarray): list of (M_t, 3) arrays of node indices for each time step
        node_positions (list of ndarray): list of (N_t, 3) arrays of node positions for each time step
        centroids (list of ndarray): list of (M_t, 3) arrays of centroids for each time step
        initial_positions (ndarray): (P, 3) array of initial particle positions
        final_positions (ndarray): (P, 3) array of final particle positions
        initial_time (int): initial time step index
        final_time (int): final time step index
        neighborhood (int): number of neighbors to consider
        lam (float): regularization parameter

    Returns:
        FTLE (ndarray): (P,) FTLE values
    """


    position_kdtree = cKDTree(initial_positions)
    centroid_kdtree_initial = cKDTree(centroids[initial_time])
    centroid_kdtree_final = cKDTree(centroids[final_time])
    num_points = initial_positions.shape[0]
    FTLE = np.zeros(num_points)

    for i in range(num_points - 1):
        _, closest_indexes = position_kdtree.query(initial_positions[i], neighborhood + 1)
        _, face_idx_initial = centroid_kdtree_initial.query(initial_positions[i])
        _, face_idx_final = centroid_kdtree_final.query(final_positions[i])

        face_initial = node_connections[initial_time][face_idx_initial]
        face_final = node_connections[final_time][face_idx_final]

        pos_face_initial = node_positions[initial_time][face_initial]
        pos_face_final = node_positions[final_time][face_final]

        I_closest = initial_positions[closest_indexes[1:]]
        F_closest = final_positions[closest_indexes[1:]]

        I_local_coords = np.zeros((neighborhood, 2))
        F_local_coords = np.zeros((neighborhood, 2))

        for j in range(neighborhood):
            I_local_coords[j] = local_tangent_project(pos_face_initial[0], pos_face_initial[1], pos_face_initial[2], I_closest[j])
            F_local_coords[j] = local_tangent_project(pos_face_final[0], pos_face_final[1], pos_face_final[2], F_closest[j])

        combs = list(combinations(range(neighborhood), 2))
        ind1 = [c[0] for c in combs]
        ind2 = [c[1] for c in combs]

        X = np.zeros((2, len(combs)))
        Y = np.zeros((2, len(combs)))
        X[0, :] = I_local_coords[ind1, 0] - I_local_coords[ind2, 0]
        X[1, :] = I_local_coords[ind1, 1] - I_local_coords[ind2, 1]
        Y[0, :] = F_local_coords[ind1, 0] - F_local_coords[ind2, 0]
        Y[1, :] = F_local_coords[ind1, 1] - F_local_coords[ind2, 1]

        A = Y @ X.T + lam * len(closest_indexes) * np.eye(2)
        B = X @ X.T + lam * len(closest_indexes) * np.eye(2)
        DF = A @ np.linalg.inv(B)
        FTLE[i] = np.log(np.linalg.norm(DF, 2)) / abs(final_time - initial_time)

    return FTLE





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
        node_velocities = node_velocities[::-1]
        time_steps = time_steps[::-1]

        # Update to reflect reversed time axis
        initial_time = time_length - initial_time -1
        final_time = time_length - final_time -1

        print("old_it: ", old_it)
        print("old_ft: ", old_ft)
        print("time_length: ", time_length)
        

        print("initial_time: ", initial_time)
        print("final_time: ", final_time)
        input("Press enter to continue" )
        
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
