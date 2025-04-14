
from .utilities import plot_ftle_2d, plot_ftle_3d, interpolate
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from scipy.spatial import cKDTree


def compute_Ftle_sparse(initial_positions, final_positions, initial_time, final_time, k=5):
    """
    Compute the FTLE values for a sparse set of particle positions in 2D or 3D.

    Parameters:
        initial_positions (ndarray): (N, d) array of initial particle positions.
        final_positions (ndarray): (N, d) array of final particle positions.
        initial_time (float): Initial time of advection.
        final_time (float): Final time of advection.
        k (int): Number of nearest neighbors to use for least squares.

    Returns:
        ndarray: (N,) array of FTLE values for each initial position.
    """
    num_particles, dim = initial_positions.shape
    tree = cKDTree(initial_positions)

    total_time = np.abs(final_time - initial_time)
    ftle_values = np.zeros(num_particles)

    for i in range(num_particles):
        neighbors_idx = tree.query(initial_positions[i], k=k)[1]
        X0 = initial_positions[neighbors_idx]  # shape (k, d)
        Xf = final_positions[neighbors_idx]    # shape (k, d)

        # Solve X0 * A.T ≈ Xf → A: deformation gradient (d x d)
        A, _, _, _ = np.linalg.lstsq(X0, Xf, rcond=None)  # A: (d, d)
        A = A.T

        C = A.T @ A

        # Largest eigenvalue of C
        lambda_max = np.max(np.linalg.eigvalsh(C))  # eigvalsh is safer for symmetric C

        
        if lambda_max > 0:
            ftle_values[i] = (1 / (2 * total_time)) * np.log(lambda_max)
        else:
            ftle_values[i] = 0.0  # In case of numerical noise

    return ftle_values




def FTLE_2d_sparse(
    velocity_points,
    velocity_vectors,
    particle_positions,
    dt,
    initial_time,
    final_time,
    time_steps,
    direction,
    time_indepedent=False,
    plot_ftle=False,
    neighborhood=10,
    lam=1e-10
):
    """
    Advects particles using a sparse velocity field with RK4 integration.

    Parameters:
        velocity_points (ndarray): (M, 2) array of known velocity locations, fixed in time.
        velocity_vectors (ndarray): (M, 2) array of velocity vectors at those locations.
                                    If time-dependent, this is of shape (M, 2, T).
        particle_positions (ndarray): (N, 2) array of initial particle positions.
        dt (float): Time step size for integration. Must satisfy 0 < dt <= 1.
        initial_time (int): Index of initial time.
        final_time (int): Index of final time.
        time_steps (ndarray): Array of integer time step indices.
        direction (str): "forward" or "backward" advection.
        time_indepedent (bool): Whether velocity is time-independent.
        neighborhood (int): Number of neighbors to consider for FTLE computation.
        lam (float): Small value for numerical stability in FTLE.

    Returns:
        ftle (ndarray): (N,) array of FTLE values.
        trajectories (ndarray): (N, 2, T) array of particle positions over time.
    """
    
    # --- Error checking ---
    if dt > 1 or dt <= 0:
        raise ValueError("Error: dt must be in the interval (0, 1].")
    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Error: Initial/final time must be in the given time steps.")
    if initial_time == final_time:
        raise ValueError("Error: Initial and final times must differ.")

    direction = direction.lower()

    if direction == "forward":
        if initial_time > final_time:
            raise ValueError("Error: Forward advection requires initial_time < final_time.")
    elif direction == "backward":
        if initial_time < final_time:
            raise ValueError("Error: Backward advection requires initial_time > final_time.")
        
        # Reverse time indexing
        temp_initial_time = len(time_steps) - final_time - 1
        final_time = len(time_steps) - initial_time - 1
        initial_time = final_time
        final_time = temp_initial_time

        if not time_indepedent:
            velocity_vectors = velocity_vectors[:, :, ::-1]
        dt = -dt

    num_particles = particle_positions.shape[0]

    # Fine-grained time for RK4 integration
    fine_time = np.arange(initial_time, final_time + np.abs(dt), np.abs(dt))
    fine_time_length = len(fine_time)

    # Initialize trajectory storage
    trajectories = np.zeros((num_particles, 2, fine_time_length))
    trajectories[:, :, 0] = particle_positions

    fine_time = fine_time[:-1]  # We only loop up to the last available time step

    # --- Time-independent velocity field ---
    if time_indepedent:
        interp_u = LinearNDInterpolator(velocity_points, velocity_vectors[:, 0], fill_value=0)
        interp_v = LinearNDInterpolator(velocity_points, velocity_vectors[:, 1], fill_value=0)

        for t_index, _ in enumerate(fine_time):
            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]

            # RK4 integration
            k1_x, k1_y = interp_u(x_curr, y_curr), interp_v(x_curr, y_curr)
            k2_x, k2_y = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y), \
                         interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y)
            k3_x, k3_y = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y), \
                         interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y)
            k4_x, k4_y = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y), \
                         interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y)

            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next

    # --- Time-dependent velocity field ---
    else:
        for t_index, t in enumerate(fine_time):
            t_floor = int(np.floor(t))
            t_ceiling = int(np.ceil(t))
            t_fraction = t - t_floor  

            # Interpolate velocity vectors at this time
            u_interp = interpolate(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceiling], t_fraction)
            v_interp = interpolate(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceiling], t_fraction)

            interp_u = LinearNDInterpolator(velocity_points, u_interp, fill_value=0)
            interp_v = LinearNDInterpolator(velocity_points, v_interp, fill_value=0)

            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]

            # RK4 integration
            k1_x, k1_y = interp_u(x_curr, y_curr), interp_v(x_curr, y_curr)
            k2_x, k2_y = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y), \
                         interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y)
            k3_x, k3_y = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y), \
                         interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y)
            k4_x, k4_y = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y), \
                         interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y)

            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next

    # Compute FTLE on the final positions
    ftle = compute_Ftle_sparse(particle_positions, trajectories[:, :, -1], initial_time, final_time, neighborhood)

    if plot_ftle:
        plot_ftle_2d(particle_positions, ftle)

    return ftle, trajectories


def FTLE_3d_sparse(
    velocity_points,
    velocity_vectors,
    particle_positions,
    dt,
    initial_time,
    final_time,
    time_steps,
    direction,
    time_independent=False,
    plot_ftle=False,
    neighborhood=10,
    lam=1e-10
):
    """
    Advects particles using a sparse 3D velocity field with RK4 integration.

    Parameters:
        velocity_points (ndarray): (M, 3) array of known velocity locations, fixed in time.
        velocity_vectors (ndarray): (M, 3) if time-independent, or (M, 3, T) if time-dependent.
        particle_positions (ndarray): (N, 3) array of initial particle positions.
        dt (float): Time step size for integration (0 < dt <= 1).
        initial_time (int): Starting index in time_steps.
        final_time (int): Ending index in time_steps.
        time_steps (ndarray): Integer-valued array of valid time steps.
        direction (str): 'forward' or 'backward'.
        time_independent (bool): Whether the velocity field is time-invariant.
        neighborhood (int): Neighbor count for FTLE computation.
        lam (float): Numerical stability parameter for FTLE.

    Returns:
        ftle (ndarray): Array of FTLE values at initial positions.
        trajectories (ndarray): (N, 3, T) array of particle positions over time.
    """
    # --- Error checking ---
    if dt > 1 or dt <= 0:
        raise ValueError("dt must be in (0, 1].")
    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Initial and final times must be in time_steps.")
    if initial_time == final_time:
        raise ValueError("Initial and final times must be different.")

    direction = direction.lower()
    if direction == "forward":
        if initial_time > final_time:
            raise ValueError("For forward advection, initial_time must be < final_time.")
    elif direction == "backward":
        if initial_time < final_time:
            raise ValueError("For backward advection, initial_time must be > final_time.")

        # Reverse time indexing
        temp_initial_time = len(time_steps) - final_time - 1
        final_time = len(time_steps) - initial_time - 1
        initial_time = final_time
        final_time = temp_initial_time

        if not time_independent:
            velocity_vectors = velocity_vectors[:, :, ::-1]
        dt = -dt

    # --- Setup ---
    num_particles = particle_positions.shape[0]
    fine_time = np.arange(initial_time, final_time + np.abs(dt), np.abs(dt))
    fine_time_length = len(fine_time)

    trajectories = np.zeros((num_particles, 3, fine_time_length))
    trajectories[:, :, 0] = particle_positions
    fine_time = fine_time[:-1]

    # --- Time-independent advection ---
    if time_independent:
        interp_u = LinearNDInterpolator(velocity_points, velocity_vectors[:, 0], fill_value=0)
        interp_v = LinearNDInterpolator(velocity_points, velocity_vectors[:, 1], fill_value=0)
        interp_w = LinearNDInterpolator(velocity_points, velocity_vectors[:, 2], fill_value=0)

        for t_index, _ in enumerate(fine_time):
            x = trajectories[:, 0, t_index]
            y = trajectories[:, 1, t_index]
            z = trajectories[:, 2, t_index]

            # RK4 steps
            k1_x, k1_y, k1_z = interp_u(x, y, z), interp_v(x, y, z), interp_w(x, y, z)
            k2_x, k2_y, k2_z = interp_u(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z), \
                               interp_v(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z), \
                               interp_w(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z)
            k3_x, k3_y, k3_z = interp_u(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z), \
                               interp_v(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z), \
                               interp_w(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z)
            k4_x, k4_y, k4_z = interp_u(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z), \
                               interp_v(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z), \
                               interp_w(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z)

            x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            y_next = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
            z_next = z + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next
            trajectories[:, 2, t_index + 1] = z_next

    # --- Time-dependent advection ---
    else:
        for t_index, t in enumerate(fine_time):
            t_floor = int(np.floor(t))
            t_ceil = int(np.ceil(t))
            t_fraction = t - t_floor  

            # Interpolate velocity field at this fractional time
            u_interp = interpolate(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceil], t_fraction)
            v_interp = interpolate(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceil], t_fraction)
            w_interp = interpolate(velocity_vectors[:, 2, t_floor], velocity_vectors[:, 2, t_ceil], t_fraction)

            interp_u = LinearNDInterpolator(velocity_points, u_interp, fill_value=0)
            interp_v = LinearNDInterpolator(velocity_points, v_interp, fill_value=0)
            interp_w = LinearNDInterpolator(velocity_points, w_interp, fill_value=0)

            x = trajectories[:, 0, t_index]
            y = trajectories[:, 1, t_index]
            z = trajectories[:, 2, t_index]

            # RK4 steps
            k1_x, k1_y, k1_z = interp_u(x, y, z), interp_v(x, y, z), interp_w(x, y, z)
            k2_x, k2_y, k2_z = interp_u(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z), \
                               interp_v(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z), \
                               interp_w(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z)
            k3_x, k3_y, k3_z = interp_u(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z), \
                               interp_v(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z), \
                               interp_w(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z)
            k4_x, k4_y, k4_z = interp_u(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z), \
                               interp_v(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z), \
                               interp_w(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z)

            x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            y_next = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
            z_next = z + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next
            trajectories[:, 2, t_index + 1] = z_next

    # --- FTLE computation ---
    ftle = compute_Ftle_sparse(particle_positions, trajectories[:, :, -1], initial_time, final_time, neighborhood)

    if plot_ftle:
        plot_ftle_3d(particle_positions, ftle)

    return ftle, trajectories

