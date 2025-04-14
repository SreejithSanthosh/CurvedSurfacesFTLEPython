import numpy as np
from scipy.interpolate import LinearNDInterpolator
from numba import njit
import math
from .utilities import plot_ftle_2d, plot_ftle_3d, interpolate





@njit
def FTLE_2d_compute(x_initial, y_initial, x_final, y_final, time, index_shift=1):
    """
    Compute FTLE field on a uniform 2D grid using finite differences.

    Parameters:
        x_initial, y_initial: 2D arrays of initial grid positions.
        x_final, y_final: 2D arrays of advected grid positions.
        time (float): Total advection time.
        index_shift (int): Grid spacing for central difference.

    Returns:
        FTLE (2D ndarray): Finite-time Lyapunov exponent values.
    """
    nx, ny = x_initial.shape
    FTLE = np.full((nx, ny), np.nan)
    F = np.zeros((2, 2))

    for i in range(index_shift, nx - index_shift):
        for j in range(index_shift, ny - index_shift):

            # Skip NaNs in initial positions
            if math.isnan(x_initial[i, j]) or math.isnan(y_initial[i, j]):
                continue

            # Local grid spacing
            dx = x_initial[i + index_shift, j] - x_initial[i - index_shift, j]
            dy = y_initial[i, j + index_shift] - y_initial[i, j - index_shift]

            if dx == 0 or dy == 0:
                continue

            # Compute finite difference deformation gradient ∂Xf/∂X0
            F[0, 0] = (x_final[i + index_shift, j] - x_final[i - index_shift, j]) / (2 * dx)
            F[0, 1] = (x_final[i, j + index_shift] - x_final[i, j - index_shift]) / (2 * dy)
            F[1, 0] = (y_final[i + index_shift, j] - y_final[i - index_shift, j]) / (2 * dx)
            F[1, 1] = (y_final[i, j + index_shift] - y_final[i, j - index_shift]) / (2 * dy)

            # Cauchy-Green strain tensor: C = Fᵀ F
            C = F.T @ F

            if np.isnan(C).any() or np.isinf(C).any():
                continue

            # Maximum eigenvalue of C
            eigenvalues = np.linalg.eigvalsh(C)
            max_eigenvalue = np.max(eigenvalues)

            if max_eigenvalue <= 0:
                continue

            FTLE[i, j] = (1 / (2 * time)) * np.log(np.sqrt(max_eigenvalue))

    return FTLE


@njit
def FTLE_3d_compute(x_initial, y_initial, z_initial, x_final, y_final, z_final, time, index_shift=1):
    nx, ny, nz = x_initial.shape
    FTLE = np.full((nx, ny, nz), np.nan)
    F_right = np.zeros((3, 3))

    for z_index in range(index_shift, nz - index_shift):
        for x_index in range(index_shift, nx - index_shift):
            for y_index in range(index_shift, ny - index_shift):

                if math.isnan(x_initial[x_index, y_index, z_index]) or math.isnan(y_initial[x_index, y_index, z_index]):
                    continue

                dx = x_initial[x_index + index_shift, y_index, z_index] - x_initial[x_index - index_shift, y_index, z_index]
                dy = y_initial[x_index, y_index + index_shift, z_index] - y_initial[x_index, y_index - index_shift, z_index]
                dz = z_initial[x_index, y_index, z_index + index_shift] - z_initial[x_index, y_index, z_index - index_shift]

                if dx == 0 or dy == 0 or dz == 0:
                    continue

                # ∂Xf/∂X0 (deformation gradient matrix, F_right)
                F_right[0, 0] = (x_final[x_index + index_shift, y_index, z_index] - x_final[x_index - index_shift, y_index, z_index]) / (2 * dx)
                F_right[0, 1] = (x_final[x_index, y_index + index_shift, z_index] - x_final[x_index, y_index - index_shift, z_index]) / (2 * dy)
                F_right[0, 2] = (x_final[x_index, y_index, z_index + index_shift] - x_final[x_index, y_index, z_index - index_shift]) / (2 * dz)

                F_right[1, 0] = (y_final[x_index + index_shift, y_index, z_index] - y_final[x_index - index_shift, y_index, z_index]) / (2 * dx)
                F_right[1, 1] = (y_final[x_index, y_index + index_shift, z_index] - y_final[x_index, y_index - index_shift, z_index]) / (2 * dy)
                F_right[1, 2] = (y_final[x_index, y_index, z_index + index_shift] - y_final[x_index, y_index, z_index - index_shift]) / (2 * dz)

                F_right[2, 0] = (z_final[x_index + index_shift, y_index, z_index] - z_final[x_index - index_shift, y_index, z_index]) / (2 * dx)
                F_right[2, 1] = (z_final[x_index, y_index + index_shift, z_index] - z_final[x_index, y_index - index_shift, z_index]) / (2 * dy)
                F_right[2, 2] = (z_final[x_index, y_index, z_index + index_shift] - z_final[x_index, y_index, z_index - index_shift]) / (2 * dz)

                # Cauchy-Green strain tensor
                C = F_right.T @ F_right

                if np.isnan(C).any() or np.isinf(C).any():
                    continue

                eigenvalues = np.linalg.eigh(C)[0]
                max_eigen = np.max(eigenvalues)

                if max_eigen <= 0:
                    continue

                FTLE[x_index, y_index, z_index] = (1 / (2 * time)) * np.log(np.sqrt(max_eigen))

    return FTLE






def FTLE_2d_dense(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    dt,
    initial_time,
    final_time,
    time_steps,
    direction,
    time_indepedent=False,
    plot_ftle=False
):
    """
    Advects a uniform grid of particles using a sparse velocity field with RK4 integration.

    Parameters:
        velocity_points (ndarray): (M, 2) array of known velocity locations, fixed in time.
        velocity_vectors (ndarray): (M, 2) or (M, 2, T) array of velocity vectors at those locations.
        x_grid_parts (ndarray): 2D array of X-coordinates from np.meshgrid.
        y_grid_parts (ndarray): 2D array of Y-coordinates from np.meshgrid.
        dt (float): Time step size for integration (0 < dt <= 1).
        initial_time (int): Start index for advection.
        final_time (int): End index for advection.
        time_steps (ndarray): 1D array of integer time steps.
        direction (str): "forward" or "backward" advection.
        time_indepedent (bool): Whether the velocity field is time-independent.

    Returns:
        ftle (ndarray): Flattened array of FTLE values.
        trajectories (ndarray): (N, 2, T) array of particle positions over time.
    """

    # --- Error checking ---
    if dt > 1 or dt <= 0:
        raise ValueError("Error: dt must be in the interval (0,1]")
    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Error: Initial/final time must be in the given time steps")
    if initial_time == final_time:
        raise ValueError("Error: Initial and final times must differ.")

    direction = direction.lower()
    if direction == "forward":
        if initial_time > final_time:
            raise ValueError("Error: Forward advection requires initial_time < final_time")
    elif direction == "backward":
        if initial_time < final_time:
            raise ValueError("Error: Backward advection requires initial_time > final_time")

        # Reverse time indexing
        temp_initial_time = len(time_steps) - final_time - 1
        final_time = len(time_steps) - initial_time - 1
        initial_time = final_time
        final_time = temp_initial_time

        if not time_indepedent:
            velocity_vectors = velocity_vectors[:, :, ::-1]
        dt = -dt

    # --- Setup grid and particle data ---
    x_dim1, x_dim2 = x_grid_parts.shape
    y_dim1, y_dim2 = y_grid_parts.shape

    particles_positions = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten()]).T
    num_particles = particles_positions.shape[0]

    fine_time = np.arange(initial_time, final_time + np.abs(dt), np.abs(dt))
    fine_time_length = len(fine_time)

    trajectories = np.zeros((num_particles, 2, fine_time_length))
    trajectories[:, :, 0] = particles_positions

    fine_time = fine_time[:-1]  # Exclude final point (we advect up to t_index + 1)

    # --- Time-independent advection ---
    if time_indepedent:
        interp_u = LinearNDInterpolator(velocity_points, velocity_vectors[:, 0], fill_value=0)
        interp_v = LinearNDInterpolator(velocity_points, velocity_vectors[:, 1], fill_value=0)

        for t_index, _ in enumerate(fine_time):
            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]

            # RK4 steps
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

    # --- Time-dependent advection ---
    else:
        for t_index, t in enumerate(fine_time):
            t_floor = int(np.floor(t))
            t_ceiling = int(np.ceil(t))
            t_fraction = t - t_floor  

            # Interpolate velocity at current time step
            u_interp = interpolate(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceiling], t_fraction)
            v_interp = interpolate(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceiling], t_fraction)

            interp_u = LinearNDInterpolator(velocity_points, u_interp, fill_value=0)
            interp_v = LinearNDInterpolator(velocity_points, v_interp, fill_value=0)

            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]

            # RK4 steps
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

    # --- Compute FTLE from reshaped results ---
    x_traj = trajectories[:, 0, :].reshape(x_dim1, x_dim2, fine_time_length)
    y_traj = trajectories[:, 1, :].reshape(y_dim1, y_dim2, fine_time_length)

    ftle = FTLE_2d_compute(
        x_grid_parts, y_grid_parts,
        x_traj[:, :, -1], y_traj[:, :, -1],
        np.abs(final_time - initial_time)
    )

    if plot_ftle:
        particles_positions = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten()]).T
        plot_FLTE_2d(particles_positions, ftle)

    return ftle.flatten(), trajectories



def FTLE_3d_dense(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    z_grid_parts,
    dt,
    initial_time,
    final_time,
    time_steps,
    direction,
    time_indepedent=False,
    plot_ftle=False
):
    """
    Advects a uniform 3D grid of particles using a sparse velocity field with RK4 integration.

    Parameters:
        velocity_points (ndarray): (M, 3) array of known velocity locations, fixed in time.
        velocity_vectors (ndarray): (M, 3) if time-independent, or (M, 3, T) if time-dependent.
        x_grid_parts, y_grid_parts, z_grid_parts (ndarray): 3D meshgrid arrays of particle positions.
        dt (float): Time step size for integration (0 < dt <= 1).
        initial_time (int): Start index for advection.
        final_time (int): End index for advection.
        time_steps (ndarray): Array of time step indices.
        direction (str): "forward" or "backward".
        time_indepedent (bool): Whether velocity is independent of time.

    Returns:
        ftle (ndarray): Flattened FTLE values.
        trajectories (ndarray): (N, 3, T) particle positions over time.
    """

    # --- Error checking ---
    if dt > 1 or dt <= 0:
        raise ValueError("Error: dt must be in the interval (0, 1].")
    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Error: Initial/final time must be in time_steps.")
    if initial_time == final_time:
        raise ValueError("Error: initial_time and final_time must differ.")

    direction = direction.lower()
    if direction == "forward":
        if initial_time > final_time:
            raise ValueError("Error: Forward advection requires initial_time < final_time.")
    elif direction == "backward":
        if initial_time < final_time:
            raise ValueError("Error: Backward advection requires initial_time > final_time.")

        # Time reversal
        temp_initial_time = len(time_steps) - final_time - 1
        final_time = len(time_steps) - initial_time - 1
        initial_time = final_time
        final_time = temp_initial_time

        if not time_indepedent:
            velocity_vectors = velocity_vectors[:, :, ::-1]
        dt = -dt

    # --- Grid setup ---
    x_dim1, x_dim2, x_dim3 = x_grid_parts.shape
    y_dim1, y_dim2, y_dim3 = y_grid_parts.shape
    z_dim1, z_dim2, z_dim3 = z_grid_parts.shape

    particle_positions = np.vstack([
        x_grid_parts.flatten(),
        y_grid_parts.flatten(),
        z_grid_parts.flatten()
    ]).T

    num_particles = particle_positions.shape[0]

    fine_time = np.arange(initial_time, final_time + np.abs(dt), np.abs(dt))
    fine_time_length = len(fine_time)

    trajectories = np.zeros((num_particles, 3, fine_time_length))
    trajectories[:, :, 0] = particle_positions

    fine_time = fine_time[:-1]  # Integrate up to last point

    # --- Time-independent advection ---
    if time_indepedent:
        interp_u = LinearNDInterpolator(velocity_points, velocity_vectors[:, 0], fill_value=0)
        interp_v = LinearNDInterpolator(velocity_points, velocity_vectors[:, 1], fill_value=0)
        interp_w = LinearNDInterpolator(velocity_points, velocity_vectors[:, 2], fill_value=0)

        for t_index, _ in enumerate(fine_time):
            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]
            z_curr = trajectories[:, 2, t_index]

            k1_x = interp_u(x_curr, y_curr, z_curr)
            k1_y = interp_v(x_curr, y_curr, z_curr)
            k1_z = interp_w(x_curr, y_curr, z_curr)

            k2_x = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z)
            k2_y = interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z)
            k2_z = interp_w(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z)

            k3_x = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z)
            k3_y = interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z)
            k3_z = interp_w(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z)

            k4_x = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z)
            k4_y = interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z)
            k4_z = interp_w(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z)

            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            z_next = z_curr + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)

            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next
            trajectories[:, 2, t_index + 1] = z_next

    # --- Time-dependent advection ---
    else:
        for t_index, t in enumerate(fine_time):
            t_floor = int(np.floor(t))
            t_ceiling = int(np.ceil(t))
            t_fraction = t - t_floor  

            # Interpolate velocity field in time
            u_interp = interpolate(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceiling], t_fraction)
            v_interp = interpolate(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceiling], t_fraction)
            w_interp = interpolate(velocity_vectors[:, 2, t_floor], velocity_vectors[:, 2, t_ceiling], t_fraction)

            interp_u = LinearNDInterpolator(velocity_points, u_interp, fill_value=0)
            interp_v = LinearNDInterpolator(velocity_points, v_interp, fill_value=0)
            interp_w = LinearNDInterpolator(velocity_points, w_interp, fill_value=0)

            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]
            z_curr = trajectories[:, 2, t_index]

            k1_x = interp_u(x_curr, y_curr, z_curr)
            k1_y = interp_v(x_curr, y_curr, z_curr)
            k1_z = interp_w(x_curr, y_curr, z_curr)

            k2_x = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z)
            k2_y = interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z)
            k2_z = interp_w(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z)

            k3_x = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z)
            k3_y = interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z)
            k3_z = interp_w(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z)

            k4_x = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z)
            k4_y = interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z)
            k4_z = interp_w(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z)

            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            z_next = z_curr + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)

            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next
            trajectories[:, 2, t_index + 1] = z_next

    # --- Reshape trajectories and compute FTLE ---
    x_traj = trajectories[:, 0, :].reshape(x_dim1, x_dim2, x_dim3, fine_time_length)
    y_traj = trajectories[:, 1, :].reshape(y_dim1, y_dim2, y_dim3, fine_time_length)
    z_traj = trajectories[:, 2, :].reshape(z_dim1, z_dim2, z_dim3, fine_time_length)

    ftle = FTLE_3d_compute(
        x_grid_parts, y_grid_parts, z_grid_parts,
        x_traj[:, :, :, -1], y_traj[:, :, :, -1], z_traj[:, :, :, -1],
        np.abs(final_time - initial_time)
    )

    if plot_ftle:
        particles_positions = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten()], z_grid_parts.flatten()).T
        plot_FTLE_3d(particles_positions, ftle)
        
    return ftle.flatten(), trajectories
