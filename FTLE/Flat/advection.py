from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import numpy as np



def RK4_advection_3d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):
    
    def interpolate(floor_data, ceiling_data, t_fraction):
        return t_fraction*ceiling_data + (1-t_fraction)*floor_data
        
    # --- Time-independent advection ---
    if time_independent:
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
            print(t)
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

    return trajectories

def RK4_advection_2d_old(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):

    def interpolate(floor_data, ceiling_data, t_fraction):
        return t_fraction*ceiling_data + (1-t_fraction)*floor_data
        
    if time_independent:
        interp_u = LinearNDInterpolator(velocity_points, velocity_vectors[:, 0], fill_value=0)
        interp_v = LinearNDInterpolator(velocity_points, velocity_vectors[:, 1], fill_value=0)
    
        for t_index, _ in enumerate(fine_time):
            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]
    
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
        
    return trajectories


from numba import njit


from scipy.spatial import KDTree
def average_nearest_neighbor_distance(points):
    """
    Computes the average distance to the nearest neighbor for each point.

    Parameters:
        points (ndarray): shape (N, d), where each row is a point in d-dimensional space.

    Returns:
        float: average nearest-neighbor distance.
    """
    tree = KDTree(points)
    # Query for the two closest points: the point itself and its nearest neighbor
    distances, _ = tree.query(points, k=2)
    # distances[:, 0] is zero (distance to itself), so we use distances[:, 1]
    return np.mean(distances[:, 1])


@njit
def gaussian_interpolator_2d(x_eval, y_eval, velocity_points, velocity_vectors, c):
    """
    Gaussian-kernel interpolator for 2D velocity field.
    
    Parameters:
        x_eval, y_eval (float): target coordinates.
        velocity_points (ndarray): shape (N, 2), known velocity sample locations.
        velocity_vectors (ndarray): shape (N, 2), known velocity vectors at those locations.
        c (float): standard deviation for the Gaussian kernel.

    Returns:
        (vx, vy): interpolated velocity vector at (x_eval, y_eval)
    """
    vx = 0.0
    vy = 0.0
    w_sum = 0.0

    for i in range(velocity_points.shape[0]):
        dx = x_eval - velocity_points[i, 0]
        dy = y_eval - velocity_points[i, 1]
        weight = np.exp(-(dx * dx + dy * dy) / (2 * c * c))

        vx += weight * velocity_vectors[i, 0]
        vy += weight * velocity_vectors[i, 1]
        w_sum += weight

    if w_sum > 0:
        vx /= w_sum
        vy /= w_sum

    return vx, vy



def RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):
    """
    Runge-Kutta 4th order particle advection in 2D using Gaussian-kernel velocity interpolation.

    Parameters:
        velocity_points: (M, 2) ndarray of known velocity sample locations.
        velocity_vectors: (M, 2) if time-independent, else (M, 2, T) with T time steps.
        trajectories: (N, 2, T_fine) ndarray of particle positions over time (to be filled).
        dt: float, time step.
        fine_time: array of fractional time steps.
        time_independent: bool, whether velocity is static in time.

    Returns:
        trajectories: updated particle trajectories (in-place).
    """

    # Compute Gaussian kernel width c
    c = average_nearest_neighbor_distance(velocity_points)

    def interpolate(floor_data, ceiling_data, t_fraction):
        return t_fraction * ceiling_data + (1 - t_fraction) * floor_data

    for t_index, t in enumerate(fine_time):
        print("time: ", t)
        x_curr = trajectories[:, 0, t_index]
        y_curr = trajectories[:, 1, t_index]

        if time_independent:
            u_interp = velocity_vectors[:, 0]
            v_interp = velocity_vectors[:, 1]
        else:
            t_floor = int(np.floor(t))
            t_ceil = int(np.ceil(t))
            t_frac = t - t_floor

            u_interp = interpolate(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceil], t_frac)
            v_interp = interpolate(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceil], t_frac)

        def eval_interpolated_velocities(xs, ys):
            vx_out = np.zeros(xs.shape)
            vy_out = np.zeros(ys.shape)
            for i in range(xs.shape[0]):
                vx_out[i], vy_out[i] = gaussian_interpolator_2d(
                    xs[i], ys[i], velocity_points, np.stack([u_interp, v_interp], axis=1), c
                )
            return vx_out, vy_out

        # RK4 steps
        k1_x, k1_y = eval_interpolated_velocities(x_curr, y_curr)
        k2_x, k2_y = eval_interpolated_velocities(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y)
        k3_x, k3_y = eval_interpolated_velocities(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y)
        k4_x, k4_y = eval_interpolated_velocities(x_curr + dt * k3_x, y_curr + dt * k3_y)

        x_next = x_curr + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_next = y_curr + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

        trajectories[:, 0, t_index + 1] = x_next
        trajectories[:, 1, t_index + 1] = y_next

    return trajectories


