from scipy.interpolate import LinearNDInterpolator
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
from scipy.spatial import cKDTree

@njit
def interpolate_linear_2d(particle_x, particle_y, velocity_points, velocity_values, k=8):
    """
    Simple manual linear interpolation using inverse distance weighting.
    Compatible with numba (no scipy or fancy structures).
    """
    result = np.empty_like(particle_x)
    for i in range(len(particle_x)):
        weights = np.zeros(k)
        values = np.zeros(k)
        distances = np.zeros(k) + 1e12  # large initial distance

        px, py = particle_x[i], particle_y[i]

        for j in range(velocity_points.shape[0]):
            if j < k:
                dx = px - velocity_points[j, 0]
                dy = py - velocity_points[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                distances[j] = dist
                values[j] = velocity_values[j]
            else:
                # not dynamic: pick first k
                break

        inv_dist = 1.0 / distances
        weights = inv_dist / np.sum(inv_dist)
        result[i] = np.sum(weights * values)
    return result

# Note: this function cannot directly replace your RK4 code yet,
# but gives a numba-compatible inverse-distance weighted interpolation routine.
# Full RK4 + neighbor-search version would need cKDTree or rewritten KDTree logic.

# Let me know if you'd like the full RK4 loop refactored around this interpolation scheme.

@njit
def interpolate_time_dependent(floor_data, ceiling_data, t_fraction):
    return t_fraction * ceiling_data + (1 - t_fraction) * floor_data

@njit
def RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):
    num_particles, _, num_steps = trajectories.shape
    k_neighbors = 8  # using first k points only

    for t_index in range(len(fine_time)):
        x_curr = trajectories[:, 0, t_index]
        y_curr = trajectories[:, 1, t_index]

        if time_independent:
            u_vals = velocity_vectors[:, 0]
            v_vals = velocity_vectors[:, 1]
        else:
            t = fine_time[t_index]
            t_floor = int(np.floor(t))
            t_ceil = int(np.ceil(t))
            t_frac = t - t_floor

            u_vals = interpolate_time_dependent(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceil], t_frac)
            v_vals = interpolate_time_dependent(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceil], t_frac)

        def step(x_in, y_in):
            k1_x = interpolate_linear_2d(x_in, y_in, velocity_points, u_vals, k=k_neighbors)
            k1_y = interpolate_linear_2d(x_in, y_in, velocity_points, v_vals, k=k_neighbors)

            k2_x = interpolate_linear_2d(x_in + 0.5 * dt * k1_x, y_in + 0.5 * dt * k1_y, velocity_points, u_vals, k=k_neighbors)
            k2_y = interpolate_linear_2d(x_in + 0.5 * dt * k1_x, y_in + 0.5 * dt * k1_y, velocity_points, v_vals, k=k_neighbors)

            k3_x = interpolate_linear_2d(x_in + 0.5 * dt * k2_x, y_in + 0.5 * dt * k2_y, velocity_points, u_vals, k=k_neighbors)
            k3_y = interpolate_linear_2d(x_in + 0.5 * dt * k2_x, y_in + 0.5 * dt * k2_y, velocity_points, v_vals, k=k_neighbors)

            k4_x = interpolate_linear_2d(x_in + dt * k3_x, y_in + dt * k3_y, velocity_points, u_vals, k=k_neighbors)
            k4_y = interpolate_linear_2d(x_in + dt * k3_x, y_in + dt * k3_y, velocity_points, v_vals, k=k_neighbors)

            x_out = x_in + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            y_out = y_in + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

            return x_out, y_out

        x_next, y_next = step(x_curr, y_curr)
        trajectories[:, 0, t_index + 1] = x_next
        trajectories[:, 1, t_index + 1] = y_next

    return trajectories
