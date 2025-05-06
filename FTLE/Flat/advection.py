from scipy.interpolate import LinearNDInterpolator
import numpy as np



def RK4_advection_3d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):
    
    def interpolate(floor_data, ceiling_data, t_fraction):
        return t_fraction*ceiling_data + (1-t_fraction)*floor_data
        
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

    return trajectories

def RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):

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
