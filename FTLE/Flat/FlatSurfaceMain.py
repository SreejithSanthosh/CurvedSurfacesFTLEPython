import numpy as np
from FTLE.Flat.FTLECompute import FTLE_2d_compute, FTLE_3d_compute
from FTLE.Flat.utilities import plot_FTLE_2d, plot_FTLE_3d, interpolate, plot_FTLE_2d_scatter, subdivide_time_steps
from FTLE.Flat.advection import RK4_advection_2d, RK4_advection_3d




def run_FTLE_2d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    dt,
    initial_time,
    final_time,
    time_steps,
    time_independent=False,
    plot_ftle=False,
    save_plot_path=None
):

    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Initial and final times must be in `time_steps`.")

    if initial_time >= final_time:
        raise ValueError("initial_time must be less than or equal to final_time")

    if dt > 1 or dt <= 0:
        raise ValueError("Error: dt must be in the interval (0,1]")
        
    initial_time_index = index = np.where(time_steps == initial_time)[0][0]
    final_time_index = index = np.where(time_steps == final_time)[0][0]

    print(time_steps[final_time_index])
    ## Forward
    ftle, trajectories, isotropy = FTLE_2d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    dt,
    initial_time_index,
    final_time_index,
    time_steps,
    "Forward",
    time_independent
)

    ## Backward
    back_ftle, back_trajectories, back_isotropy = FTLE_2d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    dt,
    final_time_index,
    initial_time_index,
    time_steps,
    "Backward",
    time_independent
)

    if plot_ftle:
        particles = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten()]).T
        plot_FTLE_2d(x_grid_parts, y_grid_parts, ftle, isotropy, back_ftle, back_isotropy, initial_time, final_time, save_plot_path=save_plot_path)
        

    return ftle, trajectories, isotropy, back_ftle, back_trajectories, back_isotropy

def run_FTLE_3d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    z_grid_parts,
    dt,
    initial_time_index,
    final_time_index,
    time_steps,
    time_independent=False,
    plot_ftle=False,
    save_plot_path=None
):

    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Initial and final times must be in `time_steps`.")

    if initial_time >= final_time:
        raise ValueError("initial_time must be less than or equal to final_time")

    if dt > 1 or dt <= 0:
        raise ValueError("Error: dt must be in the interval (0,1]")
        
    initial_time_index = index = np.where(time_steps == initial_time)[0][0]
    final_time_index = index = np.where(time_steps == final_time)[0][0]


    ## Forward
    ftle, trajectories, isotropy = FTLE_3d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    z_grid_parts,
    dt,
    initial_time_index,
    final_time_index,
    time_steps,
    "Forward",
    time_independent
    )

    print("finished forward")
    ## Backward
    back_ftle, back_trajectories, back_isotropy = FTLE_3d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    z_grid_parts,
    dt,
    final_time_index,
    initial_time_index,
    time_steps,
    "Backward",
    time_independent
    )
    print("finished backward")

    if plot_ftle:
        particles_positions = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten(), z_grid_parts.flatten()]).T
        plot_FTLE_3d(particles_positions, ftle, isotropy, back_ftle, back_isotropy, initial_time, final_time, save_plot_path=save_plot_path)
        
    return ftle, trajectories, isotropy, back_ftle, back_trajectories, back_isotropy






def FTLE_2d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    dt,
    initial_time_index,
    final_time_index,
    time_steps,
    direction,
    time_independent=False
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


    direction = direction.lower()
    if direction == "backward":

        # Reverse time indexing
        temp_initial_time_index = len(time_steps) - final_time_index - 1
        final_time_index = len(time_steps) - initial_time_index - 1
        initial_time_index = final_time_index
        final_time_index = temp_initial_time_index

        if not time_independent:
            velocity_vectors = velocity_vectors[:, :, ::-1]
        dt = -dt

    # --- Setup grid and particle data ---
    x_dim1, x_dim2 = x_grid_parts.shape
    y_dim1, y_dim2 = y_grid_parts.shape

    particles_positions = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten()]).T
    num_particles = particles_positions.shape[0]

    fine_time = np.arange(time_steps[initial_time_index], time_steps[final_time_index] + np.abs(dt), np.abs(dt))
    fine_time = subdivide_time_steps(time_steps[initial_time_index:final_time_index + 1], np.abs(dt))
    print("fine_time right after subdivide",fine_time)
    fine_time_length = len(fine_time)

    trajectories = np.zeros((num_particles, 2, fine_time_length))
    trajectories[:, :, 0] = particles_positions

    fine_time = fine_time[:-1]  # Exclude final point (we advect up to t_index + 1)

    trajectories = RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent)

    # --- Compute FTLE from reshaped results ---
    x_traj = trajectories[:, 0, :].reshape(x_dim1, x_dim2, fine_time_length)
    y_traj = trajectories[:, 1, :].reshape(y_dim1, y_dim2, fine_time_length)

    ftle, isotropy = FTLE_2d_compute(
        x_grid_parts, y_grid_parts,
        x_traj[:, :, -1], y_traj[:, :, -1],
        time_steps[initial_time_index],
        time_steps[final_time_index]
    )


    return ftle, trajectories, isotropy



def FTLE_3d(
    velocity_points,
    velocity_vectors,
    x_grid_parts,
    y_grid_parts,
    z_grid_parts,
    dt,
    initial_time_index,
    final_time_index,
    time_steps,
    direction,
    time_independent=False
):
    """
    Advects a uniform 3D grid of particles using a sparse or dense velocity field with RK4 integration.

    Parameters:
        velocity_points (ndarray): (M, 3) array of known velocity locations, fixed in time.
        velocity_vectors (ndarray): (M, 3) if time-independent, or (M, 3, T) if time-dependent.
        x_grid_parts, y_grid_parts, z_grid_parts (ndarray): 3D meshgrid arrays of particle positions.
        dt (float): Time step size for integration (0 < dt <= 1).
        initial_time (int): Start index for advection.
        final_time (int): End index for advection.
        time_steps (ndarray): Array of time step indices.
        direction (str): "forward" or "backward".
        time_indepedent (bool): Whether velocity is independent of time(i.e the data has a time axis).

    Returns:
        ftle (ndarray): Flattened FTLE values.
        trajectories (ndarray): (N, 3, T) particle positions over time.
    """

    if direction == "backward":

        # Time reversal
        temp_initial_time_index = len(time_steps) - final_time_index - 1
        final_time_index = len(time_steps) - initial_time_index - 1
        initial_time_index = final_time_index
        final_time_index = temp_initial_time_index

        if not time_independent:
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

    fine_time = np.arange(time_steps[initial_time_index], time_steps[final_time_index], np.abs(dt))
    fine_time = subdivide_time_steps(time_steps[initial_time_index:final_time_index + 1], np.abs(dt))
    fine_time_length = len(fine_time)

    trajectories = np.zeros((num_particles, 3, fine_time_length))
    trajectories[:, :, 0] = particle_positions

    fine_time = fine_time[:-1]  # Integrate up to last point


    ## particle Advection
    trajectories = RK4_advection_3d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent)


    # --- Reshape trajectories and compute FTLE ---
    x_traj = trajectories[:, 0, :].reshape(x_dim1, x_dim2, x_dim3, fine_time_length)
    y_traj = trajectories[:, 1, :].reshape(y_dim1, y_dim2, y_dim3, fine_time_length)
    z_traj = trajectories[:, 2, :].reshape(z_dim1, z_dim2, z_dim3, fine_time_length)

    ftle, isotropy = FTLE_3d_compute(
        x_grid_parts, y_grid_parts, z_grid_parts,
        x_traj[:, :, :, -1], y_traj[:, :, :, -1], z_traj[:, :, :, -1],
        time_steps[initial_time_index],
        time_steps[final_time_index]
    )

        
    return ftle, trajectories, isotropy
