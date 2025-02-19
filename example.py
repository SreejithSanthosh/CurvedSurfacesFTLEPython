import numpy as np
import math
from numba import njit
from scipy.io import loadmat
import pyvista as pv
import time
from Mesh_FTLE import FTLE_compute
import time



def NEWload_mesh_data(file_path):


    mesh_data = loadmat(file_path)
    TrianT = mesh_data['TrianT'] # node connections(dtype float)
    velocity = mesh_data['v']    # velocity for each node(dtype float)
    x_positions = mesh_data['x'] # x position for each node(dtype float)
    y_positions = mesh_data['y'] # y position for each node(dtype float)
    z_positions = mesh_data['z'] # z position for each node(dtype float)


    total_time = len(TrianT)
    length_tri = len(TrianT[0,0])    # Length of tri is different because nodes appear multiple times in the list
    node_number = len(x_positions[0,0]) # Defined as the number of vertices for the mesh

    TrianT += -1 # Shift the nodes to match normal indexing(matlab mistakes)

    time_steps = np.arange(0, total_time, 1)
    time_length = len(time_steps)

    # temp Vertex connection array
    TrianT_temp = np.zeros((length_tri, 3))

    # temp Position arrays
    x_pos = np.zeros((node_number, total_time))
    y_pos = np.zeros((node_number, total_time))
    z_pos = np.zeros((node_number, total_time))

    # temp Velocity arrays
    vx = np.zeros((node_number, total_time))
    vy = np.zeros((node_number, total_time))
    vz = np.zeros((node_number, total_time))

    TrianT_temp[:, :] = TrianT[0, 0]
    
    for t in range(total_time):   
        x_pos[:, t] = x_positions[t, 0].flatten()
        y_pos[:, t] = y_positions[t, 0].flatten()
        z_pos[:, t] = z_positions[t, 0].flatten()
        vx[:, t] = velocity[0, t].flatten()
        vy[:, t] = velocity[1, t].flatten()
        vz[:, t] = velocity[2, t].flatten()



    # Barycenter and normal calculations here
    centroid = np.zeros((length_tri, 3, time_length))
    for t in range(time_length):
        for i in range(length_tri):
            
            pos1 = np.array([x_pos[int(TrianT_temp[i, 0]), t], 
                            y_pos[int(TrianT_temp[i, 0]), t], 
                            z_pos[int(TrianT_temp[i, 0]), t]])

            pos2 = np.array([x_pos[int(TrianT_temp[i, 1]), t], 
                            y_pos[int(TrianT_temp[i, 1]), t], 
                            z_pos[int(TrianT_temp[i, 1]), t]])

            pos3 = np.array([x_pos[int(TrianT_temp[i, 2]), t], 
                            y_pos[int(TrianT_temp[i, 2]), t], 
                            z_pos[int(TrianT_temp[i, 2]), t]])
            centroid[i, :, t] = NEWcompute_vectors_barycenter(pos1, pos2, pos3)


    # Create structured array for velocity
    velocity_array = np.zeros((node_number, 3, time_length)) 
    velocity_array[:, 0, :] = vx
    velocity_array[:, 1, :] = vy
    velocity_array[:, 2, :] = vz


    # Create structured array for position
    position_array = np.zeros((node_number, 3, time_length))
    position_array[:, 0, :] = x_pos
    position_array[:, 1, :] = y_pos
    position_array[:, 2, :] = z_pos

    
    # Define the structured array for all data
    all_data_dtype = np.dtype([('node_cons', np.int32, TrianT_temp.shape),
                               ('centroids', np.float64, centroid.shape), 
                               ('position', position_array.dtype, position_array.shape),
                               ('velocity', velocity_array.dtype, velocity_array.shape),
                               ('node_number', np.int32),
                               ('time_steps', np.float64, time_steps.shape),
                               ('total_time', np.int32)])

    all_data_array = np.zeros(1, dtype=all_data_dtype)
    all_data_array['node_cons'][0] = TrianT_temp
    all_data_array['centroids'][0] = centroid
    all_data_array['position'][0] = position_array
    all_data_array['velocity'][0] = velocity_array
    all_data_array['node_number'][0] = node_number
    all_data_array['time_steps'][0] = time_steps
    all_data_array['total_time'][0] = total_time

    return all_data_array


@njit
def NEWcompute_vectors_barycenter(A,B,C):

    
    # Calculate the barycenter
    barycenter = (A + B + C) / 3.0

    tan1 = B - A
    tan2 = C - A

    normal = np.cross(tan1, tan2) 
    normal = normal / np.linalg.norm(normal)
    return barycenter



def generate_points(sphere_radius, num_points):

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Generate points using the Fibonacci lattice
    i = np.arange(0, num_points, 1)
    theta = 2 * np.pi * i / phi  # Azimuthal angle
    z = 1 - (2 * i + 1) / num_points  # Z-coordinates from -1 to 1
    radius = np.sqrt(1 - z**2)  # Radius at each z

    # Calculate (x,y,z) and scale by the desired sphere radius
    x = sphere_radius * radius * np.cos(theta)
    y = sphere_radius * radius * np.sin(theta)
    z = sphere_radius * z

    # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, s=1)

    # # Set the aspect ratio to be equal
    # ax.set_box_aspect([1, 1, 1])

    # # Labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()

    return np.array([x, y, z]).T







def run_simulation(node_connections, centroids, particle_positions, node_positions, node_velocities, time_steps, dt, direction, scheme_type):

    if dt > 1 or dt <= 0:
        raise ValueError("Error: dt must be in the interval (0,1]")

    direction = direction.lower()
    scheme_type = scheme_type.lower()

    x_traj, y_traj, z_traj = None, None, None  # Initialize as None

    if direction == "forward":
        if scheme_type == "euler":
            x_traj, y_traj, z_traj = NEWforward_particle_advection(
                node_connections, centroids, particle_positions,
                node_positions, node_velocities, time_steps, dt
            )
        elif scheme_type == "fasteuler":
            x_traj, y_traj, z_traj = NEW_fast_forward_particle_advection(
                node_connections, centroids, particle_positions,
                node_positions, node_velocities, time_steps, dt
            )
        elif scheme_type == "rk4":
            x_traj, y_traj, z_traj = NEW_RK4_forward_particle_advection(
                node_connections, centroids, particle_positions,
                node_positions, node_velocities, time_steps, dt
            )
        else:
            raise ValueError("Error: specify a scheme for the particle advection: 'euler', 'fasteuler', or 'rk4'.")

    elif direction == "backward":
        if scheme_type == "euler":
            x_traj, y_traj, z_traj = NEWbackward_particle_advection(
                node_connections, centroids, particle_positions,
                node_positions, node_velocities, time_steps, dt
            )
        elif scheme_type == "fasteuler":
            x_traj, y_traj, z_traj = NEW_fast_backward_particle_advection(
                node_connections, centroids, particle_positions,
                node_positions, node_velocities, time_steps, dt
            )
        elif scheme_type == "rk4":
            x_traj, y_traj, z_traj = NEW_RK4_backward_particle_advection(
                node_connections, centroids, particle_positions,
                node_positions, node_velocities, time_steps, dt
            )
        else:
            raise ValueError("Error: specify a scheme for the particle advection: 'euler', 'fasteuler', or 'rk4'.")

    else:
        raise ValueError("Error: Specify a direction for the particle advection: 'forward' or 'backward'.")

    # Check if trajectory values are None
    if x_traj is None or y_traj is None or z_traj is None:
        raise RuntimeError("Error: Particle trajectory computation returned None. Check the computation functions.")

    # Compute FTLE
    Ftle = FTLE_compute(
        NEWmesh_data['position'][0][:,:,0],
        NEWmesh_data['centroids'][0][:,:,0],
        particle_positions,
        np.vstack([x_traj[:,-1], y_traj[:,-1], z_traj[:,-1]]).T,
        NEWmesh_data['node_cons'][0],
        NEWmesh_data['total_time'][0]
    )

    if Ftle is None:
        raise RuntimeError("Error: FTLE computation returned None.")

    return Ftle


file_path = f'C:/Users/bafen/OneDrive/Documents/Research/Dynmaic Morphoskeleton/Mesh_LCS/mesh_data.mat'

from Mesh_advect import NEWforward_particle_advection, NEW_fast_forward_particle_advection, NEW_RK4_forward_particle_advection
from Mesh_advect import NEW_fast_backward_particle_advection, NEWbackward_particle_advection, NEW_RK4_backward_particle_advection



dt = 0.2
particle_quantity = 100000

print("time step: " , dt)
print("particle quantity: ", particle_quantity)


NEWmesh_data = NEWload_mesh_data(file_path)
NEWcentroids =  NEWmesh_data['centroids'][0]
NEWnode_positions = NEWmesh_data['position'][0]
NEWnode_velocities = NEWmesh_data['velocity'][0]
NEWparticle_positions = NEWcentroids[:,:,0]





def plot_particle_trajectories(x_traj, y_traj, z_traj, time_length, pause_time=0.05):
    """
    Plots the particles from the output trajectory data over time.

    Parameters:
    x_traj, y_traj, z_traj: Arrays of particle positions over time.
    time_length: Number of time steps in the simulation.
    pause_time: Time delay (in seconds) between frames.
    """
    num_particles = x_traj.shape[0]  # Number of particles

    # Initialize PyVista plotter in interactive mode
    pl = pv.Plotter()
    pl.add_axes()
    pl.show_grid()

    # Initial scatter plot
    points = np.c_[x_traj[:, 0], y_traj[:, 0], z_traj[:, 0]]
    scatter_plot = pl.add_points(points, color="red", point_size=5.0)

    # Start interactive mode (without blocking execution)
    pl.show(interactive_update=True)  

    # Animation loop over time steps
    for t in range(time_length):
        print(f"Animating time step {t+1}/{time_length}...")

        # Update the particle positions
        points = np.c_[x_traj[:, t], y_traj[:, t], z_traj[:, t]]

        # Remove previous scatter plot and re-add updated points
        pl.remove_actor(scatter_plot)  
        scatter_plot = pl.add_points(points, color="red", point_size=5.0)  
        pl.render()  # Render the updated plot immediately

        # Introduce a small delay
        time.sleep(pause_time)  # Pause execution for smooth animation

    # Keep the window open at the last frame
    pl.show()



particle_positions = generate_points(1, particle_quantity)

from time import time as timer  # Use an alias to avoid conflicts



start_time = timer()
Fx_traj, Fy_traj, Fz_traj = NEW_fast_forward_particle_advection(NEWmesh_data['node_cons'][0],
                                            NEWcentroids, 
                                            particle_positions,
                                            NEWnode_positions,
                                            NEWnode_velocities, 
                                            NEWmesh_data['time_steps'][0],
                                            dt)


fast_for_time = timer() - start_time
print("fast euler: ", fast_for_time)


# Run the function with trajectory output
#plot_particle_trajectories(Fx_traj, Fy_traj, Fz_traj, mesh_data['total_time'][0])


particle_positions = generate_points(1, particle_quantity)


# ***************
start_time = timer()
x_traj, y_traj, z_traj  = NEWforward_particle_advection(NEWmesh_data['node_cons'][0],
                                            NEWcentroids, 
                                            particle_positions,
                                            NEWnode_positions,
                                            NEWnode_velocities, 
                                            NEWmesh_data['time_steps'][0],
                                            dt)

for_time = timer() - start_time
print("euler: ", for_time)


# Run the function with trajectory output
#plot_particle_trajectories(x_traj, y_traj, z_traj, NEWmesh_data['total_time'][0])



# ***************

start_time = timer()
Ax_traj, Ay_traj, Az_traj = NEW_RK4_forward_particle_advection(NEWmesh_data['node_cons'][0],
                                            NEWcentroids, 
                                            particle_positions,
                                            NEWnode_positions,
                                            NEWnode_velocities, 
                                            NEWmesh_data['time_steps'][0],
                                            dt)
Afor_time = timer() - start_time
print("RK4: " , Afor_time)



fftle = run_simulation(NEWmesh_data['node_cons'][0], NEWcentroids, particle_positions, NEWnode_positions, NEWnode_velocities, NEWmesh_data['time_steps'][0], dt, "forward", "fasteuler")

fast_fftle = FTLE_compute(
                    NEWmesh_data['position'][0][:,:,0],
                    NEWmesh_data['centroids'][0][:,:,0],
                    particle_positions,
                    np.vstack([Fx_traj[:,-1],Fy_traj[:,-1],Fz_traj[:,-1]]).T,
                    NEWmesh_data['node_cons'][0],
                    NEWmesh_data['total_time'][0])

aafftle = FTLE_compute(
                    NEWmesh_data['position'][0][:,:,0],
                    NEWmesh_data['centroids'][0][:,:,0],
                    particle_positions,
                    np.vstack([x_traj[:,-1],y_traj[:,-1],z_traj[:,-1]]).T,
                    NEWmesh_data['node_cons'][0],
                    NEWmesh_data['total_time'][0])

Rfftle = FTLE_compute(
                    NEWmesh_data['position'][0][:,:,0],
                    NEWmesh_data['centroids'][0][:,:,0],
                    particle_positions,
                    np.vstack([Ax_traj[:,-1],Ay_traj[:,-1],Az_traj[:,-1]]).T,
                    NEWmesh_data['node_cons'][0],
                    NEWmesh_data['total_time'][0])






point_cloud = np.vstack((particle_positions[:, 0], 
                        particle_positions[:, 1], 
                        particle_positions[:, 2])).T


perturbation = 1e-6 * (np.random.rand(*particle_positions.shape) - 0.5)
points_perturbed = particle_positions + perturbation
points = pv.PolyData(points_perturbed)


if False:
    # Assign each FTLE field as scalars for the point cloud
    points['BFTLE'] = bftle
    points['Abftle'] = Rbftle


    # Reconstruct each surface using Delaunay triangulation
    surf_fftle = points.delaunay_3d()
    surf_fftle['BFTLE'] = bftle

    surf_afftle = points.delaunay_3d()
    surf_afftle['Rbftle'] = Rbftle



    # Create a PyVista plotter with three subplots
    pl = pv.Plotter(shape=(1, 2), title="Comparison of Advection Schemes on FTLE Colormaps")

    # FFTLE Surface Plot
    pl.subplot(0, 0)
    pl.add_mesh(surf_fftle, scalars='BFTLE', cmap='jet', show_edges=False)
    pl.add_title('fast BFTLE')

    # Afftle Surface Plot
    pl.subplot(0, 1)
    pl.add_mesh(surf_afftle, scalars='Rbftle', cmap='jet', show_edges=False)
    pl.add_title('RK4 Bftle')



    # Show and save the plot
    pl.save_graphic("FTLE_comparison_plot.svg")
    pl.show()
    pl.close()






# Assign each FTLE field as scalars for the point cloud
points['fastFTLE'] = fast_fftle
points['FFTLE'] = fftle
points['Rfftle'] = Rfftle



# Reconstruct each surface using Delaunay triangulation
surf_fastftle = points.delaunay_3d()
surf_fastftle['fastFTLE'] = fast_fftle

surf_fftle = points.delaunay_3d()
surf_fftle['FFTLE'] = fftle

surf_afftle = points.delaunay_3d()
surf_afftle['Rfftle'] = Rfftle



window_size = (1920, 1080)

# Create a PyVista plotter with off-screen rendering
pl = pv.Plotter(shape=(1, 3), off_screen=False, window_size=window_size)  # Enable off-screen rendering

# === SUBPLOTS ===
pl.subplot(0, 0)
pl.add_mesh(surf_fastftle, scalars='fastFTLE', cmap='jet', show_edges=False)
pl.add_title('Fast Euler Scheme')

pl.subplot(0, 1)
pl.add_mesh(surf_fftle, scalars='FFTLE', cmap='jet', show_edges=False)
pl.add_title('Euler Scheme')

pl.add_text(f"(dt={dt})", 
            position="upper_edge", font_size=20, color="black", font="arial")

pl.subplot(0, 2)
pl.add_mesh(surf_afftle, scalars='Rfftle', cmap='jet', show_edges=False)
pl.add_title('RK4 Scheme')

pl.add_text(f"(parts={particle_quantity})", 
            position="upper_edge", font_size=20, color="black", font="arial")

# Save the screenshot
screenshot_filename = f"scheme_com_dt{dt}_parts{particle_quantity}.png"
#pl.screenshot(screenshot_filename)

# Show the plot (if needed)
pl.show()



