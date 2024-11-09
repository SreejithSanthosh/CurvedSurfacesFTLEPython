from matplotlib.image import NEAREST
import numpy as np
import math
import os
import shutil
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import pyvista as pv
from itertools import combinations

from Mesh_advect import forward_particle_advection, backward_particle_advection
from Mesh_FTLE import FTLE_compute





def load_mesh_data(file_path, dt):


    mesh_data = loadmat(file_path)
    TrianT = mesh_data['TrianT'] # node connections(dtype float)
    velocity = mesh_data['v']    # velocity for each node(dtype float)
    x_positions = mesh_data['x'] # x position for each node(dtype float)
    y_positions = mesh_data['y'] # y position for each node(dtype float)
    z_positions = mesh_data['z'] # z position for each node(dtype float)


    total_time = len(TrianT)
    length_tri = len(TrianT[0,0])    # Length of tri is different because nodes appear multiple times in the list
    node_number = len(x_positions[0,0]) # Defined as the number of vertices for the mesh

    TrianT += -1 # Shift the nodes to match python indexing(matlab code mistakes)

    time_steps = np.arange(0, total_time, dt)
    time_length = len(time_steps)

    # temp Vertex connection array
    TrianT_temp = np.zeros((length_tri, 3, total_time))

    # temp Position arrays
    x_pos = np.zeros((node_number, total_time))
    y_pos = np.zeros((node_number, total_time))
    z_pos = np.zeros((node_number, total_time))

    # temp Velocity arrays
    vx = np.zeros((node_number, total_time))
    vy = np.zeros((node_number, total_time))
    vz = np.zeros((node_number, total_time))


    for t in range(total_time):
        TrianT_temp[:, :, t] = TrianT[t, 0]
        x_pos[:, t] = x_positions[t, 0].flatten()
        y_pos[:, t] = y_positions[t, 0].flatten()
        z_pos[:, t] = z_positions[t, 0].flatten()
        vx[:, t] = velocity[0, t].flatten()
        vy[:, t] = velocity[1, t].flatten()
        vz[:, t] = velocity[2, t].flatten()

    #Interpolate node_positions((dummy))
    x_pos_mod = interpo(x_pos, time_steps, total_time)
    y_pos_mod = interpo(y_pos, time_steps, total_time)
    z_pos_mod = interpo(z_pos, time_steps, total_time)

    #Interpolate node_structure(dummy)
    TrianT_mod = np.zeros((length_tri, 3, time_length))
    for t in range(time_length):
        TrianT_mod[:, :, t] = TrianT_temp[:,:, 0]

    #Interpolate node_velocities(dummy)
    vx_mod = interpo(vx, time_steps, total_time)
    vy_mod = interpo(vy, time_steps, total_time)
    vz_mod = interpo(vz, time_steps, total_time)


    # Barycenter and normal calculations here
    centroid = np.zeros((length_tri, 3, time_length))
    normals = np.zeros((length_tri, 3, time_length))
    for t in range(time_length):
        for i in range(length_tri):
            
            pos1 = np.array([x_pos_mod[int(TrianT_mod[i, 0, t]), t], 
                            y_pos_mod[int(TrianT_mod[i, 0, t]), t], 
                            z_pos_mod[int(TrianT_mod[i, 0, t]), t]])

            pos2 = np.array([x_pos_mod[int(TrianT_mod[i, 1, t]), t], 
                            y_pos_mod[int(TrianT_mod[i, 1, t]), t], 
                            z_pos_mod[int(TrianT_mod[i, 1, t]), t]])

            pos3 = np.array([x_pos_mod[int(TrianT_mod[i, 2, t]), t], 
                            y_pos_mod[int(TrianT_mod[i, 2, t]), t], 
                            z_pos_mod[int(TrianT_mod[i, 2, t]), t]])
            normals[i, :, t], tan1, tan2, centroid[i, :, t] = compute_vectors_barycenter(pos1, pos2, pos3)


    # Create structured array for velocity
    velocity_array = np.zeros((node_number, 3, time_length)) 
    velocity_array[:, 0, :] = vx_mod
    velocity_array[:, 1, :] = vy_mod
    velocity_array[:, 2, :] = vz_mod


    # Create structured array for position
    position_array = np.zeros((node_number, 3, time_length))
    position_array[:, 0, :] = x_pos_mod
    position_array[:, 1, :] = y_pos_mod
    position_array[:, 2, :] = z_pos_mod

    
    # Define the structured array for all data
    all_data_dtype = np.dtype([('node_cons', np.int32, TrianT_mod.shape),
                               ('centroids', np.float64, centroid.shape), 
                               ('normals', np.float64, normals.shape),
                               ('position', position_array.dtype, position_array.shape),
                               ('velocity', velocity_array.dtype, velocity_array.shape),
                               ('node_number', np.int32),
                               ('time_steps', np.float64, time_steps.shape),
                               ('total_time', np.int32)])

    all_data_array = np.zeros(1, dtype=all_data_dtype)
    all_data_array['node_cons'][0] = TrianT_mod
    all_data_array['centroids'][0] = centroid
    all_data_array['normals'][0] = normals
    all_data_array['position'][0] = position_array
    all_data_array['velocity'][0] = velocity_array
    all_data_array['node_number'][0] = node_number
    all_data_array['time_steps'][0] = time_steps
    all_data_array['total_time'][0] = total_time

    return all_data_array


# butter
@njit
def interp_mesh(TrianT, time_steps):

    new_mesh = np.zeros((len(TrianT[:,0,0]), len(TrianT[0,:,0]), len(time_steps)))

    for t in range(len(time_steps)):
        new_mesh[:, :, t] = TrianT[:, :, 0]

    return 0


# important
@njit
def compute_vectors_barycenter(A,B,C):

    
    # Calculate the barycenter
    barycenter = (A + B + C) / 3.0

    tan1 = B - A
    tan2 = C - A

    normal = np.cross(tan1, tan2) 
    normal = normal / np.linalg.norm(normal)
    return normal, tan1, tan2, barycenter


# important
@njit
def interpo(data, time_steps, total_time):
    if(dt > 1):
        print("Error: dt greater than 1")
        return 0
    
    time_length = len(time_steps)
    data_length = len(data[:, 0])

    new_data = np.zeros((data_length, time_length))

    for t in range(time_length):

        t_index = math.floor(time_steps[t])
        t_fraction = time_steps[t] - t_index

        if t_index < total_time - 1:
            # Straight line homotopy
            new_data[:, t] = (data[:, t_index + 1] * t_fraction) + ((1-t_fraction) * data[:, t_index])
        else:

            new_data[:, t] = data[:, t_index]
        
    return new_data



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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return np.array([x, y, z]).T




file_path = f'C:/Users/bafen/OneDrive/Documents/Research/Dynmaic Morphoskeleton/Mesh_LCS/mesh_data.mat'


dt = 0.2
mesh_data = load_mesh_data(file_path, dt)
centroids =  mesh_data['centroids'][0]
node_positions = mesh_data['position'][0]
node_velocities = mesh_data['velocity'][0]
particle_positions = centroids[:,:,0]




x_traj, y_traj, z_traj = forward_particle_advection(mesh_data['node_cons'][0],
                                            centroids, 
                                            particle_positions,
                                            node_positions,
                                            node_velocities, 
                                            mesh_data['time_steps'][0],
                                            dt)


bx_traj, by_traj, bz_traj = backward_particle_advection(mesh_data['node_cons'][0],
                                            centroids, 
                                            particle_positions,
                                            node_positions,
                                            node_velocities, 
                                            mesh_data['time_steps'][0],
                                            dt)





bftle = FTLE_compute(
                    mesh_data['position'][0][:,:,0],
                    mesh_data['centroids'][0][:,:,0],
                    particle_positions,
                    np.vstack([bx_traj[:,-1],by_traj[:,-1],bz_traj[:,-1]]).T,
                    mesh_data['node_cons'][0][:,:,0],
                    mesh_data['total_time'][0])

fftle = FTLE_compute(
                    mesh_data['position'][0][:,:,0],
                    mesh_data['centroids'][0][:,:,0],
                    particle_positions,
                    np.vstack([x_traj[:,-1],y_traj[:,-1],z_traj[:,-1]]).T,
                    mesh_data['node_cons'][0][:,:,0],
                    mesh_data['total_time'][0])




point_cloud = np.vstack((particle_positions[:, 0], 
                         particle_positions[:, 1], 
                         particle_positions[:, 2])).T

# Wrap the point cloud into a PyVista object
points = pv.PolyData(point_cloud)

# Assign FTLE values as scalars for the point cloud
points['FFTLE'] = fftle


# Reconstruct the surface from the point cloud
surf = points.delaunay_3d()

# Explicitly interpolate both FFTLE and BFTLE onto the surface
surf['FFTLE'] = surf.interpolate(points, sharpness=5)['FFTLE']

# Create a PyVista plotter
pl = pv.Plotter(shape=(1, 2))

# Plot the point cloud with FFTLE
pl.subplot(0, 0)
pl.add_mesh(points, scalars='FFTLE', cmap='jet', point_size=10)
pl.add_title('Point Cloud of 2D Surface with FFTLE Colormap')

# Plot the reconstructed surface with FFTLE colormap
pl.subplot(0, 1)
pl.add_mesh(surf, scalars='FFTLE', cmap='jet', show_edges=False)
pl.add_title('Reconstructed Surface with FFTLE Colormap')
pl.save_graphic("FFTLE_plot.svg") 
pl.show()
pl.close()

pl = pv.Plotter(shape=(1, 2))
points['BFTLE'] = bftle
surf['BFTLE'] = surf.interpolate(points, sharpness=5)['BFTLE']
# Plot the point cloud with BFTLE
pl.subplot(0, 0)
pl.add_mesh(points, scalars='BFTLE', cmap='jet', point_size=10)
pl.add_title('Point Cloud of 2D Surface with BFTLE Colormap')

# Plot the reconstructed surface with BFTLE colormap
pl.subplot(0, 1)
pl.add_mesh(surf, scalars='BFTLE', cmap='jet', show_edges=False)
pl.add_title('Reconstructed Surface with BFTLE Colormap')
pl.save_graphic("BFTLE_plot.svg") 
pl.show()
pl.close()



