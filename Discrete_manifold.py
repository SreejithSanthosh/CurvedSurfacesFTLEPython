
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from scipy.interpolate import griddata
from itertools import combinations
import pyvista as pv


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


# butter
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

# butter
def gen_points_per_node(number_nodes, node_positions, number_points, radii=0.1):

    if number_points < number_nodes:
        print("Not enough points for particle generation.")
        return None

    x_points = []
    y_points = []
    z_points = []
    points_per_sphere = int(number_points / number_nodes)

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Generate points using the Fibonacci lattice
    i = np.arange(0, points_per_sphere, 1)
    theta = 2 * np.pi * i / phi  # Azimuthal angle
    z = 1 - (2 * i + 1) / points_per_sphere  # Z-coordinates from -1 to 1
    radius = np.sqrt(1 - z**2)  # Radius at each z

    # Calculate (x,y,z) for a unit sphere
    x = radii * radius * np.cos(theta)
    y = radii * radius * np.sin(theta)
    z = radii * z

    for i in range(0, number_nodes):
        # Shift sphere to be centered around the current node
        node_x = node_positions[i, 0]
        node_y = node_positions[i, 1]
        node_z = node_positions[i, 2]

        x_shifted = x + node_x
        y_shifted = y + node_y
        z_shifted = z + node_z

        # Append shifted points to the overall points list
        x_points.extend(x_shifted)
        y_points.extend(y_shifted)
        z_points.extend(z_shifted)

    # Convert lists to numpy arrays
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    z_points = np.array(z_points)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_points, y_points, z_points, s=1)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return np.vstack((x_points, y_points, z_points)).T

# butter
def plot_centroids_scatter(centroids, t=0):
    """
    Plot the centroids as a scatter plot at a specific time step.

    Parameters:
    centroids (ndarray): The array containing the centroids for all triangles.
    t (int): The time step to plot (default is 0).
    """
    # Extract the centroids for the given time step
    centroid_positions = centroids[:, :, t]
    
    # Plot the centroids as a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centroid_positions[:, 0], centroid_positions[:, 1], centroid_positions[:, 2], color='red', s=10)
    
    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()




# important
### Functions for particle projection
@njit
def project(A, B, C, position):
    """
    Project the current position to the plane formed by points A,B,C

    Parameters:
    A, B, C: Points in R^3
    position: the point to project onto the ABC plane
    """
    vec1 = B-A
    vec2 = C-A
    vec3 = position - A

    normal_vector = np.cross(vec1, vec2)

    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    projection_onto_normal = np.dot(vec3, normal_vector) * normal_vector
            
    # Subtract the projection from vector3 to get the projection onto the (local)plane
    projection_onto_plane = vec3 - projection_onto_normal
    return A + projection_onto_plane

# important
@njit
def is_point_in_triangle(P, A, B, C, error_margin= 0.1):
    """
    Check if point P lies within the triangle defined by points A, B, and C using barycentric coordinates.
    
    error_margin: small value to avoid edge cases
    """
    # Vectors from point A to B and C
    v0 = C - A
    v1 = B - A
    v2 = P - A

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    # Check if point is in triangle
    return (u >= 0 - error_margin) and (v >= 0 - error_margin) and (u + v <= 1 + error_margin)


# important
# optimize this
def particle_projection(TrianT, kdtree, particle_positions, node_positions):
    """
    Project all particles, involves finding the closest face centroid in the kdtree
        and projecting points to that plane formed by that face
    
    Parameters: 
    TrianT: The vertex connections
    kdtree: the kdtree made from the centroids of TrainT
    particle_positions: the current positions to be projected
    node_positions: the vertex positions of the mesh
    node_velocities: the velocities at each vertex
    """

    new_positions = np.zeros((len(particle_positions), 3))
    centroid_indices = np.zeros(len(particle_positions), dtype=int)

    # debug
    inside_triangle = np.zeros(len(particle_positions), dtype=bool)

    # for testing
    false_count = 0

    for i in range(len(particle_positions[:,0])):
        
        query_point = particle_positions[i, :]
        
        # Get nearest centriod data
        k_closest_centroids = 10
        distances, indices = kdtree.query(query_point, k_closest_centroids)
        
        outside_triangle = True
        face_count = 1


        # Do the initial projection
        nearest_face = TrianT[indices[0], :]

        A = node_positions[nearest_face[0], :]
        B = node_positions[nearest_face[1], :]
        C = node_positions[nearest_face[2], :]

        new_positions[i, :] = project(A,B,C, query_point)
        
        centroid_indices[i] = indices[0]

        outside_triangle = not is_point_in_triangle(new_positions[i, :], A, B, C)


        while(outside_triangle and face_count < k_closest_centroids):
            
            centroid_indices[i] = indices[face_count]
            nearest_face = TrianT[indices[face_count], :]

            A = node_positions[nearest_face[0], :]
            B = node_positions[nearest_face[1], :]
            C = node_positions[nearest_face[2], :]


            # Check if the projected point is inside the triangle
            outside_triangle = not is_point_in_triangle(new_positions[i, :], A, B, C)

            face_count += 1
            
        
        if outside_triangle == True:
            false_count += 1
    
    
    print("Percentage projected outside faces: ", false_count/ len(inside_triangle) * 100)
    return new_positions, centroid_indices


def get_velocity(particle_positions, faces, node_positions, node_velocities, face_indices):
    """
        Gets the velocity as a specific position by interpolation of the surrounding
        node velocites.

    Parameters: 
    particle_positions: position of particle
    faces: faces of the simplicial complex
    face_indices: Inidices of the faces 
    node_positions: the vertex positions of the mesh
    node_velocities: the velocities at each vertex
    """

    particle_faces = faces[face_indices, :]

    # Extract node positions and velocities for each particle's corresponding face
    A = node_positions[particle_faces[:, 0], :]
    B = node_positions[particle_faces[:, 1], :]
    C = node_positions[particle_faces[:, 2], :]

    v0 = node_velocities[particle_faces[:, 0], :]
    v1 = node_velocities[particle_faces[:, 1], :]
    v2 = node_velocities[particle_faces[:, 2], :]

    # Compute vectors relative to A
    v2_1 = B - A
    v2_2 = C - A
    v2_particle = particle_positions - A

    # Compute the dot products needed for barycentric coordinates
    d00 = np.einsum('ij,ij->i', v2_1, v2_1)
    d01 = np.einsum('ij,ij->i', v2_1, v2_2)
    d11 = np.einsum('ij,ij->i', v2_2, v2_2)
    d20 = np.einsum('ij,ij->i', v2_particle, v2_1)
    d21 = np.einsum('ij,ij->i', v2_particle, v2_2)

    # Compute the barycentric coordinates
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # Interpolate the velocities at the particle positions
    interpolated_velocities = u[:, np.newaxis] * v0 + v[:, np.newaxis] * v1 + w[:, np.newaxis] * v2

    return interpolated_velocities

# important
def forward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps):

    time_length = len(time_steps)
    number_particles = len(particle_positions[:,0])

    x_traj = np.zeros((number_particles, time_length))
    y_traj = np.zeros((number_particles, time_length))
    z_traj = np.zeros((number_particles, time_length))


    #### Perfomr advection from time_steps 0 to 1
    # Build the initial kdtree
    kdtree = KDTree(centroids[:,:, 0])


    # Make sure the initial particles are on the surface(project twice)
    new_positions, centroid_indices = particle_projection(TrianT[:,:, 0], kdtree, particle_positions, node_positions[:,:, 0]) 
    new_positions, centroid_indices = particle_projection(TrianT[:,:, 0], kdtree, new_positions, node_positions[:,:, 0])
    

    # Set up initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    # Get current velocities for t=0
    current_positions = np.array([x_current, y_current, z_current]).T
    velocities = get_velocity(current_positions, TrianT[:,:,0], node_positions[:,:, 0], node_velocities[:,:, 0], centroid_indices)

    for t in range(1 , time_length):

        x_current += velocities[:, 0] * dt
        y_current += velocities[:, 1] * dt
        z_current += velocities[:, 2] * dt

        kdtree = KDTree(centroids[:,:, t])

        current_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT[:,:, t], kdtree, current_positions, node_positions[:,:,t])

        x_traj[:, t] = new_positions[:,0]
        y_traj[:, t] = new_positions[:,1]
        z_traj[:, t] = new_positions[:,2]

        velocities = get_velocity(new_positions, TrianT[:,:, t], node_positions[:,:,t], node_velocities[:,:, t], centroid_indices)


    return x_traj, y_traj, z_traj

# important
def backward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps,):



    time_length = len(time_steps)  - 1
    number_particles = len(particle_positions[:,0])

    x_traj = np.zeros((number_particles, time_length))
    y_traj = np.zeros((number_particles, time_length))
    z_traj = np.zeros((number_particles, time_length))


    #### Perfomr advection from time_steps 0 to 1
    # Build the initial kdtree
    kdtree = KDTree(centroids[:,:, -1])


    # Make sure the initial particles are on the surface(project twice)
    new_positions, centroid_indices = particle_projection(TrianT[:,:, -1], kdtree, particle_positions, node_positions[:,:, -1]) 
    new_positions, centroid_indices = particle_projection(TrianT[:,:, -1], kdtree, new_positions, node_positions[:,:, -1])
    

    # Set up initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    # Get current velocities for t=0
    current_positions = np.array([x_current, y_current, z_current]).T
    velocities = get_velocity(current_positions, TrianT[:,:,-1], node_positions[:,:, -1], node_velocities[:,:, -1], centroid_indices)
    time_length -= 1 # update time length parameter
    for t in range(time_length, -1, -1):

        x_current -= velocities[:, 0] * dt
        y_current -= velocities[:, 1] * dt
        z_current -= velocities[:, 2] * dt

        kdtree = KDTree(centroids[:,:, t])

        current_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT[:,:, t], kdtree, current_positions, node_positions[:,:,t])

        x_traj[:, time_length  - t] = new_positions[:,0]
        y_traj[:, time_length  - t] = new_positions[:,1]
        z_traj[:, time_length  - t] = new_positions[:,2]

        velocities = get_velocity(new_positions, TrianT[:,:, t], node_positions[:,:,t], node_velocities[:,:, t], centroid_indices)


    return x_traj, y_traj, z_traj

node_positions = mesh_data['position'][0]
centroids = mesh_data['centroids'][0]

#Debugging (for checking centroids and faces)
if False:
    
    trianT = mesh_data['node_cons'][0][:,:,0]

    cen = centroids[:100, :]
    vertices = trianT[:100, :]
    vertices = vertices.astype(int)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], s=50, c='y')

    ax.scatter(node_positions[vertices[:,0], 0],
            node_positions[vertices[:,0], 1],
            node_positions[vertices[:,0], 2], s=50, c='b')
    ax.scatter(node_positions[vertices[:,1], 0],
            node_positions[vertices[:,1], 1],
            node_positions[vertices[:,1], 2], s=50, c='b')
    ax.scatter(node_positions[vertices[:,2], 0],
            node_positions[vertices[:,2], 1],
            node_positions[vertices[:,2], 2], s=50, c='b')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  


    plt.show()


# butter
def generate_concentric_circles(node_position, node_radius, outer_radii, num_points=1000):
    # Calculate nu0Cap and phi0Cap
    nu0Cap = np.arctan2(np.sqrt(node_position[0]**2 + node_position[1]**2), node_position[2])
    phi0Cap = np.arctan2(node_position[1], node_position[0])

    # Tangent vectors
    tang1 = np.array([np.cos(nu0Cap) * np.cos(phi0Cap), np.cos(nu0Cap) * np.sin(phi0Cap), -np.sin(nu0Cap)])
    tang2 = np.array([np.sin(nu0Cap) * np.sin(phi0Cap), -np.sin(nu0Cap) * np.cos(phi0Cap), 0])

    # Initialize arrays to hold points
    points = []

    # Loop through radii
    for rCap in outer_radii:
        for phiCap in np.linspace(0, 2 * np.pi, num_points):
            x_pt = node_position[0] + rCap * (np.cos(phiCap) * tang1[0] + np.sin(phiCap) * tang2[0])
            y_pt = node_position[1] + rCap * (np.cos(phiCap) * tang1[1] + np.sin(phiCap) * tang2[1])
            z_pt = node_position[2] + rCap * (np.cos(phiCap) * tang1[2] + np.sin(phiCap) * tang2[2])
            points.append([x_pt, y_pt, z_pt])

    return np.array(points)

# (Debug) Test projection 
if True:
    kdtree = KDTree(centroids[:, : , 0])
    particle_positions = generate_points(1, 10000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(particle_positions[:,0], particle_positions[:,1], particle_positions[:,2], s=1)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show() 
    plt.close()

    x_pos = mesh_data['position'][0][:, 0, :]
    y_pos = mesh_data['position'][0][:, 1, :]
    z_pos = mesh_data['position'][0][:, 2, :]
    TrianT = mesh_data['node_cons'][0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(2):
    
        ax.cla()

        new_positions, centroid_indices = particle_projection(mesh_data['node_cons'][0][:,:,0], kdtree, particle_positions, node_positions[:,:,0])

        ax.scatter(particle_positions[:,0], particle_positions[:,1], particle_positions[:,2], s=1)
        
        ax.scatter(new_positions[:,0], new_positions[:,1], new_positions[:,2], s=1)
        ax.plot_trisurf(x_pos[:, 0], y_pos[:, 0], z_pos[:, 0], triangles=TrianT[:, :, 0], color='grey', alpha=0.5)

        # Set the aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')  
        
        particle_positions =  new_positions

        plt.show()


node_velocities = mesh_data['velocity'][0]



x_traj, y_traj, z_traj = forward_particle_advection(mesh_data['node_cons'][0],
                                            centroids, 
                                            particle_positions,
                                            node_positions,
                                            node_velocities, 
                                            mesh_data['time_steps'][0])


bx_traj, by_traj, bz_traj = backward_particle_advection(mesh_data['node_cons'][0],
                                            centroids, 
                                            particle_positions,
                                            node_positions,
                                            node_velocities, 
                                            mesh_data['time_steps'][0])


#  butter
if False:

    def plot_trajectories_multiple_views(xt, yt, zt, TrianT, x_pos, y_pos, z_pos, time_steps):
        fig = plt.figure(figsize=(18, 6))
        
        # Define view angles (elevation, azimuth)
        views = [(30, 30), (0, 90), (90, 0)]
        
        total_time = xt.shape[1]

        # Loop through each time frame and plot the positions of the particles
        for t in range(total_time):
            fig.clear()
            
            for i, (elev, azim) in enumerate(views):
                ax = fig.add_subplot(1, len(views), i + 1, projection='3d')

                # Plot particle positions
                ax.scatter(xt[:, t], yt[:, t], zt[:, t], s=1, c='b', marker='o')

                # Plot the mesh surface using node positions and connections
                ax.plot_trisurf(x_pos[:, 0], y_pos[:, 0], z_pos[:, 0], triangles=TrianT[:, :, 0], color='grey', alpha=0.5)

                ax.set_title(f'Time frame: {time_steps[t]:.4f}\nView: elev={elev}, azim={azim}')
                ax.set_xlabel('X position')
                ax.set_ylabel('Y position')
                ax.set_zlabel('Z position')
                
                # Set the view angle
                ax.view_init(elev=elev, azim=azim)

                # Optionally, you can set limits to make the plot easier to follow
                ax.set_xlim([xt.min(), xt.max()])
                ax.set_ylim([yt.min(), yt.max()])
                ax.set_zlim([zt.min(), zt.max()])

            plt.pause(0.1)  # Pause to create an animation effect
            
        plt.show()


    # butter
    def plot_manifolds(TrianT, x_pos, y_pos, z_pos, vx, vy, vz):
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(121, projection='3d')
        
        # Plot the position mesh
        ax.plot_trisurf(x_pos[:, 0], y_pos[:, 0], z_pos[:, 0], triangles=TrianT[:, :, 0], color='blue', alpha=0.5)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        ax.set_title('Position Mesh')

        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot the velocity mesh
        ax2.plot_trisurf(vx[:, 0], vy[:, 0], vz[:, 0], triangles=TrianT[:, :, 0], color='red', alpha=0.5)
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Y position')
        ax2.set_zlabel('Z position')
        ax2.set_title('Velocity Mesh')

        plt.show()

        
    # butter
    def make_mesh(point_cloud):

        kdtree = KDTree(point_cloud)

        connections = np.zeros((len(point_cloud), 3))

        for i in range(len(point_cloud)):

            distances, indexes = kdtree.query(point_cloud[i, :], 6)

            connections[i, :] = indexes[1:]

        return connections 


# important
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

    # Project vec3 onto the tangent vectors
    x_local_1 = np.dot(vec3, vec1)
    x_local_2 = np.dot(vec3, vec2)

    # The local coordinates in the tangent plane
    return np.array([x_local_1, x_local_2])



# important
def FTLE_compute(node_positions, centroids, initial_positions, final_positions, TrianT, final_time, lam=1e-10):

    """
    Computes the Finite Time Lyaponov Exponent, for sparse data input

    Parameters:
    node_positions: Positions of the simplicial complex vertices
    particle positions: Kdtree of the particle positions
    centroids: Centroid of each face
    initial_positions: Initial Positions, dimensions (number of positions, 3)
    final_positions: Final Positions, dimensions (number of positions, 3)
    TrianT: Connections of vertices

    Returns:
    FTLE: Finite Time Lyaponov Exponent array
    """
    position_kdtree = KDTree(initial_positions)
    centroid_kdtree = KDTree(centroids)

    number_points = initial_positions.shape[0]

    FTLE = np.zeros(number_points)
    for i in range(number_points - 1):
        
        # Get Initial position data
        _, closest_indexes = position_kdtree.query(initial_positions[i,:], 5)
        _, tangent_face_index = centroid_kdtree.query(initial_positions[i,:], 1)  
        Iclosest_positions = initial_positions[closest_indexes[1:]]
        Itangent_face = TrianT[tangent_face_index, :]
        Itangent_face_positions = node_positions[Itangent_face, :]

        # Get Final position data
        Fclosest_positions = final_positions[closest_indexes[1:], :]
        _, Ftangent_face_index = centroid_kdtree.query(final_positions[i, :], 1)
        Ftangent_face = TrianT[Ftangent_face_index, :]
        Ftangent_face_positions = node_positions[Ftangent_face, :]

        # Arrays for storing positions with respect to the tangent plane
        Ilocal_tangent_coords = np.zeros((Iclosest_positions.shape[0], 2))
        Flocal_tangent_coords = np.zeros((Iclosest_positions.shape[0], 2))

        # Local tangent plane coordinates
        for j in range(Iclosest_positions.shape[0]):
            Ilocal_tangent_coords[j,:] = local_tangent_project(Itangent_face_positions[0, :],
                                            Itangent_face_positions[1, :],
                                            Itangent_face_positions[2, :], 
                                            Iclosest_positions[j, :])
            
            Flocal_tangent_coords[j,:] = local_tangent_project(Ftangent_face_positions[0, :],
                                Ftangent_face_positions[1, :],
                                Ftangent_face_positions[2, :], 
                                Fclosest_positions[j, :])

        combs = list(combinations(range(len(closest_indexes[1:])),2))
        
        ind1 = [comb[0] for comb in combs]
        ind2 = [comb[1] for comb in combs]

        X = np.zeros((2, len(combs)))
        Y = np.zeros((2, len(combs)))
        X[0, :] = Ilocal_tangent_coords[ind1, 0] - Ilocal_tangent_coords[ind2, 0]
        X[1, :] = Ilocal_tangent_coords[ind1, 1] - Ilocal_tangent_coords[ind2, 1]
        Y[0, :] = Flocal_tangent_coords[ind1, 0] - Flocal_tangent_coords[ind2, 0]
        Y[1, :] = Flocal_tangent_coords[ind1, 1] - Flocal_tangent_coords[ind2, 1]

        # Least square fit of flow map gradient
        A = Y@X.T + lam*max(1,len(closest_indexes))*np.eye(2)
        B = X@X.T + lam*max(1,len(closest_indexes))*np.eye(2)
        DF = A@np.linalg.inv(B)
        
        # Calculate FTLE as the largest singular value of DF
        FTLE[i] = np.log(np.linalg.norm(DF,2)) / final_time

    return FTLE




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
