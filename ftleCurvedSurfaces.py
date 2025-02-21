import numpy as np
from scipy.io import loadmat
import pyvista as pv
from time import  time as timer 
from Mesh_FTLE import FTLE_compute
from scipy.spatial import cKDTree
from numba import njit
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
from itertools import combinations


def load_Matlabmesh_data(file_path):


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
                               ('position', position_array.dtype, position_array.shape),
                               ('velocity', velocity_array.dtype, velocity_array.shape),
                               ('node_number', np.int32),
                               ('time_steps', np.float64, time_steps.shape),
                               ('total_time', np.int32)])

    all_data_array = np.zeros(1, dtype=all_data_dtype)
    all_data_array['node_cons'][0] = TrianT_temp
    all_data_array['position'][0] = position_array
    all_data_array['velocity'][0] = velocity_array
    all_data_array['node_number'][0] = node_number
    all_data_array['time_steps'][0] = time_steps
    all_data_array['total_time'][0] = total_time

    return all_data_array



def particle_projection(node_connections, kdtree, particle_positions, node_positions):
    """
    Simplified particle projection: Projects all particles to the first closest centroid face.
    
    Parameters: 
    TrianT: The vertex connections (Nx3 array, each row contains the indices of the vertices forming a face)
    kdtree: KDTree built from the centroids of TrianT
    particle_positions: The positions of particles to project (Mx3 array)
    node_positions: The vertex positions of the mesh (Px3 array)
    
    Returns:
    new_positions: Projected positions of the particles
    centroid_indices: Indices of the closest centroids used for projection
    """

    # Query the nearest centroid for each particle
    _, indices = kdtree.query(particle_positions, k=1)
    
    # Retrieve the corresponding faces
    nearest_faces = node_connections[indices]
    
    # Extract the vertices of the nearest faces
    A = node_positions[nearest_faces[:, 0]]
    B = node_positions[nearest_faces[:, 1]]
    C = node_positions[nearest_faces[:, 2]]

    # Perform vectorized projection for all particles
    def project(A, B, C, P):
        """Projects point P onto the plane defined by triangle (A, B, C)"""
        normal = np.cross(B - A, C - A)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        vector_to_plane = A - P
        distance = np.einsum('ij,ij->i', vector_to_plane, normal)
        return P + distance[:, None] * normal

    # Compute the new positions
    new_positions = project(A, B, C, particle_positions)
    
    return new_positions, indices


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


def interpolate(initial_data, later_data, t_fraction):

    return t_fraction * initial_data  + (1-t_fraction) * later_data


def compute_centroids(node_connections, node_positions):
    """
    Computes the centroids of triangles defined by `node_connections` over all time steps.

    Parameters:
        node_connections (ndarray): (M, 3) array of node indices defining each triangle.
        node_positions (ndarray): (N, 3, T) array of node positions, where:
                                  - N = number of nodes,
                                  - 3 = spatial coordinates (x, y, z),
                                  - T = number of time steps.

    Returns:
        centroids (ndarray): (M, 3, T) array of computed centroids over time.
    """

    return (node_positions[node_connections[:, 0], :, :] + node_positions[node_connections[:, 1], :, :] + node_positions[node_connections[:, 2], :, :]) / 3.0





def forward_particle_advection(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt):
    """
    Particle advection alternating between Backward Euler and Forward Euler.
    
    Parameters:
    TrianT: Array of triangle face indices over time [num_faces, 3].
    centroids: Array of face centroids over time [num_faces, 3, time_length].
    particle_positions: Initial positions of particles [num_particles, 3].
    node_positions: Array of node positions over time [num_nodes, 3, time_length].
    node_velocities: Array of node velocities over time [num_nodes, 3, time_length].
    time_steps: Array of time steps.
    dt: Time step size.
    
    Returns:
    x_traj, y_traj, z_traj: Trajectories of particles over time.
    """

    centroids = compute_centroids(node_connections, node_positions)

    num_particles = len(particle_positions[:, 0])
    fine_time = np.arange(0, final_time - 1 + dt - initial_time, dt)
    fine_time_length = len(fine_time)

    
    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial KDTree
    kdtree = cKDTree(centroids[:, :, initial_time])

    # Project initial particle positions onto the surface
    new_positions, centroid_indices = particle_projection(node_connections, kdtree, particle_positions, node_positions[:, :, initial_time])

    # Set initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    fine_time = fine_time[:-1] # remove last index
    

    # Main time-stepping loop
    for t_index, t in enumerate(fine_time):
        
        floor_t = np.floor(t).astype(int) 
        ceiling_t = np.ceil(t).astype(int) 
        t_fraction = t - floor_t

        t_next = t_fraction + dt


        current_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t], t_fraction)
        current_node_velocities = interpolate(node_velocities[:, :, floor_t], node_velocities[:, :, ceiling_t], t_fraction)

        next_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t], t_next)
        next_node_velocities = interpolate(node_velocities[:, :, floor_t], node_velocities[:, :, ceiling_t], t_next)
        next_centroids = interpolate(centroids[:, :, floor_t], centroids[:, :, ceiling_t], t_next)

        # Build the KDTree for the current time step
        kdtree = cKDTree(next_centroids)
        current_positions = np.array([x_current, y_current, z_current]).T

        if t_index % 2 == 1:
            velocities = get_velocity(current_positions, node_connections, current_node_positions, current_node_velocities, centroid_indices)

            # Forward Euler update
            x_current += velocities[:, 0] * dt
            y_current += velocities[:, 1] * dt
            z_current += velocities[:, 2] * dt

        else:  # Odd time steps - Backward Euler
                    # Initial guess for Backward Euler
                    velocities = get_velocity(current_positions, node_connections, current_node_positions, current_node_velocities, centroid_indices)
                    x_guess = x_current + velocities[:, 0] * dt
                    y_guess = y_current + velocities[:, 1] * dt
                    z_guess = z_current + velocities[:, 2] * dt

                    # Iteratively correct the position using the backward scheme
                    for _ in range(3):  # Fixed number of iterations for simplicity
                        guess_positions = np.array([x_guess, y_guess, z_guess]).T
                        velocities = get_velocity(guess_positions, node_connections, next_node_positions, next_node_velocities, centroid_indices)

                        x_guess = x_current + velocities[:, 0] * dt
                        y_guess = y_current + velocities[:, 1] * dt
                        z_guess = z_current + velocities[:, 2] * dt

                    # Update positions after convergence
                    x_current = x_guess
                    y_current = y_guess
                    z_current = z_guess

        # Project updated positions onto the surface
        updated_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(node_connections, kdtree, updated_positions, next_node_positions)

        # Update trajectories
        x_traj[:, t_index+1] = new_positions[:, 0]
        y_traj[:, t_index+1] = new_positions[:, 1]
        z_traj[:, t_index+1] = new_positions[:, 2]


    return x_traj, y_traj, z_traj, centroids


def fast_forward_particle_advection(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt):
    """
    Particle advection alternating between Backward Euler and Forward Euler.
    
    Parameters:
    TrianT: Array of triangle face indices over time [num_faces, 3, time_length].
    centroids: Array of face centroids over time [num_faces, 3, time_length].
    particle_positions: Initial positions of particles [num_particles, 3].
    node_positions: Array of node positions over time [num_nodes, 3, time_length].
    node_velocities: Array of node velocities over time [num_nodes, 3, time_length].
    time_steps: Array of time steps.
    dt: Time step size.
    
    Returns:
    x_traj, y_traj, z_traj: Trajectories of particles over time.
    """

    centroids = compute_centroids(node_connections, node_positions)

    
    num_particles = len(particle_positions[:, 0])
    fine_time = np.arange(0, final_time - 1 + dt - initial_time, dt)
    fine_time_length = len(fine_time)
    

    # Initialize trajectories
    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial KDTree
    kdtree = cKDTree(centroids[:, :, initial_time])

    # Project initial particle positions onto the surface
    new_positions, centroid_indices = particle_projection(node_connections, kdtree, particle_positions, node_positions[:, :, initial_time])

    # Set initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    current_positions = np.array([x_current, y_current, z_current]).T
    velocities = get_velocity(current_positions, node_connections, node_positions[:,:, initial_time], node_velocities[:,:, initial_time], centroid_indices)

    fine_time = fine_time[:-1] # remove last index

    for t_index, t in enumerate(fine_time):
        
        
        x_current += velocities[:, 0] * dt
        y_current += velocities[:, 1] * dt
        z_current += velocities[:, 2] * dt


        if t_index % 2 == 1:

            floor_t = np.floor(t).astype(int) 
            ceiling_t = np.ceil(t).astype(int) 
            t_fraction = t - floor_t


            t_next = t_fraction + dt

            next_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t], t_next)
            next_node_velocities = interpolate(node_velocities[:, :, floor_t], node_velocities[:, :, ceiling_t], t_next)
            next_centroids = interpolate(centroids[:, :, floor_t], centroids[:, :, ceiling_t], t_next)

            kdtree = cKDTree(next_centroids)
            current_positions = np.array([x_current, y_current, z_current]).T
            new_positions, centroid_indices = particle_projection(node_connections, kdtree, current_positions, next_node_positions)
            velocities = get_velocity(new_positions, node_connections, next_node_positions, next_node_velocities, centroid_indices)
            x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

        x_traj[:, t_index + 1] = x_current
        y_traj[:, t_index + 1] = y_current
        z_traj[:, t_index + 1] = z_current



    return x_traj, y_traj, z_traj, centroids


def RK4_forward_particle_advection(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt):


    centroids = compute_centroids(node_connections, node_positions)
                   
    num_particles = len(particle_positions[:, 0])

    fine_time = np.arange(0, final_time - 1 + dt - initial_time, dt)
 
    fine_time_length = len(fine_time)

    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial kdtree and project initial particles onto the surface
    kdtree = cKDTree(centroids[:, :, initial_time])
    new_positions, centroid_indices = particle_projection(node_connections, kdtree, particle_positions, node_positions[:, :, initial_time])

    # Initialize trajectories and running positions
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]
    x_current, y_current, z_current = x_traj[:, 0], y_traj[:, 0], z_traj[:, 0]

    fine_time = fine_time[:-1] # remove last index

    for t_index, t in enumerate(fine_time):

        floor_t = np.floor(t).astype(int) 
        ceiling_t = np.ceil(t).astype(int) 
        t_fraction = t - floor_t

        t_next = t_fraction + dt

        next_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t], t_next)
        next_node_velocities = interpolate(node_velocities[:, :, floor_t], node_velocities[:, :, ceiling_t], t_next)
        next_centroids = interpolate(centroids[:, :, floor_t], centroids[:, :, ceiling_t], t_next)

        def velocity_at_position(x, y, z):
            pos = np.array([x, y, z]).T
            velocities = get_velocity(pos, node_connections, next_node_positions, next_node_velocities, centroid_indices)
            return velocities[:, 0], velocities[:, 1], velocities[:, 2]


        # Calculate RK4 stages
        k1_x, k1_y, k1_z = velocity_at_position(x_current, y_current, z_current)
        k2_x, k2_y, k2_z = velocity_at_position(x_current + 0.5 * dt * k1_x, y_current + 0.5 * dt * k1_y, z_current + 0.5 * dt * k1_z)
        k3_x, k3_y, k3_z = velocity_at_position(x_current + 0.5 * dt * k2_x, y_current + 0.5 * dt * k2_y, z_current + 0.5 * dt * k2_z)
        k4_x, k4_y, k4_z = velocity_at_position(x_current + dt * k3_x, y_current + dt * k3_y, z_current + dt * k3_z)

        x_current += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_current += (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        z_current += (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

        # Project particles to surface
        kdtree = cKDTree(next_centroids)
        next_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(node_connections, kdtree, next_positions, next_node_positions)


        # Update stored trajectories
        x_traj[:, t_index + 1] = new_positions[:, 0]
        y_traj[:, t_index + 1] = new_positions[:, 1]
        z_traj[:, t_index + 1] = new_positions[:, 2]
        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj, centroids





def backward_particle_advection(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt):
    """
    Particle advection backwards in time alternating between Backward Euler and Forward Euler.
    
    Parameters:
    TrianT: Array of triangle face indices over time [num_faces, 3, time_length].
    centroids: Array of face centroids over time [num_faces, 3, time_length].
    particle_positions: Initial positions of particles [num_particles, 3].
    node_positions: Array of node positions over time [num_nodes, 3, time_length].
    node_velocities: Array of node velocities over time [num_nodes, 3, time_length].
    time_steps: Array of time steps.
    dt: Time step size.
    
    Returns:
    x_traj, y_traj, z_traj: Trajectories of particles over time.
    """
    centroids = compute_centroids(node_connections, node_positions)

    # Reverse time indexing
    node_positions = node_positions[:,:, ::-1]
    node_velocities = node_velocities[:,:, ::-1]
    centroids = centroids[:,:, ::-1]

    
    num_particles = len(particle_positions[:, 0])
    fine_time = np.arange(0, final_time - 1 + dt - initial_time, dt)
    dt = -dt

    fine_time_length = len(fine_time)
 
    # Initialize trajectories
    x_traj = np.zeros((num_particles, fine_time_length ))
    y_traj = np.zeros((num_particles, fine_time_length ))
    z_traj = np.zeros((num_particles, fine_time_length ))

    # Build the initial KDTree for the last time step
    kdtree = cKDTree(centroids[:, :, initial_time])

    # Project initial particle positions onto the surface
    new_positions, centroid_indices = particle_projection(node_connections, kdtree, particle_positions, node_positions[:, :, initial_time])

    # Set initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions 
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    fine_time = fine_time[:-1] # remove last index


    for t_index, t in enumerate(fine_time):
      
        
        floor_t = np.floor(t).astype(int) 
        ceiling_t = np.ceil(t).astype(int) 
        t_fraction = (t - floor_t)

        t_next = t_fraction - dt


        current_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t] , t_fraction)
        current_node_velocities = interpolate(node_velocities[:, :, ceiling_t], node_velocities[:, :, floor_t], t_fraction)


        next_node_positions = interpolate(node_positions[:, :, floor_t ], node_positions[:, :, ceiling_t] , t_next)
        next_node_velocities = interpolate(node_velocities[:, :, floor_t ], node_velocities[:, :, ceiling_t], t_next)
        next_centroids = interpolate(centroids[:, :, floor_t ], centroids[:, :, ceiling_t], t_next)

        # Build the KDTree for the current time step
        kdtree = cKDTree(next_centroids)
        current_positions = np.array([x_current, y_current, z_current]).T

        if t_index % 2 == 0:
            velocities = get_velocity(current_positions, node_connections, current_node_positions, current_node_velocities, centroid_indices)

            # Forward Euler update
            x_current += velocities[:, 0] * dt
            y_current += velocities[:, 1] * dt
            z_current += velocities[:, 2] * dt

        else:  # Odd time steps - Backward Euler
                    # Initial guess for Backward Euler
                    velocities = get_velocity(current_positions, node_connections, current_node_positions, current_node_velocities, centroid_indices)
                    x_guess = x_current + velocities[:, 0] * dt
                    y_guess = y_current + velocities[:, 1] * dt
                    z_guess = z_current + velocities[:, 2] * dt

                    # Iteratively correct the position using the backward scheme
                    for _ in range(3):  
                        guess_positions = np.array([x_guess, y_guess, z_guess]).T
                        velocities = get_velocity(guess_positions, node_connections, next_node_positions, next_node_velocities, centroid_indices)

                        x_guess = x_current + velocities[:, 0] * dt
                        y_guess = y_current + velocities[:, 1] * dt
                        z_guess = z_current + velocities[:, 2] * dt

                    # Update positions after convergence
                    x_current = x_guess
                    y_current = y_guess
                    z_current = z_guess

        # Project updated positions onto the surface
        updated_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(node_connections, kdtree, updated_positions, current_node_positions)

        # Update trajectories
        x_traj[:, t_index+1] = new_positions[:, 0]
        y_traj[:, t_index+1] = new_positions[:, 1]
        z_traj[:, t_index+1] = new_positions[:, 2]


    return x_traj, y_traj, z_traj, centroids


def fast_backward_particle_advection(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt):


    centroids = compute_centroids(node_connections, node_positions)

    # Reverse time indexing
    node_positions = node_positions[:,:, ::-1]
    node_velocities = node_velocities[:,:, ::-1]
    centroids = centroids[:,:, ::-1]

    

    num_particles = len(particle_positions[:, 0])    
    fine_time = np.arange(0, final_time - 1 + dt - initial_time, dt)
    dt = -dt

    fine_time_length = len(fine_time)

    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))


    # Build the initial kdtree
    kdtree = cKDTree(centroids[:,:, initial_time])


    # Make sure the initial particles are on the surface
    new_positions, centroid_indices = particle_projection(node_connections, kdtree, particle_positions, node_positions[:,:, initial_time]) 

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
    velocities = get_velocity(current_positions, node_connections, node_positions[:,:, initial_time], node_velocities[:,:, initial_time], centroid_indices)


    fine_time = fine_time[:-1] # remove last index

    for t_index ,t in enumerate(fine_time):

        x_current += velocities[:, 0] * dt
        y_current += velocities[:, 1] * dt
        z_current += velocities[:, 2] * dt

        
        if t_index % 2 == 1:


            floor_t = np.floor(t).astype(int) 
            ceiling_t = np.ceil(t).astype(int) 
            t_fraction = (t - floor_t)

            t_next = t_fraction - dt

            next_node_positions = interpolate(node_positions[:, :, floor_t ], node_positions[:, :, ceiling_t] , t_next)
            next_node_velocities = interpolate(node_velocities[:, :, floor_t ], node_velocities[:, :, ceiling_t], t_next)
            next_centroids = interpolate(centroids[:, :, floor_t ], centroids[:, :, ceiling_t], t_next)

            kdtree = cKDTree(next_centroids)
            current_positions = np.array([x_current, y_current, z_current]).T
            new_positions, centroid_indices = particle_projection(node_connections, kdtree, current_positions, next_node_positions)
            velocities = get_velocity(new_positions, node_connections, next_node_positions, next_node_velocities, centroid_indices)
            x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]


        x_traj[:, t_index + 1] = x_current
        y_traj[:, t_index + 1] = y_current
        z_traj[:, t_index + 1] = z_current



    return x_traj, y_traj, z_traj, centroids


def RK4_backward_particle_advection(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt):


    centroids = compute_centroids(node_connections, node_positions)

    # Reverse time indexing
    node_positions = node_positions[:,:, ::-1]
    node_velocities = node_velocities[:,:, ::-1]
    centroids = centroids[:,:, ::-1]

    num_particles = len(particle_positions[:, 0])

    fine_time = np.arange(0, final_time - 1 + dt - initial_time, dt)
    dt = -dt
 
    fine_time_length = len(fine_time)

    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial kdtree and project initial particles onto the surface
    kdtree = cKDTree(centroids[:, :, initial_time])
    new_positions, centroid_indices = particle_projection(node_connections, kdtree, particle_positions, node_positions[:, :, initial_time])

    # Initialize trajectories and running positions
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]
    x_current, y_current, z_current = x_traj[:, 0], y_traj[:, 0], z_traj[:, 0]

    fine_time = fine_time[:-1]

    for t_index, t in enumerate(fine_time):

        floor_t = np.floor(t).astype(int) 
        ceiling_t = np.ceil(t).astype(int) 
        t_fraction = t - floor_t

        t_next = t_fraction - dt

        next_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t], t_next)
        next_node_velocities = interpolate(node_velocities[:, :, floor_t], node_velocities[:, :, ceiling_t], t_next)
        next_centroids = interpolate(centroids[:, :, floor_t], centroids[:, :, ceiling_t], t_next)

        def velocity_at_position(x, y, z):
            pos = np.array([x, y, z]).T
            velocities = get_velocity(pos, node_connections, next_node_positions, next_node_velocities, centroid_indices)
            return velocities[:, 0], velocities[:, 1], velocities[:, 2]


        # Calculate RK4 stages
        k1_x, k1_y, k1_z = velocity_at_position(x_current, y_current, z_current)
        k2_x, k2_y, k2_z = velocity_at_position(x_current + 0.5 * dt * k1_x, y_current + 0.5 * dt * k1_y, z_current + 0.5 * dt * k1_z)
        k3_x, k3_y, k3_z = velocity_at_position(x_current + 0.5 * dt * k2_x, y_current + 0.5 * dt * k2_y, z_current + 0.5 * dt * k2_z)
        k4_x, k4_y, k4_z = velocity_at_position(x_current + dt * k3_x, y_current + dt * k3_y, z_current + dt * k3_z)

        x_current += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_current += (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        z_current += (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

        # Project particles to surface
        kdtree = cKDTree(next_centroids)
        next_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(node_connections, kdtree, next_positions, next_node_positions)


        # Update stored trajectories
        x_traj[:, t_index + 1] = new_positions[:, 0]
        y_traj[:, t_index + 1] = new_positions[:, 1]
        z_traj[:, t_index + 1] = new_positions[:, 2]
        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj, centroids









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






def FTLE_compute(node_connections, node_positions, centroids, initial_positions, final_positions, initial_time, final_time, neighborhood = 15, lam=1e-10):

    """
    Computes the Finite Time Lyaponov Exponent, for sparse data input

    Parameters:
    node_positions: Positions of the simplicial complex vertices
    particle positions: Kdtree of the particle positions
    centroids: Centroid of each face
    initial_positions: Initial Positions, dimensions (number of positions, 3)
    final_positions: Final Positions, dimensions (number of positions, 3)
    TrianT: Connections of vertices
    lam: Regularity constant

    Returns:
    FTLE: Finite Time Lyaponov Exponent 
    """

    position_kdtree = cKDTree(initial_positions)
    centroid_kdtree = cKDTree(centroids)

    number_points = initial_positions.shape[0]


    FTLE = np.zeros(number_points)
    for i in range(number_points - 1):
        
        # Get Initial position data
        _, closest_indexes = position_kdtree.query(initial_positions[i,:], neighborhood  + 1)
        _, tangent_face_index = centroid_kdtree.query(initial_positions[i,:], 1)  
        Iclosest_positions = initial_positions[closest_indexes[1:]]
        Itangent_face = node_connections[tangent_face_index, :]
        Itangent_face_positions = node_positions[Itangent_face, :]


       
        # Get Final position data
        Fclosest_positions = final_positions[closest_indexes[1:], :]
        _, Ftangent_face_index = centroid_kdtree.query(final_positions[i, :], 1)
        Ftangent_face = node_connections[Ftangent_face_index, :]
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
        FTLE[i] = np.log(np.linalg.norm(DF,2)) / abs(final_time -initial_time)

    return FTLE










def run_simulation(node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, time_steps, dt, direction, scheme_type, neighborhood=15, lam=1e-10):
    """
    Runs a particle advection simulation on a 2D surface embedded in 3D space.

    This function simulates the movement of particles over a triangulated mesh using 
    different numerical advection schemes. It supports both forward and backward advection 
    and computes the Finite-Time Lyapunov Exponent (FTLE) field.

    Parameters:
        node_connections (ndarray): 
            (M, 3) Connectivity matrix defining the mesh topology, where each row 
            represents a triangle formed by three node indices.

        node_positions (ndarray): 
            (N, 3, T) or (N, 3) Array containing the (x, y, z) positions of each node over time.
            If time-dependent, it should have shape (N, 3, T), otherwise, it will be expanded.

        node_velocities (ndarray): 
            (N, 3, T) or (N, 3) Array containing velocity vectors at each node.
            If not time-dependent, it will be expanded to (N, 3, T).

        particle_positions (ndarray): 
            (P, 3) Initial positions of the particles in 3D space.

        initial_time (int): 
            Index of the initial time step.

        final_time (int): 
            Index of the final time step.

        time_steps (ndarray): 
            (T,) Array of available time steps.

        dt (float): 
            Time step size for numerical integration. Must be in the range (0,1].

        direction (str): 
            Specifies whether the advection is "forward" or "backward" in time.

        scheme_type (str): 
            Specifies the numerical integration scheme:
            - "euler": Standard Euler method.
            - "fasteuler": Optimized Euler method.
            - "rk4": Fourth-order Runge-Kutta method.

        neighborhood (int, optional): 
            Number of neighboring nodes to consider when computing FTLE. Default is 15.

    Returns:
        Ftle (ndarray): 
            (N,) or (P,) Array of FTLE values computed at the particle positions.

    Raises:
        ValueError: If input parameters are inconsistent or invalid.
        RuntimeError: If trajectory computation or FTLE calculation fails.

    Notes:
        - If node_positions or node_velocities are time-independent (i.e., shape (N,3)), 
          they are expanded to (N,3,T) by copying across time steps.
        - The function supports both forward and backward advection, with time 
          adjustments for backward mode.
        - Particle trajectories are computed using the chosen advection scheme, and 
          FTLE values are calculated based on initial and final particle positions.
    """

    total_time = len(time_steps)
    
    # Ensure node_positions is [total_nodes, 3, total_time]
    if node_positions.ndim == 2:  # Shape [total_nodes, 3]
        node_positions = np.tile(node_positions[:, :, np.newaxis], (1, 1, total_time))

    # Ensure node_velocities is [total_nodes, 3, total_time]
    if node_velocities.ndim == 2:  # Shape [total_nodes, 3]
        node_velocities = np.tile(node_velocities[:, :, np.newaxis], (1, 1, total_time))



    direction = direction.lower()
    scheme_type = scheme_type.lower()

    # Time consistency checks
    if initial_time not in time_steps:
        raise ValueError("Error: Initial time must be in the given time values")
    if final_time not in time_steps:
        raise ValueError("Error: Final time must be in the given time values")
    if initial_time == final_time:
        raise ValueError("Error: Initial time and final time for advection should be different")

    if direction == "forward":
        if initial_time > final_time:
            raise ValueError("Error: Forward advection final time must be greater than the initial time")
    elif direction == "backward":
        if initial_time < final_time:
            raise ValueError("Error: Backward advection initial time must be greater than the final time")

        # Adjust times for indexing in backward advection, and acount for time reversal in backward schemes
        temp_initial_time = len(time_steps) - final_time - 1
        final_time = len(time_steps) - initial_time  - 1
        initial_time = final_time
        final_time = temp_initial_time
        

    else:
        raise ValueError("Error: Specify a direction for the particle advection: 'forward' or 'backward'.")

    if dt > 1 or dt <= 0:
        raise ValueError("Error: The value dt must be in the interval (0,1]")



    # Advection scheme processing
    x_traj, y_traj, z_traj = None, None, None  # Initialize as None

    if direction == "forward":
        if scheme_type == "euler":
            x_traj, y_traj, z_traj, centroids = forward_particle_advection(
                node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt
            )
        elif scheme_type == "fasteuler":
            x_traj, y_traj, z_traj, centroids = fast_forward_particle_advection(
                node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt
            )
           # plot_particle_trajectories(x_traj, y_traj, z_traj, len(x_traj[0,:]))

        elif scheme_type == "rk4":
            x_traj, y_traj, z_traj, centroids = RK4_forward_particle_advection(
                node_connections, node_positions, node_velocities, particle_positions, initial_time, final_time, dt
            )
        else:
            raise ValueError("Error: Specify a scheme for the particle advection: 'euler', 'fasteuler', or 'rk4'.")

    elif direction == "backward":
        if scheme_type == "euler":
            x_traj, y_traj, z_traj, centroids = backward_particle_advection(
                node_connections,
                node_positions, node_velocities, particle_positions, initial_time, final_time, dt
            )
        elif scheme_type == "fasteuler":
            x_traj, y_traj, z_traj, centroids = fast_backward_particle_advection(
                node_connections,
                node_positions, node_velocities, particle_positions, initial_time, final_time, dt
            )
        elif scheme_type == "rk4":
            x_traj, y_traj, z_traj, centroids = RK4_backward_particle_advection(
                node_connections,
                node_positions, node_velocities, particle_positions, initial_time, final_time, dt
            )
        else:
            raise ValueError("Error: Specify a scheme for the particle advection: 'euler', 'fasteuler', or 'rk4'.")

    # Check if trajectory values are None
    if x_traj is None or y_traj is None or z_traj is None:
        raise RuntimeError("Error: Particle trajectory computation returned None. Check the computation functions.")

    # Compute FTLE
    Ftle = FTLE_compute(
        node_connections,
        node_positions[:,:, initial_time],
        centroids[:,:, initial_time],
        particle_positions,
        np.vstack([x_traj[:, -1], y_traj[:,-1], z_traj[:, -1]]).T,
        initial_time,
        final_time,
        neighborhood,
        lam
    )

    if Ftle is None:
        raise RuntimeError("Error: FTLE computation returned None.")

    return Ftle



#file_path = f'C:/Users/bafen/OneDrive/Documents/Research/Dynmaic Morphoskeleton/Mesh_LCS/mesh_data.mat'
file_path = f'C:/Users/bafen/OneDrive/Documents/Research/Dynmaic Morphoskeleton/Mesh_LCS/dataHeart.mat'

mesh_data = load_Matlabmesh_data(file_path)

node_positions = mesh_data['position'][0]
node_velocities = mesh_data['velocity'][0]
particle_positions = node_positions[:,:,0]

dt = 0.1



direction = "forward"
initial_time = 0
final_time = 28



start_time = timer()
fast_fftle = run_simulation(mesh_data['node_cons'][0], node_positions, node_velocities, particle_positions, initial_time, final_time, mesh_data['time_steps'][0], dt, direction, "fasteuler", neighborhood=15)
print("fasteuler: ", timer() - start_time)

start_time = timer()
fftle = run_simulation(mesh_data['node_cons'][0], node_positions, node_velocities, particle_positions, initial_time, final_time, mesh_data['time_steps'][0], dt, direction, "euler", neighborhood=15)
print("euler: ", timer() - start_time)

start_time = timer()
Rfftle = run_simulation(mesh_data['node_cons'][0], node_positions, node_velocities, particle_positions, initial_time, final_time, mesh_data['time_steps'][0], dt, direction, "RK4", neighborhood=15)
print("RK4: ", timer() - start_time)





# Define scalar bar settings
scalar_bar_args = {
    "vertical": True,
    "title_font_size": 12,
    "label_font_size": 10,
    "n_labels": 5,  # Reduce number of labels
    "position_x": 0.85,  # Adjust position
    "position_y": 0.1,
    "width": 0.1,
    "height": 0.7
}

# Extract node positions (first time step)
points = node_positions[:, :, 0]  

# Initialize PyVista Plotter with 3 subplots
window_size = (1920, 1080)
pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=window_size)

# === SUBPLOTS ===
pl.subplot(0, 0)
pl.add_points(points, scalars=fast_fftle, cmap='jet', point_size=5.0, scalar_bar_args=scalar_bar_args)
pl.add_title('Fast Euler Scheme')


pl.subplot(0, 1)
pl.add_points(points, scalars=fftle, cmap='jet', point_size=5.0, scalar_bar_args=scalar_bar_args)
pl.add_title('Euler Scheme')

pl.subplot(0, 2)
pl.add_points(points, scalars=Rfftle, cmap='jet', point_size=5.0, scalar_bar_args=scalar_bar_args)
pl.add_title('RK4 Scheme')


# # Save the plot as an image
screenshot_filename = "repoFTLEscaterExample.png"
pl.screenshot(screenshot_filename)
print(f"Screenshot saved as: {screenshot_filename}")

# Show plot (optional, remove if only saving)
pl.show()




scalar_bar_args = {
    "vertical": True,            # Vertical color bar
    "title_font_size": 12,        # Title size
    "label_font_size": 10,        # Label size
    "n_labels": 5,                # Reduce the number of tick labels
    "position_x": 0.85,           # Adjust position in subplot
    "position_y": 0.1,            # Adjust position in subplot
    "width": 0.1,                 # Adjust width
    "height": 0.7                 # Adjust height
}

# Create PyVista mesh
surf = pv.PolyData(node_positions[:, :, 0])  # Use first time step positions
faces = np.hstack((np.full((mesh_data['node_cons'][0].shape[0], 1), 3), mesh_data['node_cons'][0])).flatten()
surf.faces = faces

# Assign scalar FTLE values
surf['fastFTLE'] = fast_fftle
surf['FFTLE'] = fftle
surf['Rfftle'] = Rfftle

surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)

# Plot settings
smooth_interpolation = True
plot_kwargs = dict(smooth_shading=True, show_edges=False, ambient=0.5, diffuse=0.6, specular=0.3)

# Initialize PyVista plotter with 3 subplots
window_size = (1920, 1080)
pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=window_size)

# === SUBPLOTS ===
pl.subplot(0, 0)
pl.add_mesh(surf, scalars='fastFTLE', cmap='jet', interpolate_before_map=smooth_interpolation, scalar_bar_args=scalar_bar_args, **plot_kwargs)
pl.add_title('Fast Euler Scheme')

pl.subplot(0, 1)
pl.add_mesh(surf, scalars='FFTLE', cmap='jet', interpolate_before_map=smooth_interpolation, scalar_bar_args=scalar_bar_args, **plot_kwargs)
pl.add_title('Euler Scheme')

pl.subplot(0, 2)
pl.add_mesh(surf, scalars='Rfftle', cmap='jet', interpolate_before_map=smooth_interpolation, scalar_bar_args=scalar_bar_args, **plot_kwargs)
pl.add_title('RK4 Scheme')

screenshot_filename = "repoftleSchemeExample.png"
pl.screenshot(screenshot_filename)
print(f"Screenshot saved as: {screenshot_filename}")


pl.show()


















