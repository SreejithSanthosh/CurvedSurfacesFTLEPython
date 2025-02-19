from numba import njit
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv


# @njit
# def project(A, B, C, position):
#     """
#     Project the current position to the plane formed by points A,B,C

#     Parameters:
#     A, B, C: Points in R^3
#     position: the point to project onto the ABC plane
#     """
#     vec1 = B-A
#     vec2 = C-A
#     vec3 = position - A

#     normal_vector = np.cross(vec1, vec2)

#     normal_vector = normal_vector / np.linalg.norm(normal_vector)

#     projection_onto_normal = np.dot(vec3, normal_vector) * normal_vector
            
#     # Subtract the projection from vector3 to get the projection onto the (local)plane
#     projection_onto_plane = vec3 - projection_onto_normal
#     return A + projection_onto_plane


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


def particle_projection(TrianT, kdtree, particle_positions, node_positions):
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
    nearest_faces = TrianT[indices]
    
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







def NEWforward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):
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
 
    time_length = len(time_steps) 
    num_particles = len(particle_positions[:, 0])
    

    fine_time = np.arange(0, time_length - 1 + dt, dt)
 
    fine_time_length = len(fine_time)

    

    # Initialize trajectories
    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial KDTree
    kdtree = KDTree(centroids[:, :, 0])

    # Project initial particle positions onto the surface
    new_positions, centroid_indices = particle_projection(TrianT, kdtree, particle_positions, node_positions[:, :, 0])

    # Set initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    fine_time = fine_time[:-1]
    

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
        kdtree = KDTree(next_centroids)
        current_positions = np.array([x_current, y_current, z_current]).T

        if t_index % 2 == 0:
            velocities = get_velocity(current_positions, TrianT, current_node_positions, current_node_velocities, centroid_indices)

            # Forward Euler update
            x_current += velocities[:, 0] * dt
            y_current += velocities[:, 1] * dt
            z_current += velocities[:, 2] * dt

        else:  # Odd time steps - Backward Euler
                    # Initial guess for Backward Euler
                    velocities = get_velocity(current_positions, TrianT, current_node_positions, current_node_velocities, centroid_indices)
                    x_guess = x_current + velocities[:, 0] * dt
                    y_guess = y_current + velocities[:, 1] * dt
                    z_guess = z_current + velocities[:, 2] * dt

                    # Iteratively correct the position using the backward scheme
                    for _ in range(3):  # Fixed number of iterations for simplicity
                        guess_positions = np.array([x_guess, y_guess, z_guess]).T
                        velocities = get_velocity(guess_positions, TrianT, next_node_positions, next_node_velocities, centroid_indices)

                        x_guess = x_current + velocities[:, 0] * dt
                        y_guess = y_current + velocities[:, 1] * dt
                        z_guess = z_current + velocities[:, 2] * dt

                    # Update positions after convergence
                    x_current = x_guess
                    y_current = y_guess
                    z_current = z_guess

        # Project updated positions onto the surface
        updated_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT, kdtree, updated_positions, next_node_positions)

        # Update trajectories
        x_traj[:, t_index+1] = new_positions[:, 0]
        y_traj[:, t_index+1] = new_positions[:, 1]
        z_traj[:, t_index+1] = new_positions[:, 2]


    return x_traj, y_traj, z_traj


def NEW_fast_forward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):
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
    
    time_length = len(time_steps)
    num_particles = len(particle_positions[:, 0])

    fine_time = np.arange(0, time_length - 1 + dt, dt)
 
    fine_time_length = len(fine_time)
    

    # Initialize trajectories
    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial KDTree
    kdtree = KDTree(centroids[:, :, 0])

    # Project initial particle positions onto the surface
    new_positions, centroid_indices = particle_projection(TrianT, kdtree, particle_positions, node_positions[:, :, 0])

    # Set initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    current_positions = np.array([x_current, y_current, z_current]).T
    velocities = get_velocity(current_positions, TrianT, node_positions[:,:, 0], node_velocities[:,:, 0], centroid_indices)

    fine_time = fine_time[:-1]

    # Main time-stepping loop
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

            kdtree = KDTree(next_centroids)
            current_positions = np.array([x_current, y_current, z_current]).T
            new_positions, centroid_indices = particle_projection(TrianT, kdtree, current_positions, next_node_positions)
            velocities = get_velocity(new_positions, TrianT, next_node_positions, next_node_velocities, centroid_indices)
            x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

        x_traj[:, t_index + 1] = x_current
        y_traj[:, t_index + 1] = y_current
        z_traj[:, t_index + 1] = z_current



    return x_traj, y_traj, z_traj


def NEW_RK4_forward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):
                                       
    time_length = len(time_steps)
    num_particles = len(particle_positions[:, 0])

    fine_time = np.arange(0, time_length - 1 + dt, dt)
 
    fine_time_length = len(fine_time)

    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial kdtree and project initial particles onto the surface
    kdtree = KDTree(centroids[:, :, 0])
    new_positions, centroid_indices = particle_projection(TrianT, kdtree, particle_positions, node_positions[:, :, 0])

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

        t_next = t_fraction + dt

        next_node_positions = interpolate(node_positions[:, :, floor_t], node_positions[:, :, ceiling_t], t_next)
        next_node_velocities = interpolate(node_velocities[:, :, floor_t], node_velocities[:, :, ceiling_t], t_next)
        next_centroids = interpolate(centroids[:, :, floor_t], centroids[:, :, ceiling_t], t_next)

        def velocity_at_position(x, y, z):
            pos = np.array([x, y, z]).T
            velocities = get_velocity(pos, TrianT, next_node_positions, next_node_velocities, centroid_indices)
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
        kdtree = KDTree(next_centroids)
        next_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT, kdtree, next_positions, next_node_positions)


        # Update stored trajectories
        x_traj[:, t_index + 1] = new_positions[:, 0]
        y_traj[:, t_index + 1] = new_positions[:, 1]
        z_traj[:, t_index + 1] = new_positions[:, 2]
        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj





def NEWbackward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):
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
    # Reverse time indexing
    node_positions = node_positions[:,:, ::-1]
    node_velocities = node_velocities[:,:, ::-1]
    centroids = centroids[:,:, ::-1]

    

    time_length = len(time_steps)
    num_particles = len(particle_positions[:, 0])

    
    fine_time = np.arange(0, time_length - 1 + dt, dt)
    dt = -dt

    fine_time_length = len(fine_time)
 
    # Initialize trajectories
    x_traj = np.zeros((num_particles, fine_time_length ))
    y_traj = np.zeros((num_particles, fine_time_length ))
    z_traj = np.zeros((num_particles, fine_time_length ))

    # Build the initial KDTree for the last time step
    kdtree = KDTree(centroids[:, :, 0])

    # Project initial particle positions onto the surface
    new_positions, centroid_indices = particle_projection(TrianT, kdtree, particle_positions, node_positions[:, :, 0])

    # Set initial trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Running positions t_index = 0
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    fine_time = fine_time[:-1]


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
        kdtree = KDTree(next_centroids)
        current_positions = np.array([x_current, y_current, z_current]).T

        if t_index % 2 == 0:
            velocities = get_velocity(current_positions, TrianT, current_node_positions, current_node_velocities, centroid_indices)

            # Forward Euler update
            x_current += velocities[:, 0] * dt
            y_current += velocities[:, 1] * dt
            z_current += velocities[:, 2] * dt

        else:  # Odd time steps - Backward Euler
                    # Initial guess for Backward Euler
                    velocities = get_velocity(current_positions, TrianT, current_node_positions, current_node_velocities, centroid_indices)
                    x_guess = x_current + velocities[:, 0] * dt
                    y_guess = y_current + velocities[:, 1] * dt
                    z_guess = z_current + velocities[:, 2] * dt

                    # Iteratively correct the position using the backward scheme
                    for _ in range(3):  
                        guess_positions = np.array([x_guess, y_guess, z_guess]).T
                        velocities = get_velocity(guess_positions, TrianT, next_node_positions, next_node_velocities, centroid_indices)

                        x_guess = x_current + velocities[:, 0] * dt
                        y_guess = y_current + velocities[:, 1] * dt
                        z_guess = z_current + velocities[:, 2] * dt

                    # Update positions after convergence
                    x_current = x_guess
                    y_current = y_guess
                    z_current = z_guess

        # Project updated positions onto the surface
        updated_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT, kdtree, updated_positions, current_node_positions)

        # Update trajectories
        x_traj[:, t_index+1] = new_positions[:, 0]
        y_traj[:, t_index+1] = new_positions[:, 1]
        z_traj[:, t_index+1] = new_positions[:, 2]


    return x_traj, y_traj, z_traj


def NEW_fast_backward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):


    # Reverse time indexing
    node_positions = node_positions[:,:, ::-1]
    node_velocities = node_velocities[:,:, ::-1]
    centroids = centroids[:,:, ::-1]

    

    time_length = len(time_steps)
    num_particles = len(particle_positions[:, 0])

    
    fine_time = np.arange(0, time_length - 1 + dt, dt)
    dt = -dt

    fine_time_length = len(fine_time)

    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))


    # Build the initial kdtree
    kdtree = KDTree(centroids[:,:, 0])


    # Make sure the initial particles are on the surface
    new_positions, centroid_indices = particle_projection(TrianT, kdtree, particle_positions, node_positions[:,:, 0]) 

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
    velocities = get_velocity(current_positions, TrianT, node_positions[:,:, 0], node_velocities[:,:, 0], centroid_indices)


    fine_time = fine_time[:-1] # remove the last index to avoid out of bounds indexing

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

            kdtree = KDTree(next_centroids)
            current_positions = np.array([x_current, y_current, z_current]).T
            new_positions, centroid_indices = particle_projection(TrianT, kdtree, current_positions, next_node_positions)
            velocities = get_velocity(new_positions, TrianT, next_node_positions, next_node_velocities, centroid_indices)
            x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]


        x_traj[:, t_index + 1] = x_current
        y_traj[:, t_index + 1] = y_current
        z_traj[:, t_index + 1] = z_current



    return x_traj, y_traj, z_traj


def NEW_RK4_backward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):


    # Reverse time indexing
    node_positions = node_positions[:,:, ::-1]
    node_velocities = node_velocities[:,:, ::-1]
    centroids = centroids[:,:, ::-1]

    time_length = len(time_steps)
    num_particles = len(particle_positions[:, 0])

    fine_time = np.arange(0, time_length - 1 + dt, dt)
    dt = -dt
 
    fine_time_length = len(fine_time)

    x_traj = np.zeros((num_particles, fine_time_length))
    y_traj = np.zeros((num_particles, fine_time_length))
    z_traj = np.zeros((num_particles, fine_time_length))

    # Build the initial kdtree and project initial particles onto the surface
    kdtree = KDTree(centroids[:, :, 0])
    new_positions, centroid_indices = particle_projection(TrianT, kdtree, particle_positions, node_positions[:, :, 0])

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
            velocities = get_velocity(pos, TrianT, next_node_positions, next_node_velocities, centroid_indices)
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
        kdtree = KDTree(next_centroids)
        next_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT, kdtree, next_positions, next_node_positions)


        # Update stored trajectories
        x_traj[:, t_index + 1] = new_positions[:, 0]
        y_traj[:, t_index + 1] = new_positions[:, 1]
        z_traj[:, t_index + 1] = new_positions[:, 2]
        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj




