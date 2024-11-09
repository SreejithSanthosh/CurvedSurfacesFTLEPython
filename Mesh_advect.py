
from numba import njit
import numpy as np
from scipy.spatial import KDTree



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

    for i in range(len(particle_positions[:,0])):
        
        query_point = particle_positions[i, :]
        
        # Get nearest centriod data
        k_closest_centroids = 10
        _, indices = kdtree.query(query_point, k_closest_centroids)
        
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




def forward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):

    time_length = len(time_steps)
    number_particles = len(particle_positions[:, 0])

    x_traj = np.zeros((number_particles, time_length))
    y_traj = np.zeros((number_particles, time_length))
    z_traj = np.zeros((number_particles, time_length))

    # Build the initial kdtree and project initial particles onto the surface
    kdtree = KDTree(centroids[:, :, 0])
    new_positions, centroid_indices = particle_projection(TrianT[:, :, 0], kdtree, particle_positions, node_positions[:, :, 0])
    new_positions, centroid_indices = particle_projection(TrianT[:, :, 0], kdtree, new_positions, node_positions[:, :, 0])

    # Initialize trajectories and running positions
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]
    x_current, y_current, z_current = x_traj[:, 0], y_traj[:, 0], z_traj[:, 0]

    for t in range(1, time_length):
        
        def velocity_at_position(x, y, z):
            pos = np.array([x, y, z]).T
            velocities = get_velocity(pos, TrianT[:, :, t], node_positions[:, :, t], node_velocities[:, :, t], centroid_indices)
            return velocities[:, 0], velocities[:, 1], velocities[:, 2]

        #RK4 
        k1_x, k1_y, k1_z = velocity_at_position(x_current, y_current, z_current)
        k2_x, k2_y, k2_z = velocity_at_position(x_current + 0.5 * dt * k1_x, y_current + 0.5 * dt * k1_y, z_current + 0.5 * dt * k1_z)
        k3_x, k3_y, k3_z = velocity_at_position(x_current + 0.5 * dt * k2_x, y_current + 0.5 * dt * k2_y, z_current + 0.5 * dt * k2_z)
        k4_x, k4_y, k4_z = velocity_at_position(x_current + dt * k3_x, y_current + dt * k3_y, z_current + dt * k3_z)

        x_current += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_current += (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        z_current += (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

        # Project new positions back onto the surface
        kdtree = KDTree(centroids[:, :, t])
        current_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT[:, :, t], kdtree, current_positions, node_positions[:, :, t])

        
        x_traj[:, t] = new_positions[:, 0]
        y_traj[:, t] = new_positions[:, 1]
        z_traj[:, t] = new_positions[:, 2]
        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj


def backward_particle_advection(TrianT, centroids, particle_positions, node_positions, node_velocities, time_steps, dt):
    time_length = len(time_steps) - 1
    number_particles = len(particle_positions[:, 0])

    x_traj = np.zeros((number_particles, time_length))
    y_traj = np.zeros((number_particles, time_length))
    z_traj = np.zeros((number_particles, time_length))

    # Build the initial KDTree
    kdtree = KDTree(centroids[:, :, -1])

    # Project initial particles onto the surface 
    new_positions, centroid_indices = particle_projection(TrianT[:, :, -1], kdtree, particle_positions, node_positions[:, :, -1])
    

    # Initialize trajectories
    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]

    # Set initial positions
    x_current = x_traj[:, 0]
    y_current = y_traj[:, 0]
    z_current = z_traj[:, 0]

    time_length -= 1  # Update time length for indexing
    for t in range(time_length, -1, -1):

        def velocity_at_position(x, y, z):
            pos = np.array([x, y, z]).T
            velocities = get_velocity(pos, TrianT[:, :, t], node_positions[:, :, t], node_velocities[:, :, t], centroid_indices)
            return velocities[:, 0], velocities[:, 1], velocities[:, 2]

        #RK4
        k1_x, k1_y, k1_z = velocity_at_position(x_current, y_current, z_current)
        k2_x, k2_y, k2_z = velocity_at_position(x_current + 0.5 * dt * k1_x, y_current + 0.5 * dt * k1_y, z_current + 0.5 * dt * k1_z)
        k3_x, k3_y, k3_z = velocity_at_position(x_current + 0.5 * dt * k2_x, y_current + 0.5 * dt * k2_y, z_current + 0.5 * dt * k2_z)
        k4_x, k4_y, k4_z = velocity_at_position(x_current + dt * k3_x, y_current + dt * k3_y, z_current + dt * k3_z)

        
        x_current -= (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_current -= (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        z_current -= (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)


        # Update KDTree and project particles onto the surface
        kdtree = KDTree(centroids[:, :, t])
        current_positions = np.array([x_current, y_current, z_current]).T
        new_positions, centroid_indices = particle_projection(TrianT[:, :, t], kdtree, current_positions, node_positions[:, :, t])

        # Update trajectories
        x_traj[:, time_length - t] = new_positions[:, 0]
        y_traj[:, time_length - t] = new_positions[:, 1]
        z_traj[:, time_length - t] = new_positions[:, 2]

        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj
