import numpy as np
from typing import List





def compute_centroids_staggered(node_connections: List[np.ndarray], node_positions: List[np.ndarray]):
    centroids = []


    for con, pos in zip(node_connections, node_positions):
        
        A = pos[con[:, 0]]
        B = pos[con[:, 1]]
        C = pos[con[:, 2]]
        centroids.append((A + B + C) / 3.0)
    return centroids

def RK4_particle_advection(
    node_connections: List[np.ndarray],
    node_positions: List[np.ndarray],
    node_velocities: List[np.ndarray],
    particle_positions: np.ndarray,
    initial_time: int,
    final_time: int
):
    centroids = compute_centroids_staggered(node_connections, node_positions)

    num_particles = particle_positions.shape[0]
    time_indices = list(range(initial_time, final_time ))
    steps = len(time_indices) + 1

    x_traj = np.zeros((num_particles, steps))
    y_traj = np.zeros((num_particles, steps))
    z_traj = np.zeros((num_particles, steps))

    from scipy.spatial import cKDTree

    def particle_projection(node_connections, kdtree, particle_positions, node_positions):
        def compute_barycentric(A, B, C, P):
            v0, v1, v2 = B - A, C - A, P - A
            d00 = np.einsum('ij,ij->i', v0, v0)
            d01 = np.einsum('ij,ij->i', v0, v1)
            d11 = np.einsum('ij,ij->i', v1, v1)
            d20 = np.einsum('ij,ij->i', v2, v0)
            d21 = np.einsum('ij,ij->i', v2, v1)
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            return np.stack([u, v, w], axis=1)

        def check_in_centroid(bary_coords):
            return np.all((bary_coords >= -0.05) & (bary_coords <= 1.05), axis=1)

        def project(A, B, C, P):
            normal = np.cross(B - A, C - A)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            vector_to_plane = A - P
            distance = np.einsum('ij,ij->i', vector_to_plane, normal)
            return P + distance[:, None] * normal

        _, indices = kdtree.query(particle_positions, k=12)
        first_test_positions = None
        final_indices = np.full(particle_positions.shape[0], -1, dtype=int)

        for attempt in range(12):
            nearest_faces = node_connections[indices[:, attempt]]
            A = node_positions[nearest_faces[:, 0]]
            B = node_positions[nearest_faces[:, 1]]
            C = node_positions[nearest_faces[:, 2]]

            projected_positions = project(A, B, C, particle_positions)
            bary_coords = compute_barycentric(A, B, C, projected_positions)
            inside = check_in_centroid(bary_coords)

            if first_test_positions is None:
                first_test_positions = projected_positions

            if np.any(inside):
                mask = inside & (final_indices == -1)
                final_indices[mask] = indices[mask, attempt]

            if np.all(final_indices != -1):
                break

        failed_mask = final_indices == -1
        final_indices[failed_mask] = indices[failed_mask, 0]
        projected_positions[failed_mask] = first_test_positions[failed_mask]

        return projected_positions, final_indices

    def get_velocity(particle_positions, faces, node_positions, node_velocities, face_indices):
        particle_faces = faces[face_indices, :]
        A = node_positions[particle_faces[:, 0], :]
        B = node_positions[particle_faces[:, 1], :]
        C = node_positions[particle_faces[:, 2], :]

        v0 = node_velocities[particle_faces[:, 0], :]
        v1 = node_velocities[particle_faces[:, 1], :]
        v2 = node_velocities[particle_faces[:, 2], :]

        v2_1 = B - A
        v2_2 = C - A
        v2_particle = particle_positions - A

        d00 = np.einsum('ij,ij->i', v2_1, v2_1)
        d01 = np.einsum('ij,ij->i', v2_1, v2_2)
        d11 = np.einsum('ij,ij->i', v2_2, v2_2)
        d20 = np.einsum('ij,ij->i', v2_particle, v2_1)
        d21 = np.einsum('ij,ij->i', v2_particle, v2_2)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        interpolated_velocities = u[:, np.newaxis] * v0 + v[:, np.newaxis] * v1 + w[:, np.newaxis] * v2
        return interpolated_velocities

    kdtree = cKDTree(centroids[initial_time])
    new_positions, face_indices = particle_projection(
        node_connections[initial_time], kdtree, particle_positions, node_positions[initial_time]
    )

    x_traj[:, 0] = new_positions[:, 0]
    y_traj[:, 0] = new_positions[:, 1]
    z_traj[:, 0] = new_positions[:, 2]
    x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    for i, t_index in enumerate(time_indices):

        
        pos = np.stack([x_current, y_current, z_current], axis=1)
        k1 = get_velocity(pos, node_connections[t_index], node_positions[t_index], node_velocities[t_index], face_indices)

        pos = np.stack([x_current + 0.5 * k1[:, 0], y_current + 0.5 * k1[:, 1], z_current + 0.5 * k1[:, 2]], axis=1)
        k2 = get_velocity(pos, node_connections[t_index], node_positions[t_index], node_velocities[t_index], face_indices)

        pos = np.stack([x_current + 0.5 * k2[:, 0], y_current + 0.5 * k2[:, 1], z_current + 0.5 * k2[:, 2]], axis=1)
        k3 = get_velocity(pos, node_connections[t_index], node_positions[t_index], node_velocities[t_index], face_indices)

        pos = np.stack([x_current + 0.5 * k3[:, 0], y_current + 0.5 * k3[:, 1], z_current + 0.5 * k3[:, 2]], axis=1)
        k4 = get_velocity(pos, node_connections[t_index], node_positions[t_index], node_velocities[t_index], face_indices)

        x_current += (1.0 / 6.0) * (k1[:, 0] + 2 * k2[:, 0] + 2 * k3[:, 0] + k4[:, 0])
        y_current += (1.0 / 6.0) * (k1[:, 1] + 2 * k2[:, 1] + 2 * k3[:, 1] + k4[:, 1])
        z_current += (1.0 / 6.0) * (k1[:, 2] + 2 * k2[:, 2] + 2 * k3[:, 2] + k4[:, 2])

        kdtree = cKDTree(centroids[t_index + 1])
        next_positions = np.stack([x_current, y_current, z_current], axis=1)
        new_positions, face_indices = particle_projection(
            node_connections[t_index + 1], kdtree, next_positions, node_positions[t_index + 1]
        )

        x_traj[:, i + 1] = new_positions[:, 0]
        y_traj[:, i + 1] = new_positions[:, 1]
        z_traj[:, i + 1] = new_positions[:, 2]

        x_current, y_current, z_current = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]

    return x_traj, y_traj, z_traj, centroids

