


from scipy.spatial import KDTree
import numpy as np
from numba import njit
from itertools import combinations



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