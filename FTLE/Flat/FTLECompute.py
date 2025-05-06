import numpy as np
import math




def FTLE_2d_compute(x_initial, y_initial, x_final, y_final, initial_time, final_time):
    """
    Compute FTLE field on a uniform 2D grid using finite differences.

    Parameters:
        x_initial, y_initial: 2D arrays of initial grid positions.
        x_final, y_final: 2D arrays of advected grid positions.
        time (float): Total advection time.

    Returns:
        FTLE (2D ndarray): Finite-time Lyapunov exponent values.
    """
    nx, ny = x_initial.shape
    FTLE = np.full((nx, ny), np.nan)
    isotropy = np.full((nx,ny),np.nan)
    F = np.zeros((2, 2))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # Skip NaNs in initial positions
            if math.isnan(x_initial[i, j]) or math.isnan(y_initial[i, j]):
                continue

            # Local grid spacing
            dx = x_initial[i + 1, j] - x_initial[i - 1, j]
            dy = y_initial[i, j + 1] - y_initial[i, j - 1]

            print(dx)
            print(dy)

            if dx == 0 or dy == 0:
                continue

            # Compute finite difference deformation gradient ∂Xf/∂X0
            F[0, 0] = (x_final[i + 1, j] - x_final[i - 1, j]) / (2 * dx)
            F[0, 1] = (x_final[i, j + 1] - x_final[i, j - 1]) / (2 * dy)
            F[1, 0] = (y_final[i + 1, j] - y_final[i - 1, j]) / (2 * dx)
            F[1, 1] = (y_final[i, j + 1] - y_final[i, j - 1]) / (2 * dy)

            # Cauchy-Green strain tensor: C = Fᵀ F
            C = F.T @ F

            if np.isnan(C).any() or np.isinf(C).any():
                continue
            print(C)
            # Maximum eigenvalue of C
            eigenvalues = np.linalg.eigvalsh(C)
            max_eigenvalue = np.max(eigenvalues)

            if max_eigenvalue <= 0:
                continue
            time = np.abs(initial_time - final_time)
            FTLE[i, j] = (1 / (2 * time)) * np.log(np.sqrt(max_eigenvalue))
            isotropy[i,j] = (1 / (2 * time)) * np.log(np.linalg.det(C))

    return FTLE.flatten(), isotropy.flatten()



def FTLE_3d_compute(x_initial, y_initial, z_initial, x_final, y_final, z_final, initial_time, final_time):
    nx, ny, nz = x_initial.shape
    FTLE = np.full((nx, ny, nz), np.nan)
    isotropy[i,j] = np.full((nx,ny,nz), np.nan)
    F_right = np.zeros((3, 3))

    for z_index in range(1, nz - 1):
        for x_index in range(1, nx - 1):
            for y_index in range(1, ny - 1):

                if math.isnan(x_initial[x_index, y_index, z_index]) or math.isnan(y_initial[x_index, y_index, z_index]):
                    continue

                dx = x_initial[x_index + 1, y_index, z_index] - x_initial[x_index - 1, y_index, z_index]
                dy = y_initial[x_index, y_index + 1, z_index] - y_initial[x_index, y_index - 1, z_index]
                dz = z_initial[x_index, y_index, z_index + 1] - z_initial[x_index, y_index, z_index - 1]

                if dx == 0 or dy == 0 or dz == 0:
                    continue

                # ∂Xf/∂X0 (deformation gradient matrix, F_right)
                F_right[0, 0] = (x_final[x_index + 1, y_index, z_index] - x_final[x_index - 1, y_index, z_index]) / (2 * dx)
                F_right[0, 1] = (x_final[x_index, y_index + 1, z_index] - x_final[x_index, y_index - 1, z_index]) / (2 * dy)
                F_right[0, 2] = (x_final[x_index, y_index, z_index + 1] - x_final[x_index, y_index, z_index - 1]) / (2 * dz)

                F_right[1, 0] = (y_final[x_index + index_shift, y_index, z_index] - y_final[x_index - 1, y_index, z_index]) / (2 * dx)
                F_right[1, 1] = (y_final[x_index, y_index + index_shift, z_index] - y_final[x_index, y_index - 1, z_index]) / (2 * dy)
                F_right[1, 2] = (y_final[x_index, y_index, z_index + index_shift] - y_final[x_index, y_index, z_index - 1]) / (2 * dz)

                F_right[2, 0] = (z_final[x_index + 1, y_index, z_index] - z_final[x_index - 1, y_index, z_index]) / (2 * dx)
                F_right[2, 1] = (z_final[x_index, y_index + 1, z_index] - z_final[x_index, y_index - 1, z_index]) / (2 * dy)
                F_right[2, 2] = (z_final[x_index, y_index, z_index + 1] - z_final[x_index, y_index, z_index - 1]) / (2 * dz)

                # Cauchy-Green strain tensor
                C = F_right.T @ F_right

                if np.isnan(C).any() or np.isinf(C).any():
                    continue

                eigenvalues = np.linalg.eigh(C)[0]
                max_eigen = np.max(eigenvalues)

                if max_eigen <= 0:
                    continue

                time = np.abs(initial_time - final_time)
                FTLE[x_index, y_index, z_index] = (1 / (2 * time)) * np.log(np.sqrt(max_eigen))
                isotropy[x_index, y_index, z_index] = (1 / (2 * time)) * np.log(np.linalg.det(C))

    return FTLE.flatten(), isotropy.flatten()

