


import matplotlib.pyplot as plt
import numpy as np





def interpolate(floor_data, ceiling_data, t_fraction):
    return t_fraction*ceiling_data + (1-t_fraction)*floor_data



def plot_ftle_2d(particles, ftle):
    plt.figure(figsize=(6, 6))


    # Scatter plot of FTLE values with color mapping
    sc = plt.scatter(particles[:, 0], particles[:, 1], c=ftle, cmap='plasma', s=10)
    plt.colorbar(sc, label="FTLE Value")

    plt.title("FTLE Field at Initial Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

    return None

def plot_ftle_3d(particles, ftle, tol=0.01):
    """
    particles: (N, 3) array of particle positions in 3D
    ftle: (N,) array of FTLE values
    tol: float, tolerance for selecting slices near specific y values
    """
    y_values = particles[:, 1]
    y_slices = [np.min(y_values), np.median(y_values), np.max(y_values)]

    fig = plt.figure(figsize=(16, 10))

    # Full 3D plot (left side)
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax3d.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                      c=ftle, cmap='plasma', s=10)
    ax3d.set_title("Full 3D FTLE Field")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    cbar = fig.colorbar(sc, ax=ax3d, shrink=0.6)
    cbar.set_label("FTLE Value")

    # Add 3 XZ slice subplots (right side)
    for i, y_val in enumerate(y_slices):
        ax = fig.add_subplot(2, 2, i + 2)  # Plots go into 2nd, 3rd, and 4th subplot spots
        mask = np.abs(particles[:, 1] - y_val) < tol
        if np.any(mask):
            slice_particles = particles[mask]
            slice_ftle = ftle[mask]
            sc2 = ax.scatter(slice_particles[:, 0], slice_particles[:, 2],
                             c=slice_ftle, cmap='plasma', s=10)
            ax.set_title(f"XZ Slice at y ≈ {y_val:.3f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
        else:
            ax.set_title(f"No points found near y = {y_val:.3f}")

    plt.tight_layout()
    plt.show()

    return None
