import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata



@njit
def interpolate(floor_data, ceiling_data, t_fraction):
    return t_fraction*ceiling_data + (1-t_fraction)*floor_data



def plot_FTLE_2d(
    particles,
    ftle,
    isotropy,
    back_ftle,
    back_isotropy,
    initial_time, 
    final_time,
    resolution=200,
    method='linear',
    save_plot_path=None
):
    """
    Interpolates and plots 2D scalar fields (FTLE/isotropy, forward/backward) in 2x2 subplots.

    Parameters:
        particles (ndarray): shape (N, 2), particle positions in 2D.
        ftle, isotropy, back_ftle, back_isotropy (ndarray): scalar values at particles.
        resolution (int): grid resolution for interpolation.
        method (str): interpolation method: 'linear', 'cubic', or 'nearest'.
        save_path (str or None): if not None, path to save the plot as an image.
    """
    
    x, y = particles[:, 0], particles[:, 1]

    xi = np.linspace(x.min(), x.max(), int(resolution))
    yi = np.linspace(y.min(), y.max(), int(resolution))
    X, Y = np.meshgrid(xi, yi)

    # Interpolate each field
    fields = [
        (f"Forward FTLE, Time: {initial_time}-{final_time}", ftle),
        (f"Forward Isotropy, Time: {initial_time}-{final_time}", isotropy),
        (f"Backward FTLE, Time: {final_time}-{initial_time}", back_ftle),
        (f"Backward Isotropy, Time: {final_time}-{initial_time}", back_isotropy)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    print(len(ftle))
    print(ftle.shape)
    print(len(particle_positions))
    print(particle_positions.shape)
    for ax, (title, field) in zip(axes.flat, fields):
        Z = griddata(particles, field, (X, Y), method=method)
        pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap='plasma')
        fig.colorbar(pcm, ax=ax, label=title)
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal')

    plt.tight_layout()

    if save_plot_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()
    return None

def plot_FTLE_3d(coords, ftle, isotropy, back_ftle, back_isotropy, 
                             grid_resolution=50, save_plot_path=None):
    """
    Interpolate four scalar fields (ftle, isotropy, back_ftle, back_isotropy) defined on scattered 3D points
    onto a dense 3D grid, and visualize them in a 2x2 subplot (3D scatter volume for each field).
    
    Parameters:
        coords : ndarray of shape (N, 3) with XYZ positions of points.
        ftle, isotropy, back_ftle, back_isotropy : arrays of length N with scalar field values at those points.
        grid_resolution : int or tuple(int,int,int), number of grid points along each axis (default 50).
        save_path : string or Path, optional path to save the figure. If None, the plot is shown interactively.
    """
    # Ensure coords is an array
    coords = np.array(coords)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # Define grid resolution for each axis
    if isinstance(grid_resolution, int):
        nx = ny = nz = grid_resolution
    else:
        nx, ny, nz = grid_resolution
    
    # Create a regular grid covering the data domain
    x_lin = np.linspace(x.min(), x.max(), int(nx))
    y_lin = np.linspace(y.min(), y.max(), int(ny))
    z_lin = np.linspace(z.min(), z.max(), int(nz))
    Xg, Yg, Zg = np.meshgrid(x_lin, y_lin, z_lin, indexing='xy')  # 3D grid coordinates
    
    # Interpolate each scalar field onto the grid (linear interpolation)
    ftle_grid = griddata((x, y, z), ftle, (Xg, Yg, Zg), method='linear')
    iso_grid  = griddata((x, y, z), isotropy, (Xg, Yg, Zg), method='linear')
    bftle_grid = griddata((x, y, z), back_ftle, (Xg, Yg, Zg), method='linear')
    biso_grid  = griddata((x, y, z), back_isotropy, (Xg, Yg, Zg), method='linear')
    
    # Fill any NaN values by nearest-neighbor interpolation to cover full domain
    # (This step ensures no gaps if linear interpolation misses corners or edges)
    if np.isnan(ftle_grid).any():
        ftle_grid_near = griddata((x, y, z), ftle, (Xg, Yg, Zg), method='nearest')
        ftle_grid = np.where(np.isnan(ftle_grid), ftle_grid_near, ftle_grid)
    if np.isnan(iso_grid).any():
        iso_grid_near = griddata((x, y, z), isotropy, (Xg, Yg, Zg), method='nearest')
        iso_grid = np.where(np.isnan(iso_grid), iso_grid_near, iso_grid)
    if np.isnan(bftle_grid).any():
        bftle_grid_near = griddata((x, y, z), back_ftle, (Xg, Yg, Zg), method='nearest')
        bftle_grid = np.where(np.isnan(bftle_grid), bftle_grid_near, bftle_grid)
    if np.isnan(biso_grid).any():
        biso_grid_near = griddata((x, y, z), back_isotropy, (Xg, Yg, Zg), method='nearest')
        biso_grid = np.where(np.isnan(biso_grid), biso_grid_near, biso_grid)
    
    # Flatten the grid points and values for plotting
    Xf = Xg.flatten();  Yf = Yg.flatten();  Zf = Zg.flatten()
    ftle_vals  = ftle_grid.flatten()
    iso_vals   = iso_grid.flatten()
    bftle_vals = bftle_grid.flatten()
    biso_vals  = biso_grid.flatten()
    
    # Set up 2x2 subplots for 3D scatter plots
    fig = plt.figure(figsize=(10, 10))
    axes = []
    axes.append(fig.add_subplot(2, 2, 1, projection='3d'))
    axes.append(fig.add_subplot(2, 2, 2, projection='3d'))
    axes.append(fig.add_subplot(2, 2, 3, projection='3d'))
    axes.append(fig.add_subplot(2, 2, 4, projection='3d'))

    # Plot each scalar field as a scatter of colored points
    axes[0].scatter(Xf, Yf, Zf, c=ftle_vals, cmap='plasma', marker='.', depthshade=False)
    axes[0].set_title(f"Forward FTLE, Time: {initial_time}-{final_time}")
    axes[1].scatter(Xf, Yf, Zf, c=iso_vals, cmap='plasma', marker='.', depthshade=False)
    axes[1].set_title(f"Forward Isotropy, Time: {initial_time}-{final_time}")
    axes[2].scatter(Xf, Yf, Zf, c=bftle_vals, cmap='plasma', marker='.', depthshade=False)
    axes[2].set_title(f"Backward FTLE, Time: {final_time}-{initial_time}")
    axes[3].scatter(Xf, Yf, Zf, c=biso_vals, cmap='plasma', marker='.', depthshade=False)
    axes[3].set_title(f"Backward Isotropy, Time: {final_time}-{initial_time}")
    
    # Improve visualization: equal aspect ratio and no axis clutter
    for ax in axes:
        # Set equal aspect ratio for x, y, z
        ax.set_box_aspect((1, 1, 1))
        # Turn off the axes panes and ticks for clarity
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        # Optionally, we could also set ax.set_facecolor('black') for a dark background
    plt.tight_layout()
    
    # Save or show the figure
    if save_plot_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    plt.show()

    return None
