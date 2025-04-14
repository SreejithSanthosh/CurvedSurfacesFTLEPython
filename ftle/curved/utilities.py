
import numpy as np
import pyvista as pv
from time import  time as timer 
from scipy.spatial import cKDTree
from numba import njit
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
from itertools import combinations
import h5py
from typing import List








def plot_FTLE_mesh(node_cons, node_positions, initial_time, final_time, ftle, direction):
    """
    Plots the FTLE field over a mesh using PyVista, compatible with staggered arrays.

    Parameters:
        node_cons (List[np.ndarray]): List of triangle connectivity arrays for each time step.
        node_positions (List[np.ndarray]): List of node position arrays for each time step.
        initial_time (int): Starting time index for the FTLE computation.
        final_time (int): Ending time index for the FTLE computation.
        ftle (np.ndarray): FTLE values to visualize.
        direction (str): Direction of advection ("forward" or "backward").
    """
    scalar_bar_args = {
        "vertical": True,
        "title_font_size": 12,
        "label_font_size": 10,
        "n_labels": 5,
        "position_x": 0.85,
        "position_y": 0.1,
        "width": 0.1,
        "height": 0.7
    }

    # Get mesh data for initial_time
    verts = node_positions[initial_time]
    conns = node_cons[initial_time]

    # Build PyVista face array: [3, a, b, c] for each triangle
    faces = np.hstack([np.full((conns.shape[0], 1), 3), conns]).astype(np.int32).flatten()

    # Create surface mesh
    surf = pv.PolyData(verts, faces)
    surf["FTLE"] = ftle

    surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)

    # Plotting config
    plot_kwargs = dict(smooth_shading=True, show_edges=False, ambient=0.5, diffuse=0.6, specular=0.3)
    pl = pv.Plotter(off_screen=False, window_size=(1920, 1080))
    pl.add_mesh(surf, scalars='FTLE', cmap='jet', interpolate_before_map=True,
                scalar_bar_args=scalar_bar_args, **plot_kwargs)
    pl.add_title(f'{direction.title()} FTLE: Time {initial_time} to {final_time}')
    pl.show()

    return 0
