import pyvista as pv
import numpy as np






def plot_FTLE_mesh(
    node_cons,
    node_positions,
    initial_time,
    final_time,
    ftle,
    direction,
    save_path=None,        # Optional: str or None
    view_angle=None        # Optional: tuple (azimuth, elevation)
):
    """
    Plots the FTLE field over a mesh using PyVista, compatible with staggered arrays.

    Parameters:
        node_cons (List[np.ndarray]): List of triangle connectivity arrays for each time step.
        node_positions (List[np.ndarray]): List of node position arrays for each time step.
        initial_time (int): Starting time index for the FTLE computation.
        final_time (int): Ending time index for the FTLE computation.
        ftle (np.ndarray): FTLE values to visualize.
        direction (str): Direction of advection ("forward" or "backward").
        save_path (str, optional): If provided, saves the figure to this path (as .png)
        view_angle (tuple, optional): Tuple (azimuth, elevation) for camera angle in degrees.
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

    verts = node_positions[initial_time]
    conns = node_cons[initial_time]
    faces = np.hstack([np.full((conns.shape[0], 1), 3), conns]).astype(np.int32).flatten()

    surf = pv.PolyData(verts, faces)
    surf["FTLE"] = ftle
    surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)

    plot_kwargs = dict(smooth_shading=True, show_edges=False, ambient=0.5, diffuse=0.6, specular=0.3)
    pl = pv.Plotter(off_screen=save_path is not None, window_size=(1920, 1080))

    pl.add_mesh(surf, scalars='FTLE', cmap='jet', interpolate_before_map=True,
                scalar_bar_args=scalar_bar_args, **plot_kwargs)
    pl.add_title(f'{direction.title()} FTLE: Time {initial_time} to {final_time}')

    # Handle custom view angle
    if view_angle is not None:
        az, el = view_angle
        pl.camera.azimuth(az)
        pl.camera.elevation(el)

    # Add camera angle indicator (only if not saving directly)
    if save_path is None:
        azimuth = pl.camera.azimuth()
        elevation = pl.camera.elevation()
        pl.add_text(f"Azimuth: {azimuth:.1f}°, Elevation: {elevation:.1f}°",
                    position='upper_left',
                    font_size=10,
                    color='white')

    if save_path is not None:
        pl.show(screenshot=save_path)
        print(f"Figure saved to {save_path}")
    else:
        pl.show()

    return 0



