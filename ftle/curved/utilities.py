import numpy as np
import pyvista as pv


def camera_position_to_angles(position, focal_point):
    """
    Converts camera position relative to focal point into (azimuth, elevation) in degrees.

    Parameters:
        position: (3,) array-like, camera position
        focal_point: (3,) array-like, camera focal point

    Returns:
        azimuth, elevation (degrees)
    """
    vec = np.array(position) - np.array(focal_point)
    r = np.linalg.norm(vec)

    azimuth = np.degrees(np.arctan2(vec[1], vec[0]))
    elevation = np.degrees(np.arcsin(vec[2] / r))

    return azimuth, elevation


def plot_FTLE_mesh(
    node_cons,
    node_positions,
    initial_time,
    final_time,
    ftle,
    direction,
    save_path=None,    # str or None → if provided, save figure
    view_angle=None    # (azimuth, elevation) or None
):
    """
    Plots the FTLE field over a mesh using PyVista, compatible with staggered arrays.

    Parameters:
        node_cons (List[np.ndarray]): Face connections for each timestep
        node_positions (List[np.ndarray]): Node positions for each timestep
        initial_time (int): Start time
        final_time (int): End time
        ftle (np.ndarray): FTLE values for mesh
        direction (str): 'forward' or 'backward'
        save_path (str, optional): Save path for figure
        view_angle (tuple, optional): (azimuth, elevation) for camera
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

    pl = pv.Plotter(off_screen=save_path is not None, window_size=(1920, 1080))

    pl.add_mesh(surf, scalars='FTLE', cmap='jet', interpolate_before_map=True,
                scalar_bar_args=scalar_bar_args, smooth_shading=True, show_edges=False,
                ambient=0.5, diffuse=0.6, specular=0.3)

    pl.add_title(f'{direction.title()} FTLE: Time {initial_time} to {final_time}')

    # Handle optional user view angle
    if view_angle is not None:
        az, el = view_angle
        pl.camera.azimuth = az
        pl.camera.elevation = el

    if save_path is not None:
        pl.show(screenshot=save_path)
        print(f"Figure saved to {save_path}")
    else:
        # Add live view info text
        pl.add_text("Press 'c' to print current view angle", position='upper_left',
                    font_size=10, color='white', name='cam_text')

        # Add key press event: Press 'c' to print azimuth/elevation
        def report_camera_position():
            az, el = camera_position_to_angles(pl.camera.position, pl.camera.focal_point)
            print(f'Current View → Azimuth: {az:.2f}°, Elevation: {el:.2f}°')

        pl.add_key_event('c', report_camera_position)
        pl.show()

    return 0
