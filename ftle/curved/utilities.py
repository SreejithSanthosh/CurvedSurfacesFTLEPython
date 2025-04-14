import numpy as np
import pyvista as pv


def plot_FTLE_mesh(
    node_cons,
    node_positions,
    initial_time,
    final_time,
    ftle,
    direction,
    save_path=None,           # Optional: str or None
    camera_setup=None         # Optional: tuple (position, focal_point, roll)
):
    """
    Plots the FTLE field over a mesh using PyVista, compatible with staggered arrays.

    Parameters:
        node_cons (List[np.ndarray]): Face connectivity arrays per time step.
        node_positions (List[np.ndarray]): Node positions per time step.
        initial_time (int): Starting time step.
        final_time (int): Ending time step.
        ftle (np.ndarray): FTLE values for plotting.
        direction (str): Advection direction: 'forward' or 'backward'.
        save_path (str, optional): If given, saves the plot as an image at this path.
        camera_setup (tuple, optional): (position, focal_point, roll) to fully specify camera.
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

    # Set full camera state if provided
    if camera_setup is not None:
        position, focal_point, roll = camera_setup
        pl.camera.position = position
        pl.camera.focal_point = focal_point
        pl.camera.roll = roll

    if save_path is not None:
        pl.show(screenshot=save_path)
        print(f"Figure saved to {save_path}")
    else:
        # Add on-screen prompt
        pl.add_text("Press 'c' to print camera state", position='upper_left',
                    font_size=10, color='white', name='cam_text')

        def report_camera_state():
            cam = pl.camera
            print("Camera Info:")
            print(f"  position     = {tuple(np.round(cam.position, 4))}")
            print(f"  focal_point  = {tuple(np.round(cam.focal_point, 4))}")
            print(f"  roll         = {round(cam.roll, 2)}°")

        pl.add_key_event('c', report_camera_state)
        pl.show()

    return 0
