import numpy as np
import pyvista as pv


def plot_FTLE_mesh(
    node_cons,
    node_positions,
    initial_time,
    final_time,
    ftle,
    direction,
    save_path=None,          # Optional: str or None
    camera_position=None     # Optional: tuple (x, y, z)
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
        camera_position (tuple, optional): Camera (x, y, z) position to use.
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

    # Set user-provided camera position
    if camera_position is not None:
        pl.camera.position = camera_position

    if save_path is not None:
        pl.show(screenshot=save_path)
        print(f"Figure saved to {save_path}")
    else:
        # Add text indicator for key press
        pl.add_text("Press 'c' to print camera position", position='upper_left',
                    font_size=15, color='red', name='cam_text')

        def report_camera_position():
            print(f'Current Camera Position: {pl.camera.position}')

        pl.add_key_event('c', report_camera_position)
        pl.show()

    return 0
