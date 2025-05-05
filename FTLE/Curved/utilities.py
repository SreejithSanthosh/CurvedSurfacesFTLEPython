import numpy as np
import pyvista as pv



def plot_FTLE_mesh4(
    node_cons,
    node_positions,
    ftle,
    isotropy,
    back_node_cons,
    back_node_positions,
    back_ftle,
    back_isotropy,
    initial_time,
    final_time,
    save_path=None,
    camera_setup=None
):
    """
    Plots FTLE and isotropy fields (forward and backward) over a mesh using PyVista as 2x2 subplots.
    """

    plotter = pv.Plotter(shape=(2, 2), window_size=(1920, 1080), off_screen=save_path is not None)

    fields = [
        ("Forward FTLE", node_positions, node_cons, ftle, initial_time, final_time),
        ("Forward Isotropy", node_positions, node_cons, isotropy, initial_time, final_time),
        ("Backward FTLE", back_node_positions, back_node_cons, back_ftle, final_time, initial_time),
        ("Backward Isotropy", back_node_positions, back_node_cons, back_isotropy, final_time, initial_time)
    ]

    for idx, (title, positions, conns, field_data, i_t, f_t) in enumerate(fields):
        verts = positions[i_t]
        faces = np.hstack([np.full((conns[i_t].shape[0], 1), 3), conns[i_t]]).astype(np.int32).flatten()

        surf = pv.PolyData(verts, faces)
        surf["field"] = field_data
        surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)
        smooth_surf = surf.subdivide(4)
        smooth_surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)

        plotter.subplot(idx // 2, idx % 2)

        # Per-subplot scalar bar settings with unique name
        scalar_bar_args = {
            "title": title,
            "vertical": True,
            "title_font_size": 14,
            "label_font_size": 14,
            "n_labels": 5,
            "position_x": 0.85,
            "position_y": 0.1,
            "width": 0.1,
            "height": 0.7,
        }

        plotter.add_mesh(
            smooth_surf,
            scalars="field",
            cmap="turbo",
            scalar_bar_args=scalar_bar_args,
            interpolate_before_map=True,
            smooth_shading=True,
            show_edges=False,
            ambient=0.5,
            diffuse=0.6,
            specular=0.3,
            show_scalar_bar=True  
        )

        plotter.add_text(f"{title}\nTime {i_t} to {f_t}", font_size=12)

        if camera_setup:
            position, focal_point, roll = camera_setup
            plotter.camera.position = position
            plotter.camera.focal_point = focal_point
            plotter.camera.roll = roll

    if save_path:
        plotter.show(screenshot=save_path)
        print(f"Saved 2x2 FTLE visualization to {save_path}")
    else:
        plotter.show()

    return 0


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
    smooth_surf = surf.subdivide(4)  # You can try 1, 2, or 3 (more = finer)

    smooth_surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)

    pl = pv.Plotter(off_screen=save_path is not None, window_size=(1920, 1080))

    pl.add_mesh(smooth_surf, scalars='FTLE', cmap='viridis', interpolate_before_map=True,
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
                    font_size=12, color='red', name='cam_text')

        def report_camera_state():
            cam = pl.camera
            print("Camera Info:")
            print(f"  position     = {tuple(np.round(cam.position, 4))}")
            print(f"  focal_point  = {tuple(np.round(cam.focal_point, 4))}")
            print(f"  roll         = {round(cam.roll, 2)}°")

        pl.add_key_event('c', report_camera_state)
        pl.show()

    return 0
