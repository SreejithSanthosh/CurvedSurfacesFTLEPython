import numpy as np
import pyvista as pv



def plot_FTLE_mesh(
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


def make_vector_field_video(
    node_cons,
    node_positions,
    node_velocities,
    initial_time,
    final_time,
    time_steps,
    save_path=None,
    framerate=10,
    camera_setup=None
):
    """
    Create a video visualizing velocity vector fields over the surface mesh.

    Parameters:
        node_cons: List of (M_t, 3) arrays for mesh connectivity.
        node_positions: List of (N_t, 3) arrays for mesh nodes at each time step.
        node_velocities: List of (N_t, 3) arrays for node velocities at each time step.
        initial_time: A float of the lower bound of the time interval.
        final_time: A float of the upper bound of the time interval.
        time_steps: list or array of floats corresponding to time steps.
        save_path: Optional string to save the video (e.g., "vector_field_video.mp4").
        framerate: Frames per second for the video.
        camera_setup: Optional tuple (position, focal_point, roll) to set camera state.
    """
    time_steps = time_steps.tolist()

    if initial_time >= final_time:
        raise ValueError("Initial time must be less than final time")
    if initial_time not in time_steps or final_time not in time_steps:
        raise ValueError("Initial and final times must be in `time_steps`.")
    if initial_time == final_time:
        raise ValueError("Initial and final time steps must differ.")

    initial_time_index = time_steps.index(initial_time)
    final_time_index = time_steps.index(final_time)

    pl = pv.Plotter(off_screen=save_path is not None, window_size=(1920, 1080))
    if save_path:
        pl.open_movie(save_path, framerate=framerate)

    for t_index in range(initial_time_index, final_time_index + 1):
        pl.clear()

        verts = node_positions[t_index]
        conns = node_cons[t_index]
        faces = np.hstack([np.full((conns.shape[0], 1), 3), conns]).astype(np.int32).flatten()
        velocities = node_velocities[t_index]

        surf = pv.PolyData(verts, faces)
        surf["vectors"] = velocities
        surf.compute_normals(cell_normals=False, point_normals=True, feature_angle=45, inplace=True)

        pl.add_mesh(surf, color="white", opacity=0.6, show_edges=True)
        pl.add_arrows(verts, velocities, mag=0.5, color='blue')

        pl.add_text(f"Time: {time_steps[t_index]}", font_size=20)

        # Apply fixed camera setup if provided
        if camera_setup:
            position, focal_point, roll = camera_setup
            pl.camera.position = position
            pl.camera.focal_point = focal_point
            pl.camera.roll = roll

        if save_path:
            pl.write_frame()
        else:
            pl.show(auto_close=False, interactive_update=True)
            pl.update()

    if save_path:
        pl.close()

    print(f"Finished rendering. {'Saved video at ' + save_path if save_path else 'Visualization complete.'}")



