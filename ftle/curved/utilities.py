import numpy as np
import pyvista as pv

def camera_position_to_angles(position):
    """
    Converts camera 3D position to (azimuth, elevation) in degrees.
    """
    x, y, z = position
    r = np.linalg.norm([x, y, z])
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z / r))
    return azimuth, elevation


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

    if view_angle is not None:
        az, el = view_angle
        pl.camera.azimuth = az
        pl.camera.elevation = el

    if save_path is not None:
        pl.show(screenshot=save_path)
        print(f"Figure saved to {save_path}")

    else:
        # Add a callback to update camera angles dynamically
        def update_camera_info():
            az, el = camera_position_to_angles(pl.camera.position)
            pl.add_text(f"Azimuth: {az:.1f}°, Elevation: {el:.1f}°",
                        position='upper_left', font_size=10, color='white',
                        name='cam_text')

        # Add callback before rendering every frame
        pl.add_callback(update_camera_info, interval=100)  # ms
        pl.show()

    return 0
