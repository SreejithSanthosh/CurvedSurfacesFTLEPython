<!-- ---
title: FTLE on Curved Surfaces - Python Implementation 
parent: Tutorial - FTLE codes
layout: home
nav_order: 4
--- -->

## Documentation

The python package for computing the coherent structures and Lagrangian deformation for flow on a curved surface is given in this [link](https://github.com/SreejithSanthosh/CurvedSurfacesFTLEPython). To understand the mathematical background or additional information on the methods discussed here, we refer you to the accompanying manuscript [S. Santhosh et al](Necessary Link).




## Installation 

To install the python package you can use the following command in your python environment:

```
pip install -e git+https://github.com/bfencil/ftlePackage.git#egg=ftlePackage
```

This pacakge was made in python version 3.11.8.


## Dense FTLE Computations in $\mathbb{R}^2$ and $\mathbb{R}^3$

### Overview

The dense FTLE codes differ from the sparse FTLE codes in that the initial particle positions are set on a uniform grid that spans the entire domain. This is opposed to the sparse FTLE codes where particles are placed arbitrarily in space.

This uniform grid structure is useful when:

- A global picture of the entire velocity field is desired.
- The velocity data itself is sparse but the user wants a continuous FTLE field.
- It simplifies the FTLE computation using finite-difference schemes.

The primary functions of interest for dense FTLE computations are:

- `FTLE_2d_dense` — for dense FTLE computation in $\mathbb{R}^2$.
- `FTLE_3d_dense` — for dense FTLE computation in $\mathbb{R}^3$.


---

### Example Usage

A typical workflow using `FTLE_2d_dense` looks like this:

```python
from ftle.flat.dense import FTLE_2d_dense

ftle_values, trajectories = FTLE_2d_dense(
    velocity_points,
    velocity_vectors,
    x_grid,
    y_grid,
    dt,
    initial_time,
    final_time,
    time_steps,
    direction,
    time_indepedent=False,
    plot_ftle=True
)
```

---

### Inputs for `FTLE_2d_dense`

- `velocity_points`: (M, 2) array of the known locations where velocity data is given.
- `velocity_vectors`: (M, 2) if time-independent or (M, 2, T) if time-dependent velocity vectors.
- `x_grid`: 2D array of X positions from `np.meshgrid`.
- `y_grid`: 2D array of Y positions from `np.meshgrid`.
- `dt`: Float, time step size for RK4 integration. Must have $0 < dt \leq 1$.
- `initial_time`: Starting time index.
- `final_time`: Ending time index.
- `time_steps`: List of available time steps.
- `direction`: String `"forward"` or `"backward"`.
- `time_indepedent`: Boolean, if True uses static velocity field.
- `plot_ftle`: Boolean, if True plots the FTLE field directly.


---

### Inputs for `FTLE_3d_dense`

Very similar to `FTLE_2d_dense` but adapted for 3D.

```python
from ftle.flat.dense import FTLE_3d_dense

ftle_values, trajectories = FTLE_3d_dense(
    velocity_points,
    velocity_vectors,
    x_grid,
    y_grid,
    z_grid,
    dt,
    initial_time,
    final_time,
    time_steps,
    direction,
    time_indepedent=False,
    plot_ftle=True
)
```

#### Inputs

- `x_grid, y_grid, z_grid`: 3D arrays from `np.meshgrid`.
- All other inputs are the same as `FTLE_2d_dense`.

---

### FTLE Computation Methodology

The FTLE computation for dense grids uses a finite difference scheme to compute the deformation gradient tensor $\partial X_f / \partial X_0$.

- For 2D: This is done in `FTLE_2d_compute`.
- For 3D: This is done in `FTLE_3d_compute`.

Both methods:

- Use central differences across the local grid.
- Build the deformation gradient $F$.
- Compute the Cauchy-Green strain tensor $C = F^T F$.
- Extract the largest eigenvalue of $C$.
- Compute FTLE as:

$$
\text{FTLE} = \frac{1}{2 |t_f - t_i|} \ln(\sqrt{\lambda_{max}})
$$

where $\lambda_{max}$ is the largest eigenvalue of $C$.

---

### Notes on Usage

- The uniform grid from `np.meshgrid` must fully cover the desired domain.
- For plotting FTLE:

```python
from ftle.flat.utilities import plot_FTLE_2d, plot_FTLE_3d
```

- The plotting functions will automatically scatter the FTLE values over the grid points.

- In 3D, the plotting shows the full 3D cloud of points and also provides 2D slices along the $XZ$ plane at select $Y$ values.

---

### Example FTLE Plot in 2D

```python
plot_FTLE_2d(particles_positions, ftle)
```
Place example here!
---

### Example FTLE Plot in 3D

```python
plot_FTLE_3d(particles_positions, ftle)
```
Place example here!
---



## Sparse FTLE Computations in $\mathbb{R}^2$ and $\mathbb{R}^3$

### Overview

The sparse FTLE routines are designed to work with velocity fields that are defined only at scattered locations in space. These methods do not require a grid or mesh. Instead, they interpolate the velocity field from known velocity locations and compute FTLE from particle trajectories.

Sparse FTLE computations are especially useful when:

- Your velocity field is only known at irregular points (experimental data, scattered sensors).
- You want to avoid gridding or meshing overhead.

Sparse FTLE methods rely on KD-Trees and least-squares estimation of the deformation gradient using nearest-neighbor points.

---

## Available Functions

- `FTLE_2d_sparse` : Sparse FTLE calculation for 2D velocity fields.
- `FTLE_3d_sparse` : Sparse FTLE calculation for 3D velocity fields.
- `compute_Ftle_sparse` : Computes FTLE values given only initial and final particle positions.
- `plot_FTLE_2d` : Visualization for 2D FTLE.
- `plot_FTLE_3d` : Visualization for 3D FTLE with optional slicing.

---

## Example Usage: FTLE_2d_sparse

```python
from ftle.flat.sparse import FTLE_2d_sparse

ftle_values, trajectories = FTLE_2d_sparse(
    velocity_points,     # (M, 2) array of known velocity locations
    velocity_vectors,    # (M, 2) or (M, 2, T) array of velocity vectors
    particle_positions,  # (N, 2) array of initial particle positions
    dt=0.1,              # Integration time step (0 < dt <= 1)
    initial_time=0,      
    final_time=10,       
    time_steps=np.arange(0, 11),  
    direction='forward', 
    time_indepedent=False,
    plot_ftle=True,      
    neighborhood=10      
)
```

## Inputs for `FTLE_2d_sparse` and `FTLE_3d_sparse`

- `velocity_points`  
  Array of shape `(M, d)` where `d=2` or `3`. These are the known locations of the velocity data.

- `velocity_vectors`  
  Array of shape `(M, d)` if the velocity is time-independent, or `(M, d, T)` if time-dependent.

- `particle_positions`  
  Initial positions of particles, of shape `(N, d)`.

- `dt`  
  Time step size for integration. Must satisfy `0 < dt <= 1`.

- `initial_time` and `final_time`  
  Integers specifying start and end time steps for the advection.

- `time_steps`  
  Array of allowed time steps (must be consecutive integers starting from 0).

- `direction`  
  Either `"forward"` or `"backward"` advection.

- `time_indepedent`  
  Boolean. If `True`, velocity field is assumed to be time-invariant.

- `plot_ftle`  
  Boolean. If `True`, generates a plot of the FTLE field.

- `neighborhood`  
  Integer. Number of nearest neighbors to use when computing FTLE values.

- `lam`  
  Small regularization constant (default $1 \times 10^{-10}$) for stability.

---

## FTLE Computation via `compute_Ftle_sparse`

The function `compute_Ftle_sparse` can be used directly if you have only the initial and final positions of particles, this works for both `d=2,3`. 

```python
from ftle.flat.sparse import compute_Ftle_sparse

ftle_values = compute_Ftle_sparse(
    initial_positions,  # (N, d)
    final_positions,    # (N, d)
    initial_time, 
    final_time, 
    k=10                # Number of nearest neighbors for FTLE computation
)
```




## Surfaces in $\mathbb{R}^3$

### Overview

This section provides functionality for computing FTLE (Finite Time Lyapunov Exponent) fields for flows on triangulated surfaces in $\mathbb{R}^3$. 

This is distinct from the dense and sparse FTLE methods in Euclidean domains, as here the mesh itself evolves over time and particles are constrained to remain on the surface throughout the advection process.

The primary function for users is `FTLE_mesh`, which wraps the entire process of particle advection and FTLE computation for a triangulated, staggered mesh surface.

Additional functions such as `FTLE_compute` are available if finer control is needed (e.g. using your own advection scheme or particle data). 

---

### Example Usage

The function `FTLE_mesh` uses the input mesh data with a defined vector field data for the mesh to advect particles constrained to the mesh and to compute the FTLE values at each initial particle position. The mesh does not need any specific structure and the mesh itself can move over time and have a different number of nodes per time step. The advection scheme used is a standard RK4 scheme and KDtrees are used to assist in the projecting particles down to the surface after each time step. It is worth noting that due to the general design of `FTLE_mesh` to work with meshes that are not well defined over time(different number of nodes per time step), the RK4 integration time step is just set to $$dt=1$$(i.e the assumed difference between time steps in the mesh data). `FTLE_mesh` returns the FTLE data along with the trajectory data. Also `FTLE_mesh` has built in plotting functionality in a function `plot_FTLE_mesh` which is discussed more in a later section. 

```python
from ftle.curved.mesh import ftle_mesh

ftle_values , trajectories = def FTLE_mesh(
    node_connections,           # List[np.ndarray], each (M_t, 3)
    node_positions,             # List[np.ndarray], each (N_t, 3)
    node_velocities,            # List[np.ndarray], each (N_t, 3)
    particle_positions,         # (P, 3)
    initial_time,               # int
    final_time,                 # int
    time_steps,                 # (T,)
    direction="forward",        # "forward" or "backward"
    plot_ftle=False,            # If True, calls plot_ftle_mesh
    save_path = None,
    camera_setup = None,
    neighborhood=15,            # For FTLE computation
    lam=1e-10                   # Regularization
):
```

### Inputs for `ftle_mesh`

The `ftle_mesh` function computes the FTLE field for flows on a triangulated surface in $\mathbb{R}^3$. It takes the following inputs:

- `node_cons`  
  A list of arrays of shape `(M_t, 3)` for each time step. Each array describes the mesh connectivity at that time step, where `M_t` is the number of faces at time `t`.

- `node_positions`  
  A list of arrays of shape `(N_t, 3)` for each time step, where `N_t` is the number of nodes at time `t`. These are the spatial coordinates of the mesh nodes.

- `node_velocities`  
  A list of arrays of shape `(N_t, 3)` for each time step. These are the velocity vectors at each node.

- `particle_positions`  
  An array of shape `(P, 3)` containing the initial positions of the particles to advect on the surface, where `P` is the number of particles, for ease of plotting the final results it is recommended that the initial positions just correspond to the node positions at the initial time.

- `initial_time`  
  An integer specifying the starting time index for the advection. This must respect final time with respect to the `direction`('forward'/'backward') i.e if `direction` is 'backward' then `initial_time` $>$ `final_time`. Just reverse the inequality for 'forward'.

- `final_time`  
  An integer specifying the ending time index for the advection. This must respect final time with respect to the `direction`('forward'/'backward') i.e if `direction` is 'backward then `initial_time` $>$ `final_time`. Just reverse the inequality for 'forward'.

- `time_steps`  
  An array of time values corresponding to the available mesh data which should be orgainized as consecutive integers starting at 0.

- `direction`  
  A string, either `"forward"` or `"backward"` indicating the direction of advection.

- `plot_ftle`  
  Optional boolean. If `True`, automatically generates a 3D PyVista plot of the FTLE field. This does not save the plot.

- `save_path`
   Optional string. If `plot_ftle` is true then having a save path specified will save the 3d plot in that directory. Take note that if a save path is given the PyVista plotting window will not appear due to PyVista plots having to be saved off screen.

- `camera_setup`
  Optional tuple: (camera position, focal point, camera roll). 

- `neighborhood`  
  Integer (default 15). The number of nearest neighboring particles used when computing the local FTLE gradient.

- `lam`  
  Regularization parameter (default $1 \times 10^{-10}$). Helps stabilize the least-squares fit of the deformation gradient in regions with near-linear configurations.

---


### Inputs for `FTLE_compute`

The `FTLE_compute` function computes FTLE values directly given initial and final particle positions. It can be imported directly `from ftle.curved.mesh import FTLE_compute` This is useful if:

- You already have your own particle trajectories(like what is returned from `ftle_mesh`).
- You want to bypass `ftle_mesh` and just compute FTLE.

Here are the expected inputs

```python
ftle.curved.mesh import FTLE_compute

ftle_values = def FTLE_mesh(
    node_connections,           # List[np.ndarray], each (M_t, 3)
    node_positions,             # List[np.ndarray], each (N_t, 3)
    node_velocities,            # List[np.ndarray], each (N_t, 3)
    particle_positions,         # (P, 3)
    initial_time,               # int
    final_time,                 # int
    time_steps,                 # (T,)
    direction="forward",        # "forward" or "backward"
    plot_ftle=False,            # If True, calls plot_ftle_mesh
    save_path = None,
    camera_setup = None,
    neighborhood=15,            # For FTLE computation
    lam=1e-10                   # Regularization
):
```

- `node_cons`  
  A list of arrays of shape `(M_t, 3)` for each time step. Each array describes the mesh connectivity at that time step, where `M_t` is the number of faces at time `t`.

- `node_positions`  
  A list of arrays of shape `(N_t, 3)` for each time step, where `N_t` is the number of nodes at time `t`. These are the spatial coordinates of the mesh nodes.
- `centroids`  
  A list of arrays of triangle centroids for each time step. These can be generated using the helper function `compute_centroids_staggered`.

- `initial_positions`  
  A `(P, 3)` array containing the positions of particles at the initial time step.

- `final_positions`  
  A `(P, 3)` array containing the positions of particles at the final time step.

- `initial_time`  
  Integer index specifying the start time.

- `final_time`  
  Integer index specifying the end time.

- `neighborhood`  
  The number of neighboring particles to use for local FTLE estimation. Defaults to 15.

- `lam`  
  A small regularization constant (default $1 \times 10^{-10}$) used to stabilize the inversion of matrices during gradient estimation.


### Inputs for `plot_FTLE_mesh` 

The `plot_FTLE_mesh` function can be called directly by importing it `from ftle.curved.utilities import plot_FTLE_mesh`.  Here are the expected inputs:

```python
from ftle.curved.utilities import plot_FTLE_mesh

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
```

- `node_cons`  
  A list of arrays of shape `(M_t, 3)` for each time step. Each array describes the mesh connectivity at that time step, where `M_t` is the number of faces at time `t`.

- `node_positions`  
  A list of arrays of shape `(N_t, 3)` for each time step, where `N_t` is the number of nodes at time `t`. These are the spatial coordinates of the mesh nodes.

- `initial_time`  
  An integer specifying the starting time index for the advection. This must respect final time with respect to the `direction`('forward'/'backward') i.e if `direction` is 'backward' then `initial_time` $>$ `final_time`. Just reverse the inequality for 'forward'.

- `final_time`  
  An integer specifying the ending time index for the advection. This must respect final time with respect to the `direction`('forward'/'backward') i.e if `direction` is 'backward then `initial_time` $>$ `final_time`. Just reverse the inequality for 'forward'.

- `ftle`
  A one dimensional list corresponding to the amount of nodes at time, initial_time. 
  
- `direction`  
  A string, either `"forward"` or `"backward"` indicating the direction of advection.

- `save_path`
   Optional string. If `plot_ftle` is true then having a save path specified will save the 3d plot in that directory. Take note that if a save path is given the PyVista plotting window will not appear due to PyVista plots having to be saved off screen.

- `camera_setup`
  Optional tuple: (camera position, focal point, camera roll). 


When no save_path is specified then you can press 'c' in the PyVista plotting window to display needed information to specifiy `camera_setup` with respect to your current camera view in the PyVista plotting window.


### Example `plot_FTLE_mesh` directly

```python
from ftle.curved.mesh import FTLE_mesh
from ftle.curved.utilities import plot_FTLE_mesh
import h5py


def load_mesh_data_h5(h5_file_path):
    """
    Load staggered curved surface mesh data from an HDF5 file.
    Expects groups:
        /node_cons/{t}
        /position/{t}
        /velocity/{t}
        /time_steps
    Returns dict with lists of per-time-step arrays.
    """
    with h5py.File(h5_file_path, 'r') as f:
        time_steps = f["time_steps"][:]
        total_time = len(time_steps)

        # Convert string keys to sorted integers
        keys = sorted(f["position"].keys(), key=lambda k: int(k))

        position = [f["position"][k][:] for k in keys]
        velocity = [f["velocity"][k][:] for k in keys]
        node_cons = [f["node_cons"][k][:] for k in keys]

        # Sanity check
        assert len(position) == total_time, "Mismatch in position data and time_steps"
        assert len(velocity) == total_time
        assert len(node_cons) == total_time

    return {
        'node_cons': node_cons,             # list of (M_t, 3)
        'position': position,               # list of (N_t, 3)
        'velocity': velocity,               # list of (N_t, 3)
        'time_steps': time_steps,
        'node_number': [p.shape[0] for p in position],  # list of node counts per step
        'total_time': total_time
    }

file_path = f'dataHeart.h5'

mesh_data = load_mesh_data_h5(file_path)


node_positions = mesh_data['position']
node_velocities = mesh_data['velocity']
time_steps = mesh_data['time_steps']
node_cons = mesh_data['node_cons']


direction = "forward"
initial_time = 0
final_time = 5
particle_positions = node_positions[initial_time]



for_ftle, for_trajectories = FTLE_mesh(node_cons, node_positions, node_velocities, particle_positions, initial_time, final_time, time_steps, direction, plot_ftle=True, neighborhood=10)


plot_FTLE_mesh(node_cons, node_positions, initial_time, final_time, for_ftle, direction, save_path="my_plot1.png", camera_setup=((130.4046, -178.1839, -1312.8635), (379.1223, 531.1033, 147.2741), -152.84 ) )

direction = "backward"
initial_time = 28
final_time = 0
particle_positions = node_positions[initial_time]

back_ftle, back_trajectories = FTLE_mesh(node_cons, node_positions, node_velocities, particle_positions, initial_time, final_time, time_steps, direction, plot_ftle=True, neighborhood=10)

plot_FTLE_mesh(node_cons, node_positions, initial_time, final_time, back_ftle, direction, save_path="my_plot2.png", camera_setup=((130.4046, -178.1839, -1312.8635), (379.1223, 531.1033, 147.2741), -152.84 ) )


```

The data in the .h5 file is for a Zrebra fish heart and for the given time have the following FTLE field: 




