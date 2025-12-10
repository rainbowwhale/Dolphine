# Transducer Mask API Documentation

## Overview

The transducer mask functionality implements band-limited interpolation for creating accurate spatial distributions of ultrasound transducer elements on computational grids. This implementation is based on the methodology described in:

**Reference:** https://doi.org/10.1121/1.5116132

## Key Features

### Correct BLI Formula
Uses proper sinc interpolation: `weight = sinc((px-gx)/dx) * sinc((py-gy)/dy) * sinc((pz-gz)/dz)` per reference paper.

### Sparse Representation
Returns (indices, weights) tuples instead of dense 3D arrays for ~98% memory savings.

### Vectorized Implementation
Fully vectorized calculation using grid axis vectors for optimal performance.

## Key Concepts

### Rectangular Transducer Elements

Each transducer element is modeled as a rectangular surface with:
- **Width** (lateral direction): Controlled by `element_width` parameter
- **Height** (elevation direction): Controlled by `element_height` parameter
- **Position**: Determined by element index and `pitch` parameter

### Band-Limited Interpolation

Instead of representing each element as a single point source, the sparse mask approach:
1. Samples points uniformly across the element surface
2. Uses sinc interpolation to distribute each point's contribution onto nearby grid cells
3. Creates a "star" pattern spatial distribution as described in reference
4. Returns only nonzero indices and weights (sparse format)

### Grid Configurations

The implementation supports:
- **Normal Grid**: Standard grid for pressure fields at cell centers
- **Staggered Grid**: Half-cell offsets for velocity components (Vx, Vy, Vz) in FDTD Yee grids

## API Reference

### Class: Transducer

#### Constructor

```python
Transducer(n_elements=64, pitch=0.0003, element_width=0.00028, kerf=0.00002, 
           center_freq=5e6, c=1540.0, element_height=0.010)
```

**Parameters:**
- `n_elements` (int): Number of transducer elements
- `pitch` (float): Center-to-center spacing between elements (meters)
- `element_width` (float): Width of each element in lateral direction (meters)
- `kerf` (float): Gap between elements (meters)
- `center_freq` (float): Center frequency of transducer (Hz)
- `c` (float): Speed of sound (m/s)
- `element_height` (float): Height of each element in elevation direction (meters)

#### Methods

##### generate_element_surface_points()

Generate uniformly distributed points on rectangular element surface.

```python
points = transducer.generate_element_surface_points(
    element_idx, 
    n_points_x,
    n_points_y
)
```

**Parameters:**
- `element_idx` (int): Index of the element (0 to n_elements-1)
- `n_points_x` (int): Number of sample points along element width
- `n_points_y` (int): Number of sample points along element height

**Returns:**
- `points` (ndarray): Array of shape (n_points_x × n_points_y, 3) with (x, y, z) coordinates in meters

**Example:**
```python
tx = Transducer(n_elements=32, element_width=0.00028, element_height=0.002)
points = tx.generate_element_surface_points(element_idx=0, n_points_x=5, n_points_y=5)
# points.shape = (25, 3)  # 5x5 = 25 points uniformly distributed
```

##### band_limited_interpolation_weights()

Compute BLI weights for source points using vectorized calculation.

```python
indices, weights = transducer.band_limited_interpolation_weights(
    grid, 
    points,
    z0=0.0,
    kernel_radius=3,
    staggered_component=None,
    use_gpu=False
)
```

**Parameters:**
- `grid` (Grid): Grid object with axis vectors (x_vec, y_vec, z_vec)
- `points` (ndarray): Array of (x, y, z) source point coordinates
- `z0` (float): Z-position of transducer surface (meters)
- `kernel_radius` (int): Sinc kernel radius in grid cells (default: 3)
- `staggered_component` (str): None for pressure, 'x'/'y'/'z' for velocity components
- `use_gpu` (bool): Use CuPy for GPU acceleration if available

**Returns:**
- `indices` (ndarray): Array of shape (N, 3) with grid indices (i, j, k)
- `weights` (ndarray): Array of shape (N,) with corresponding weights (normalized, sum=1.0)

**Formula:**
```
weight = sinc((px - gx) / dx) * sinc((py - gy) / dy) * sinc((pz - gz) / dz)
```
where (px, py, pz) are point coordinates and (gx, gy, gz) are grid node positions.

##### create_element_mask()

Create sparse BLI mask for a single element.

```python
indices, weights = transducer.create_element_mask(
    grid,
    element_idx,
    n_points_x,
    n_points_y,
    z0=0.0,
    kernel_radius=3,
    staggered_component=None,
    use_gpu=False
)
```

**Parameters:**
- `grid` (Grid): Grid object
- `element_idx` (int): Element index
- `n_points_x` (int): Number of sample points along width
- `n_points_y` (int): Number of sample points along height
- `z0` (float): Z-position of transducer surface
- `kernel_radius` (int): Sinc kernel radius (grid cells)
- `staggered_component` (str): None, 'x', 'y', or 'z' for staggered grids
- `use_gpu` (bool): Use GPU acceleration

**Returns:**
- `indices` (ndarray): (N, 3) array of grid indices
- `weights` (ndarray): (N,) array of weights

**Example:**
```python
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
tx = Transducer(n_elements=32)
indices, weights = tx.create_element_mask(grid, element_idx=0, n_points_x=5, n_points_y=5)

# Direct injection in simulation
pressure[indices[:, 0], indices[:, 1], indices[:, 2]] += signal * weights
```

##### create_element_masks_staggered()

Create staggered grid masks for velocity components.

```python
staggered_masks = transducer.create_element_masks_staggered(
    grid,
    element_idx,
    n_points_x,
    n_points_y,
    z0=0.0,
    kernel_radius=3,
    use_gpu=False
)
```

**Parameters:**
- `grid` (Grid): Grid object
- `element_idx` (int): Element index
- `n_points_x` (int): Number of sample points along width
- `n_points_y` (int): Number of sample points along height
- `z0` (float): Z-position of transducer surface
- `kernel_radius` (int): Sinc kernel radius
- `use_gpu` (bool): Use GPU acceleration

**Returns:**
- `dict`: `{'vx': (indices, weights), 'vy': (indices, weights), 'vz': (indices, weights)}`

**Example:**
```python
staggered_masks = tx.create_element_masks_staggered(grid, element_idx=0, n_points_x=3, n_points_y=3)

vx_indices, vx_weights = staggered_masks['vx']
vy_indices, vy_weights = staggered_masks['vy']
vz_indices, vz_weights = staggered_masks['vz']

# Use in FDTD simulation
vx[vx_indices[:, 0], vx_indices[:, 1], vx_indices[:, 2]] += signal * vx_weights
```

##### create_all_element_masks()

Create BLI masks for all elements.

```python
masks = transducer.create_all_element_masks(
    grid,
    n_points_x,
    n_points_y,
    z0=0.0,
    kernel_radius=3,
    staggered=False,
    use_gpu=False
)
```

**Parameters:**
- `grid` (Grid): Grid object
- `n_points_x` (int): Number of sample points along width per element
- `n_points_y` (int): Number of sample points along height per element
- `z0` (float): Z-position of transducer surface
- `kernel_radius` (int): Sinc kernel radius
- `staggered` (bool): If True, return staggered masks for velocity components
- `use_gpu` (bool): Use GPU acceleration

**Returns:**
- If `staggered=False`: list of `(indices, weights)` tuples
- If `staggered=True`: list of dicts with 'vx', 'vy', 'vz' keys

**Example:**
```python
# Pressure field masks
masks = tx.create_all_element_masks(grid, n_points_x=5, n_points_y=5, staggered=False)

for elem_idx, (indices, weights) in enumerate(masks):
    pressure[indices[:, 0], indices[:, 1], indices[:, 2]] += signal[elem_idx] * weights
```

## Usage Examples

### Basic Sparse Mask Generation

```python
from grid import Grid
from transducer import Transducer

# Create grid and transducer
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
tx = Transducer(n_elements=32, pitch=0.0003, element_width=0.00028, element_height=0.002)

# Generate sparse mask for one element
indices, weights = tx.create_element_mask(grid, element_idx=0, n_points_x=5, n_points_y=5)

# Direct injection in simulation
for step in range(n_steps):
    pressure[indices[:, 0], indices[:, 1], indices[:, 2]] += source_signal[step] * weights
```

### Multiple Elements

```python
# Generate masks for all elements
masks = tx.create_all_element_masks(grid, z0=0.0, n_points_x=5, n_points_y=5, staggered=False)

# Use in source injection
for step in range(n_steps):
    for elem_idx, (indices, weights) in enumerate(masks):
        pressure[indices[:, 0], indices[:, 1], indices[:, 2]] += signal[step] * apodization[elem_idx] * weights
```

### Staggered Grid for Velocity Components

```python
# Get staggered masks for all velocity components
staggered_masks = tx.create_element_masks_staggered(grid, element_idx=0, n_points_x=3, n_points_y=3)

# Extract components
vx_indices, vx_weights = staggered_masks['vx']
vy_indices, vy_weights = staggered_masks['vy']
vz_indices, vz_weights = staggered_masks['vz']

# Use in FDTD
vx_field[vx_indices[:, 0], vx_indices[:, 1], vx_indices[:, 2]] += signal * vx_weights
vy_field[vy_indices[:, 0], vy_indices[:, 1], vy_indices[:, 2]] += signal * vy_weights
vz_field[vz_indices[:, 0], vz_indices[:, 1], vz_indices[:, 2]] += signal * vz_weights
```

### Adjusting Sampling Density

```python
# Lower sampling (faster, less accurate)
indices_coarse, weights_coarse = tx.create_element_mask(grid, element_idx=0, n_points_x=3, n_points_y=3)

# Higher sampling (slower, more accurate)
indices_fine, weights_fine = tx.create_element_mask(grid, element_idx=0, n_points_x=7, n_points_y=7)
```

## Performance Considerations

### Sampling Density

- **n_points_x, n_points_y**: Higher values increase accuracy but also computation time
- Recommended: 3-7 points per dimension for most applications
- Total surface points = n_points_x × n_points_y

### Sparse Format Benefits

- Memory savings: ~98% vs dense arrays
- Typical sparse mask: 0.026 MB vs dense: 1.0 MB (for 64³ grid)
- Faster source injection (only update nonzero indices)

### Computational Cost

Pre-compute masks once before the time-stepping loop for best performance.

Mask generation time scales with:
- Number of surface points (n_points_x × n_points_y)
- Kernel radius (fixed at 3 cells)
- Number of elements

### GPU Acceleration

```python
# Use GPU for large arrays
indices, weights = tx.create_element_mask(grid, element_idx=0, n_points_x=5, n_points_y=5, use_gpu=True)
```

Requires CuPy installation. Beneficial for:
- Large grids (>256³)
- Many elements (>100)
- High sampling density (>10×10 points)

## Validation

The implementation has been validated to ensure:

1. **Correct BLI formula**: Uses grid spacing (dx, dy, dz) in sinc denominators
2. **Normalization**: Sum of weights = 1.0 exactly
3. **Star pattern**: Creates characteristic sinc interpolation pattern
4. **Sparse format**: Indices correspond correctly to grid positions
5. **Staggered support**: Correct half-cell offsets for velocity components
6. **Vectorization**: Efficient calculation using grid axis vectors

See `examples/test_corrected_bli.py` for comprehensive validation tests.

## References

1. Band-limited interpolation methodology: https://doi.org/10.1121/1.5116132
2. Sinc interpolation: https://en.wikipedia.org/wiki/Sinc_function
3. Staggered-grid FDTD: Virieux, J. (1986). P-SV wave propagation in heterogeneous media
