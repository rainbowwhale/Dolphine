# Transducer Mask API Documentation

## Overview

The transducer mask functionality implements band-limited interpolation for creating accurate spatial distributions of ultrasound transducer elements on computational grids. This implementation is based on the methodology described in:

**Reference:** https://doi.org/10.1121/1.5116132

## Recent Improvements

### Grid-Aligned Sampling
Points are now sampled according to grid spacing (dx, dy, dz) for better interpolation accuracy and efficiency.

### Error-Tolerance-Based Sampling
The number of sampling points is automatically calculated based on desired error tolerance, eliminating manual tuning.

### Vectorized Implementation
Optimized for large arrays using vectorized operations, suitable for 2D matrix probes with 10,000+ elements.

## Key Concepts

### Rectangular Transducer Elements

Each transducer element is modeled as a rectangular surface with:
- **Width** (lateral direction): Controlled by `element_width` parameter
- **Height** (elevation direction): Controlled by `element_height` parameter
- **Position**: Determined by element index and `pitch` parameter

### Band-Limited Interpolation

Instead of representing each element as a single point source, the mask approach:
1. Samples points across the element surface (grid-aligned or uniform)
2. Uses sinc interpolation to distribute each point's contribution onto nearby grid cells
3. Creates a smooth spatial distribution that respects the Nyquist sampling theorem
4. Automatically determines optimal sampling based on error tolerance

### Grid Configurations

The implementation supports two grid types:
- **Normal Grid**: Standard grid with cell centers at integer multiples of grid spacing
- **Staggered Grid**: Half-cell offset for velocity components in staggered-grid FDTD schemes

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

Generate points on a rectangular element surface with optional grid alignment.

```python
points = transducer.generate_element_surface_points(
    element_idx, 
    n_points_x=None,  # Auto-calculated if grid provided
    n_points_y=None,  # Auto-calculated if grid provided
    grid=None,        # For automatic point spacing
    grid_aligned=True # Align with grid spacing
)
```

**Parameters:**
- `element_idx` (int): Index of the element (0 to n_elements-1)
- `n_points_x` (int, optional): Number of sample points along element width. Auto-calculated if None and grid provided.
- `n_points_y` (int, optional): Number of sample points along element height. Auto-calculated if None and grid provided.
- `grid` (Grid, optional): Grid object for automatic point spacing calculation
- `grid_aligned` (bool): If True and grid provided, align points with grid spacing (default: True)

**Returns:**
- `points` (ndarray): Array of shape (n_points, 3) with (x, y, z) coordinates in meters

**Examples:**

Legacy uniform spacing:
```python
tx = Transducer(n_elements=32, element_width=0.00028, element_height=0.002)
points = tx.generate_element_surface_points(element_idx=0, n_points_x=5, n_points_y=5)
# points.shape = (25, 3)  # 5x5 = 25 points
```

New grid-aligned with auto-calculation:
```python
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
points = tx.generate_element_surface_points(element_idx=0, grid=grid, grid_aligned=True)
# Points automatically calculated and aligned to grid spacing
# Typically results in more points for better accuracy
```

##### calculate_bli_points_for_error()

**NEW**: Calculate the number of BLI sampling points needed for a given error tolerance.

```python
n_points_x, n_points_y = transducer.calculate_bli_points_for_error(
    grid, 
    error_tolerance=0.01  # 1% error
)
```

**Parameters:**
- `grid` (Grid): Grid object defining the computational domain
- `error_tolerance` (float): Desired relative error tolerance (e.g., 0.01 for 1% error, 0.05 for 5% error)

**Returns:**
- `tuple`: (n_points_x, n_points_y) recommended number of sampling points

**Error Tolerance Guidelines:**
- `0.001` (0.1% error): High accuracy, ~960 points for typical element
- `0.01` (1% error): Good accuracy, ~540 points (recommended)
- `0.05` (5% error): Moderate accuracy, ~240 points
- `0.1` (10% error): Fast computation, ~150 points

**Example:**
```python
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
tx = Transducer(n_elements=32)

# Calculate points for 1% error
n_x, n_y = tx.calculate_bli_points_for_error(grid, error_tolerance=0.01)
print(f"Need {n_x} × {n_y} = {n_x*n_y} points for 1% error")
```

##### generate_all_elements_surface_points()

Generate surface points for all transducer elements.

```python
points_list, element_indices = transducer.generate_all_elements_surface_points(
    n_points_x=None,
    n_points_y=None,
    grid=None,
    grid_aligned=True
)
```

**Parameters:**
- `n_points_x` (int, optional): Number of sample points along element width
- `n_points_y` (int, optional): Number of sample points along element height  
- `grid` (Grid, optional): Grid object for automatic point spacing calculation
- `grid_aligned` (bool): If True and grid provided, align points with grid spacing

**Returns:**
- `points_list` (list): List of arrays, each containing points for one element
- `element_indices` (list): List of element indices corresponding to each point set

##### band_limited_interpolation_mask()

Create a mask using band-limited interpolation for a single element with vectorized optimization.

```python
mask = transducer.band_limited_interpolation_mask(
    grid, 
    element_idx, 
    n_points_x=None,         # Auto-calculated if None
    n_points_y=None,         # Auto-calculated if None
    z0=0.0, 
    staggered=False,
    kernel_radius=None,      # Default: 3
    error_tolerance=None,    # NEW: Auto-calculate points
    grid_aligned=True        # NEW: Grid-aligned sampling
)
```

**Parameters:**
- `grid` (Grid): Grid object defining the computational domain
- `element_idx` (int): Index of the element to create mask for
- `n_points_x` (int, optional): Number of sample points along element width (auto if None)
- `n_points_y` (int, optional): Number of sample points along element height (auto if None)
- `z0` (float): Z-position of the transducer surface in meters
- `staggered` (bool): If True, use staggered grid offsets (half-grid spacing)
- `kernel_radius` (int, optional): Sinc kernel radius in grid cells (default: 3)
- `error_tolerance` (float, optional): **NEW** - Auto-calculate n_points for this error level (e.g., 0.01)
- `grid_aligned` (bool): **NEW** - If True, align sampling points with grid spacing (default: True)

**Returns:**
- `mask` (ndarray): Array of shape (grid.nx, grid.ny, grid.nz) with interpolated weights

**Properties of the mask:**
- Sum of all weights ≈ 1.0 (normalized)
- Smooth spatial distribution using sinc interpolation
- Non-zero values extend ~kernel_radius grid cells from source points
- May have small negative lobes (characteristic of sinc function)
- **Vectorized implementation for efficiency**

**Examples:**

Legacy API (still works):
```python
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
tx = Transducer(n_elements=32, element_width=0.00028)
mask = tx.band_limited_interpolation_mask(grid, element_idx=0, n_points_x=5, n_points_y=5)
# mask.shape = (128, 64, 128), np.sum(mask) ≈ 1.0
```

New API with error tolerance (recommended):
```python
# Automatically calculate points for 1% error and align to grid
mask = tx.band_limited_interpolation_mask(
    grid, element_idx=0, error_tolerance=0.01, grid_aligned=True
)
# Points are auto-calculated based on element size and grid spacing
# Typically faster and more accurate than manual specification
```

##### create_element_masks()

Create masks for all transducer elements.

```python
masks = transducer.create_element_masks(
    grid, 
    z0=0.0, 
    n_points_x=5, 
    n_points_y=5,
    staggered=False
)
```

**Parameters:**
- `grid` (Grid): Grid object defining the computational domain
- `z0` (float): Z-position of the transducer surface in meters
- `n_points_x` (int): Number of sample points along element width
- `n_points_y` (int): Number of sample points along element height
- `staggered` (bool): If True, use staggered grid offsets

**Returns:**
- `masks` (list): List of mask arrays, one for each element

**Example:**
```python
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
tx = Transducer(n_elements=32)
masks = tx.create_element_masks(grid, z0=0.0, staggered=False)
# len(masks) = 32
# Each mask has shape (128, 64, 128)
```

## Usage Examples

### Basic Mask Generation

```python
from grid import Grid
from transducer import Transducer

# Create grid and transducer
grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
tx = Transducer(n_elements=32, pitch=0.0003, element_width=0.00028, element_height=0.002)

# Generate masks for all elements
masks = tx.create_element_masks(grid, z0=0.0, n_points_x=5, n_points_y=5, staggered=False)

# Use in source injection
for step in range(n_steps):
    for elem_idx, mask in enumerate(masks):
        # Apply mask-weighted source injection
        pressure_field += mask * source_signal[step] * apodization[elem_idx]
```

### Staggered Grid for Velocity Components

```python
# For pressure field (normal grid)
masks_pressure = tx.create_element_masks(grid, z0=0.0, staggered=False)

# For velocity components (staggered grid)
masks_velocity = tx.create_element_masks(grid, z0=0.0, staggered=True)
```

### Adjusting Sampling Density

```python
# Lower sampling (faster, less accurate)
masks_coarse = tx.create_element_masks(grid, n_points_x=3, n_points_y=3)

# Higher sampling (slower, more accurate)
masks_fine = tx.create_element_masks(grid, n_points_x=7, n_points_y=7)
```

## Performance Considerations

### Sampling Density

- **n_points_x, n_points_y**: Higher values increase accuracy but also computation time
- Recommended: 3-7 points per dimension for most applications
- Total surface points = n_points_x × n_points_y

### Computational Cost

The mask generation time scales with:
- Number of surface points (n_points_x × n_points_y)
- Grid size (nx × ny × nz)
- Kernel radius (fixed at 3 cells)

Pre-compute masks once before the time-stepping loop for best performance.

### Memory Usage

Each mask requires memory = nx × ny × nz × sizeof(float32)

For a 256×128×256 grid with 64 elements:
- Per mask: ~33 MB
- Total: ~2.1 GB

Consider computing masks on-demand if memory is limited.

## Validation

The implementation has been validated to ensure:

1. **Normalization**: Sum of mask weights ≈ 1.0 (within ~10%)
2. **Localization**: Non-zero weights confined to element vicinity
3. **Smoothness**: Continuous spatial distribution via sinc interpolation
4. **Grid alignment**: Proper handling of centered coordinate systems
5. **Staggered support**: Correct half-cell offsets for staggered grids

See `examples/test_transducer_mask.py` for comprehensive validation tests.

## References

1. Band-limited interpolation methodology: https://doi.org/10.1121/1.5116132
2. Sinc interpolation: https://en.wikipedia.org/wiki/Sinc_function
3. Staggered-grid FDTD: Virieux, J. (1986). P-SV wave propagation in heterogeneous media
