# Transducer Module Documentation

## Overview

The Transducer module provides comprehensive modeling of ultrasound transducer arrays with acoustic lens support. This is a clean implementation with the following features:

- **N × M array structure**: `n_cols` (columns) × `n_rows` (rows)
- **Element properties**: Position (3D), angle (normal direction), size (width × height)
- **Array curvature**: Radius of curvature (ROC) for convex arrays
- **Multi-layer acoustic lens**: Elevational focusing with convex/concave layers
- **Plotting functions**: Visualization of array layout and lens cross-section

## Quick Start

```python
from transducer import Transducer, AcousticLens, LensLayer

# Create a simple flat array
tx = Transducer(
    n_cols=64,           # 64 columns (lateral)
    n_rows=8,            # 8 rows (elevation)
    pitch=0.0003,        # 0.3mm lateral pitch
    row_pitch=0.0004,    # 0.4mm elevation pitch
    roc=0,               # 0 = flat array
    center_freq=5e6      # 5 MHz center frequency
)

print(tx)  # Display array properties
```

## Classes

### Transducer

The main class for modeling ultrasound transducer arrays.

#### Constructor

```python
Transducer(
    n_cols: int,              # Number of columns (lateral direction)
    n_rows: int,              # Number of rows (elevation direction)
    pitch: float = 0.0003,    # Lateral element spacing (m)
    row_pitch: float = None,  # Elevation element spacing (m), default = pitch
    element_width: float = None,   # Element width (m), default = pitch - kerf
    element_height: float = None,  # Element height (m), default = row_pitch - kerf
    kerf: float = 0.00002,    # Gap between elements (m)
    roc: float = 0.0,         # Radius of curvature (m), 0 = flat
    center_freq: float = 5e6, # Center frequency (Hz)
    speed_of_sound: float = 1540.0,  # Speed of sound (m/s)
    lens: AcousticLens = None # Acoustic lens object
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_cols` | int | Number of columns |
| `n_rows` | int | Number of rows |
| `n_elements` | int | Total number of elements (n_cols × n_rows) |
| `element_positions` | ndarray (N, 3) | 3D positions of all elements |
| `element_angles` | ndarray (N, 2) | Normal angles (θx, θy) of all elements |
| `array_width` | float | Array width in meters |
| `array_height` | float | Array height in meters |
| `array_size` | tuple | (width, height) in meters |
| `min_dimension` | float | Minimum of width/height |
| `max_dimension` | float | Maximum of width/height |
| `wavelength` | float | Wavelength at center frequency |
| `x_positions` | ndarray | X-coordinates of all elements |
| `y_positions` | ndarray | Y-coordinates of all elements |
| `z_positions` | ndarray | Z-coordinates of all elements |

#### Methods

##### Element Access

```python
# Get element position (x, y, z) in meters
pos = tx.get_element_position(element_idx)

# Get element normal angles (θx, θy) in radians
angles = tx.get_element_angle(element_idx)

# Get element size (width, height) in meters
size = tx.get_element_size(element_idx)

# Get row and column from linear index
row, col = tx.get_element_row_col(element_idx)

# Get linear index from row and column
idx = tx.get_element_index(row, col)
```

##### Beamforming

```python
# Compute focusing delays
# Accepts 2D (x, z) or 3D (x, y, z) focus point
delays = tx.delays_for_focus((0.0, 0.0, 0.03))  # 30mm depth

# Compute steering delays
# Accepts (θx, θy) steering angles in radians
delays = tx.delays_for_steering((np.deg2rad(15), np.deg2rad(10)))

# Generate Hanning apodization weights
apod = tx.apodization_hanning()
```

##### Plotting

```python
# 3D array view
ax = tx.plot_array(show_elements=True, show_normals=False)

# 2D top-down view
ax = tx.plot_array_2d()

# Lens cross-section
ax = tx.plot_lens()
```

##### BLI Mask Generation

```python
from grid import Grid

grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)

# Create mask for single element
indices, weights = tx.create_element_mask(
    grid, element_idx=0,
    n_points_x=5, n_points_y=5,
    kernel_radius=3, tolerance=1e-3
)

# Create masks for all elements
masks = tx.create_all_element_masks(
    grid, n_points_x=5, n_points_y=5,
    staggered=False  # True for velocity components
)
```

---

### AcousticLens

Container for multi-layer acoustic lens modeling.

#### Constructor

```python
AcousticLens(layers: list = None)  # List of LensLayer objects
```

#### Methods

```python
# Add a layer
lens.add_layer(layer)

# Get total thickness profile
thickness = lens.get_total_thickness_profile(y_positions)

# Get layer boundary positions
boundaries = lens.get_layer_boundaries(y_positions)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_layers` | int | Number of layers |
| `layers` | list | List of LensLayer objects |
| `total_max_thickness` | float | Sum of all layer max thicknesses |

---

### LensLayer

Single layer of an acoustic lens.

#### Constructor

```python
LensLayer(
    elevational_roc: float,          # ROC in elevation (m)
    max_thickness: float,            # Maximum thickness (m)
    speed_of_sound: float = 1000.0,  # Speed of sound in lens (m/s)
    density: float = 1100.0,         # Density (kg/m³)
    name: str = ""                   # Optional layer name
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `elevational_roc` | float | Radius of curvature (+ = convex, - = concave, 0 = flat) |
| `max_thickness` | float | Maximum layer thickness |
| `is_convex` | bool | True if ROC > 0 |
| `is_concave` | bool | True if ROC < 0 |
| `is_flat` | bool | True if ROC = 0 or infinite |

---

## Examples

### Flat Array

```python
tx = Transducer(
    n_cols=64,
    n_rows=8,
    pitch=0.0003,
    row_pitch=0.0004,
    roc=0,  # Flat
    center_freq=5e6
)
```

### Curved (Convex) Array

```python
tx = Transducer(
    n_cols=64,
    n_rows=1,
    pitch=0.0004,
    roc=0.05,  # 50mm radius of curvature
    center_freq=3.5e6
)
```

### Array with Acoustic Lens

```python
# Create multi-layer lens
lens = AcousticLens()

# Convex focusing layer
lens.add_layer(LensLayer(
    elevational_roc=0.020,     # 20mm convex
    max_thickness=0.001,       # 1mm
    name="Focus layer"
))

# Flat matching layer
lens.add_layer(LensLayer(
    elevational_roc=0,         # Flat
    max_thickness=0.0005,      # 0.5mm
    name="Matching layer"
))

# Create transducer with lens
tx = Transducer(
    n_cols=64,
    n_rows=5,
    pitch=0.0003,
    lens=lens,
    center_freq=5e6
)
```

### Concave Lens (Diverging)

```python
# Negative ROC creates concave (diverging) lens
layer = LensLayer(
    elevational_roc=-0.030,    # -30mm (concave)
    max_thickness=0.0008,
    name="Diverging layer"
)
# Thickness is minimum at center, maximum at edges
```

---

## Coordinate System

- **X axis**: Lateral direction (left-right)
- **Y axis**: Elevation direction (up-down)
- **Z axis**: Axial direction (depth)
- **Origin**: Center of transducer array
- **Transducer surface**: Z = 0 for flat arrays

## Element Numbering

Elements are numbered in row-major order:
- Row 0: Elements 0 to (n_cols - 1)
- Row 1: Elements n_cols to (2 × n_cols - 1)
- Element (row, col) → index = row × n_cols + col

---

## Running the Demo

```bash
python examples/transducer_demo.py
```

This will demonstrate all features including:
- Flat and curved arrays
- Multi-layer acoustic lens
- Convex and concave lens layers
- Element indexing and access
- Array dimension properties
- Beamforming calculations
- BLI mask generation
- Plotting capabilities
