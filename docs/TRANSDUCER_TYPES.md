# Transducer Types in Dolphine

This document describes the different transducer array types supported in Dolphine and their use cases.

## Overview

Dolphine supports a unified transducer class that handles all ultrasound array types with a common interface:

1. **Linear (1D) Transducer** - Single row of elements for 2D imaging (roc=0)
2. **Convex (1D) Transducer** - Curved single row for wider field of view (roc>0)
3. **1.5D Linear Transducer** - Multiple rows (typically 3-7) with variable heights for elevation focusing (roc=0)
4. **1.5D Convex Transducer** - Multiple rows with lateral curvature for wide FOV and elevation control (roc>0) **NEW!**
5. **2D Matrix Linear Transducer** - Large 2D grid (10s-100s of elements) for 3D volumetric imaging (roc=0)
6. **2D Matrix Convex Transducer** - Large 2D curved grid for 3D imaging with wide FOV (roc>0) **NEW!**

### Unified Interface

All transducer types use a consistent interface with three key parameters:
- **n_cols**: Number of columns (lateral/x direction)
- **n_rows**: Number of rows (elevation/y direction), default=1
- **roc**: Radius of curvature (m). Use 0 or None for linear arrays, >0 for convex arrays.
  - **Applies to all array types**: The ROC parameter now works with 1D, 1.5D, and 2D arrays!
  - **Lateral curvature**: All rows follow the same curved arc in the x-z plane

## 1. Linear (1D) Transducer

### Description
The basic linear array has a single row of elements arranged along the lateral (x) axis. This is the most common transducer type for 2D ultrasound imaging.

### Key Features
- Single row of elements (typical: 64-256 elements)
- Electronic beam steering and focusing in azimuth (x-z plane)
- Fixed elevation focus (mechanical lens)
- Compact and cost-effective
- Flat surface (roc=0)

### Usage
```python
from transducer import Transducer

# Modern interface (preferred)
tx = Transducer(
    n_cols=128,              # Number of elements
    n_rows=1,                # Single row
    pitch=0.0003,            # Element spacing (m)
    element_width=0.00028,   # Element width (m)
    element_height=0.010,    # Element height (m)
    roc=0,                   # Linear (no curvature)
    center_freq=5e6          # Center frequency (Hz)
)

# Legacy interface (still supported)
tx = Transducer(
    n_elements=128,          # Number of elements
    pitch=0.0003,
    element_width=0.00028,
    element_height=0.010,
    center_freq=5e6
)

# Compute focusing delays
focus = (0.0, 0.0, 0.03)  # (x, y, z) in meters
delays = tx.delays_for_focus(focus)
```

### Applications
- General purpose 2D imaging
- Vascular imaging
- Musculoskeletal imaging
- Obstetrics

---

## 2. Convex (1D) Transducer

### Description
A convex array has a single row of elements arranged along a curved arc. The curvature provides a wider field of view compared to linear arrays, making it ideal for deep tissue imaging.

### Key Features
- Single row of elements arranged on curved surface
- Radius of curvature (ROC) parameter controls degree of curvature
- Wider field of view than linear arrays
- Better for deep tissue imaging
- Automatic position generation based on ROC

### Usage
```python
from transducer import Transducer

tx = Transducer(
    n_cols=64,               # Number of elements
    n_rows=1,                # Single row
    pitch=0.0004,            # Element spacing (m)
    roc=0.05,                # 50mm radius of curvature
    center_freq=3.5e6        # Center frequency (Hz)
)

print(f"ROC: {tx.roc}m")
print(f"Element positions vary in z: {tx.element_positions[:, 2].min():.4f} to {tx.element_positions[:, 2].max():.4f}m")

# Compute focusing delays
focus = (0.0, 0.0, 0.05)  # (x, y, z) in meters
delays = tx.delays_for_focus(focus)
```

### ROC Parameter Guidelines
- **Small ROC (20-40mm)**: Tighter curve, wider angle, better for shallow imaging
- **Medium ROC (40-60mm)**: Standard abdominal imaging
- **Large ROC (60-100mm)**: Gentler curve, narrower angle
- **ROC=0 or None**: Linear array (no curvature)

### Applications
- Abdominal imaging
- Cardiac imaging
- Deep tissue imaging
- Obstetrics (deep field of view)

---

## 3. 1.5D Transducer

### Description
A 1.5D array has multiple rows (typically 3-7, but can be more) in the elevation direction. Rows can have different heights, allowing for electronic elevation focusing while maintaining good sensitivity.

### Key Features
- Multiple rows (3-7 typical, but flexible)
- Variable row heights for optimized beam profile
- Electronic elevation focusing (no mechanical lens needed)
- Fewer elements than full 2D matrix (cost-effective)
- Good compromise between 1D and 2D arrays

### Usage

**Method 1: Modern interface with row_heights**
```python
from transducer import Transducer

# Variable row heights (Gaussian-like profile)
row_heights = [0.0003, 0.0004, 0.0005, 0.0004, 0.0003]  # meters (5 rows)

tx = Transducer(
    n_cols=32,               # Elements per row
    row_heights=row_heights, # n_rows inferred from array length (5)
    pitch=0.0003,            # Lateral pitch
    row_pitch=0.0004,        # Elevation pitch
    roc=0,                   # Linear (no curvature)
    center_freq=5e6
)
```

**Method 2: Specify n_rows for uniform heights**
```python
# All rows will have uniform default height (0.4mm)
tx = Transducer(
    n_cols=32,
    n_rows=7,                # 7 rows with uniform heights
    pitch=0.0003,
    row_pitch=0.0004,
    roc=0,
    center_freq=5e6
)
```

**Method 3: Legacy interface (backward compatible)**
```python
tx = Transducer(
    n_elements_per_row=32,
    n_rows=5,
    row_heights=row_heights,
    pitch=0.0003,
    row_pitch=0.0004,
    center_freq=5e6
)
```

**Using the transducer:**
```python
# Compute 3D focusing delays
focus = (0.0, 0.001, 0.03)  # (x, y, z) in meters
delays = tx.delays_for_focus(focus)

# Get element row/column
row, col = tx.get_element_row_col(element_idx=42)
```

### Element Numbering
Elements are numbered sequentially across rows:
- Elements 0 to (n_elements_per_row - 1): Row 0
- Elements n_elements_per_row to (2 × n_elements_per_row - 1): Row 1
- And so on...

### Design Considerations

**Row Heights:**
- **Uniform**: All rows same height → simple, uniform sensitivity
- **Gaussian**: Center rows taller → better elevation focusing
- **Tapered**: Gradually varying heights → reduced side lobes

**Number of Rows:**
- 3 rows: Minimal elevation control, compact
- 5 rows: Good elevation focusing, standard choice (typical)
- 7 rows: Excellent elevation control, more complex (typical)
- 10+ rows: Advanced elevation control (supported, less common)

### Applications
- Cardiac imaging (improved elevation resolution)
- Abdominal imaging
- Enhanced 2D imaging with elevation control
- Cost-effective alternative to full 2D matrix

### Example: Variable Row Heights
```python
import numpy as np

# Gaussian profile (emphasis on center rows)
# n_rows is derived from the length of row_heights array
sigma = 1.5
row_indices = np.arange(5) - 2  # 5 rows: -2, -1, 0, 1, 2
gaussian_weights = np.exp(-row_indices**2 / (2 * sigma**2))
gaussian_weights /= gaussian_weights.max()

# Scale to physical heights
base_height = 0.0003  # 0.3mm
max_height = 0.0005   # 0.5mm
row_heights = base_height + (max_height - base_height) * gaussian_weights

# n_rows is automatically 5 (from array length)
tx = Transducer(
    n_elements_per_row=64,
    row_heights=row_heights  # [0.3, 0.382, 0.5, 0.382, 0.3] mm
)

# For more rows, just create a longer array
n_rows_large = 10
row_heights_large = np.linspace(0.0003, 0.0005, n_rows_large)  # n_rows_large rows
tx_large = Transducer(
    n_elements_per_row=64,
    row_heights=row_heights_large  # n_rows = n_rows_large automatically
)
```

---

## 4. 1.5D Convex Transducer (NEW!)

### Description
A 1.5D convex array combines the benefits of convex (curved) arrays with multi-row elevation control. All rows follow the same curved arc in the lateral direction, providing a wider field of view while maintaining electronic elevation focusing.

### Key Features
- Multiple rows (3-7 typical) on curved surface
- Lateral curvature for wider field of view (controlled by ROC)
- Variable row heights for optimized beam profile
- Electronic elevation focusing
- Combines benefits of convex and 1.5D arrays

### Usage
```python
from transducer import Transducer

# Variable row heights with convex curvature
row_heights = [0.0003, 0.0004, 0.0005, 0.0004, 0.0003]  # meters

tx = Transducer(
    n_cols=64,               # Elements per row
    row_heights=row_heights, # 5 rows with variable heights
    pitch=0.0003,            # Lateral pitch
    row_pitch=0.0004,        # Elevation pitch
    roc=0.05,                # 50mm radius of curvature (convex!)
    center_freq=5e6
)

# All rows follow the same curved arc
print(f"ROC: {tx.roc}m")
z_range = tx.element_positions[:, 2]
print(f"Z range: {z_range.min():.4f} to {z_range.max():.4f}m (curved)")
```

### ROC Guidelines for 1.5D Convex
- **40-60mm ROC**: Standard for cardiac imaging
- **60-80mm ROC**: Gentler curve for abdominal imaging
- Curvature applies to lateral direction; rows are straight in elevation

### Applications
- Cardiac imaging with wide FOV and elevation control
- Enhanced abdominal imaging
- Deep tissue imaging with multi-row capability
- Applications requiring both wide FOV and elevation focusing

---

## 5. 2D Matrix Transducer

### Description
A 2D matrix array has a large grid of elements (tens to hundreds of rows and columns) allowing full 3D electronic beam steering and focusing without mechanical scanning.

### Key Features
- Large 2D element grid (typical: 16×16 to 128×128)
- Uniform element dimensions
- Full 3D electronic beam steering
- 3D volumetric imaging
- No mechanical scanning required
- Flat surface (roc=0)

### Usage

**Modern interface (preferred)**
```python
from transducer import Transducer

tx = Transducer(
    n_cols=32,                # Elements in X (lateral)
    n_rows=32,                # Elements in Y (elevation)
    pitch=0.0003,             # Spacing in both directions (m)
    element_width=0.00028,    # Element width (m)
    element_height=0.00028,   # Element height (m)
    roc=0,                    # Linear (no curvature)
    center_freq=5e6           # Center frequency (Hz)
)

# Total elements
print(f"Total elements: {tx.n_elements}")  # 32 × 32 = 1024

# 3D focusing
focus = (0.002, 0.001, 0.025)  # (x, y, z) in meters
delays_focus = tx.delays_for_focus(focus)

# 3D beam steering
steering = (np.deg2rad(15), np.deg2rad(10))  # (θx, θy) in radians
delays_steer = tx.delays_for_steering_3d(steering)

# Get element row/column
row, col = tx.get_element_row_col(element_idx=500)
```

**Legacy interface (backward compatible)**
```python
tx = Transducer(
    n_elements_x=32,
    n_elements_y=32,
    pitch=0.0003,
    element_width=0.00028,
    element_height=0.00028,
    center_freq=5e6
)
```

### Element Numbering
Elements are numbered in row-major order:
- Elements 0 to (n_cols - 1): Row 0
- Elements n_cols to (2 × n_cols - 1): Row 1
- Element at position (row, col): index = row × n_cols + col

### Array Sizes

| Configuration | Elements | Typical Use |
|--------------|----------|-------------|
| 16 × 16 | 256 | Small volumetric probe |
| 32 × 32 | 1,024 | Standard 3D imaging |
| 64 × 64 | 4,096 | High-resolution 3D |
| 128 × 128 | 16,384 | Research/advanced systems |

### 3D Beam Control

**Focusing:**
```python
# Focus at specific 3D point
focus_point = (x, y, z)  # meters
delays = tx.delays_for_focus(focus_point)
```

**Steering:**
```python
# Steer beam at angles
angles = (theta_x, theta_y)  # radians
delays = tx.delays_for_steering_3d(angles)
```

### Apodization
2D apodization recommended for side lobe reduction:
```python
import numpy as np

# 2D Hanning window
apod_x = np.hanning(tx.n_elements_x)
apod_y = np.hanning(tx.n_elements_y)
apod_2d = np.outer(apod_y, apod_x).flatten()

# Apply to signal
weighted_signal = signal * apod_2d
```

### Applications
- Real-time 3D/4D cardiac imaging
- Volumetric blood flow imaging
- Intracardiac echocardiography (ICE)
- Transesophageal echocardiography (TEE)
- 3D fetal imaging
- Interventional guidance

---

## 6. 2D Matrix Convex Transducer (NEW!)

### Description
A 2D matrix convex array combines large-scale 2D element grids with lateral curvature. All rows follow the same curved arc, providing full 3D electronic beam control with a wider field of view than flat matrix arrays.

### Key Features
- Large 2D element grid on curved surface (typical: 16×16 to 128×128)
- Lateral curvature for wider field of view (controlled by ROC)
- Uniform element dimensions
- Full 3D electronic beam steering and focusing
- Maximum flexibility in beam control

### Usage
```python
from transducer import Transducer

tx = Transducer(
    n_cols=32,                # Elements in X
    n_rows=32,                # Elements in Y
    pitch=0.0003,             # 0.3mm pitch (both directions)
    element_width=0.00028,    # Element width
    element_height=0.00028,   # Element height
    roc=0.06,                 # 60mm radius of curvature (convex!)
    center_freq=5e6
)

# All rows follow the same curved arc
print(f"Total elements: {tx.n_elements}")  # 32 × 32 = 1024
print(f"ROC: {tx.roc}m")
z_range = tx.element_positions[:, 2]
print(f"Z range: {z_range.min():.4f} to {z_range.max():.4f}m (curved)")
```

### ROC Guidelines for 2D Convex
- **50-70mm ROC**: Standard for cardiac volumetric imaging
- **70-100mm ROC**: Gentler curve for deep tissue 3D imaging
- Larger arrays may require larger ROC to maintain reasonable angular span

### Applications
- 3D/4D cardiac imaging with wide FOV
- Volumetric imaging with curved aperture
- Deep tissue 3D imaging
- Applications requiring maximum beam control flexibility
- Real-time volumetric blood flow imaging

---

## Comparison Table

| Feature | Linear (1D) | Convex (1D) | 1.5D Linear | 1.5D Convex | 2D Linear | 2D Convex |
|---------|-------------|-------------|-------------|-------------|-----------|-----------|
| **Interface** | n_cols=N, n_rows=1, roc=0 | n_cols=N, n_rows=1, roc>0 | n_cols=N, n_rows=3-7, roc=0 | n_cols=N, n_rows=3-7, roc>0 | n_cols=N, n_rows=N, roc=0 | n_cols=N, n_rows=N, roc>0 |
| **Elements** | 64-256 | 64-256 | 96-448 total | 96-448 total | 256-16,384 | 256-16,384 |
| **Geometry** | Flat | Curved | Flat | Curved | Flat | Curved |
| **ROC** | 0 (linear) | 20-100mm | 0 (linear) | 40-80mm | 0 (linear) | 50-100mm |
| **Focusing** | X-Z plane only | X-Z plane only | X-Z with elevation | X-Z with elevation | Full 3D | Full 3D |
| **Steering** | Azimuth only | Azimuth only | Azimuth + limited elevation | Azimuth + limited elevation | Full 3D | Full 3D |
| **Elevation** | Fixed (lens) | Fixed (lens) | Electronic | Electronic | Electronic | Electronic |
| **Field of View** | Rectangular | Fan-shaped (wider) | Rectangular | Fan-shaped (wider) | Pyramidal | Pyramidal (wider) |
| **Complexity** | Low | Low | Medium | Medium | High | High |
| **Cost** | Low | Low-Medium | Medium | Medium | High | High |
| **Imaging** | 2D | 2D (wider FOV) | Enhanced 2D | Enhanced 2D (wider FOV) | 3D/4D | 3D/4D (wider FOV) |
| **Frame Rate** | High | High | High | High | Lower (volume) | Lower (volume) |
| **Applications** | General 2D | Deep tissue, cardiac | Cardiac, enhanced 2D | Cardiac with wide FOV | Volumetric, 4D | Volumetric with wide FOV |

---

## Quick Reference

### Creating Different Array Types

```python
from transducer import Transducer

# 1D Linear
tx_linear = Transducer(n_cols=64, n_rows=1, pitch=0.0003, roc=0)

# 1D Convex
tx_convex = Transducer(n_cols=64, n_rows=1, pitch=0.0004, roc=0.05)

# 1.5D Linear
tx_1p5d_linear = Transducer(n_cols=32, n_rows=5, pitch=0.0003, row_pitch=0.0004, roc=0)

# 1.5D Convex (NEW!)
tx_1p5d_convex = Transducer(n_cols=32, n_rows=5, pitch=0.0003, row_pitch=0.0004, roc=0.05)

# 2D Matrix Linear
tx_2d_linear = Transducer(n_cols=32, n_rows=32, pitch=0.0003, roc=0)

# 2D Matrix Convex (NEW!)
tx_2d_convex = Transducer(n_cols=32, n_rows=32, pitch=0.0003, roc=0.06)
```

---

## BLI Mask Generation

All transducer types support Band-Limited Interpolation (BLI) for accurate source injection:

```python
from grid import Grid

# Create grid
grid = Grid(nx=128, ny=128, nz=256, dx=1e-4)

# Generate BLI masks for all elements
masks = tx.create_all_element_masks(
    grid,
    n_points_x=5,      # Sample points in X
    n_points_y=5,      # Sample points in Y
    kernel_radius=3,   # Sinc kernel radius
    tolerance=1e-3,    # Weight threshold
    staggered=False    # True for velocity components
)

# Each mask contains (indices, weights)
indices, weights = masks[element_idx]
```

### Memory Efficiency
BLI masks use sparse representation:
- Typical: ~700-2000 non-zero entries per element
- Dense grid: 128×128×256 = 2M entries
- Memory savings: ~99%

---

## Testing

Comprehensive test suites are available:

```bash
# Run comprehensive transducer demonstration
python examples/transducer_demo.py

# Test 1.5D transducer
python examples/test_1p5d_transducer.py

# Test 2D matrix transducer
python examples/test_matrix_transducer.py

# Test basic BLI implementation
python examples/test_corrected_bli.py
```

---

## Examples

Example simulation scripts demonstrate usage:

```bash
# Comprehensive demo of all transducer types (NEW!)
python examples/transducer_demo.py

# 1.5D transducer simulation
python examples/run_1p5d_probe.py

# 2D matrix transducer simulation
python examples/run_2d_matrix_probe.py

# Linear transducer simulation
python examples/run_linear_probe.py

# Convex array simulation
python examples/run_convex_probe.py
```

---

## References

1. Band-Limited Interpolation: https://doi.org/10.1121/1.5116132
2. Szabo, T. L. (2004). *Diagnostic Ultrasound Imaging: Inside Out*
3. Jensen, J. A. (1996). *Field: A Program for Simulating Ultrasound Systems*

---

## Implementation Notes

### Coordinate System
- **X axis**: Lateral (left-right)
- **Y axis**: Elevation (up-down)
- **Z axis**: Axial (depth)
- **Transducer surface**: Z = 0

### Element Positions
All transducer types store element positions in `element_positions`:
- Shape: (n_elements, 3) for (x, y, z) coordinates
- Centered at origin in the lateral/elevation plane (x = 0, y = 0)
- Transducer surface lies at z = 0 by convention

### Compatibility
- 1D transducers use only X positions (Y = 0, Z = 0)
- 1.5D and 2D use full (X, Y) positioning with Z = 0 at the surface (unless otherwise specified)
- All methods work with existing BLI implementation
- Backward compatible with existing code
