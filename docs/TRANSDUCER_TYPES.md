# Transducer Types in Dolphine

This document describes the different transducer array types supported in Dolphine and their use cases.

## Overview

Dolphine supports three types of ultrasound transducer arrays:

1. **Linear (1D) Transducer** - Single row of elements for 2D imaging
2. **1.5D Transducer** - Multiple rows (typically 3-7 rows, though more rows are supported) with variable heights for elevation focusing
3. **2D Matrix Transducer** - Large 2D grid (10s-100s of elements) for 3D volumetric imaging

## 1. Linear (1D) Transducer

### Description
The basic linear array has a single row of elements arranged along the lateral (x) axis. This is the most common transducer type for 2D ultrasound imaging.

### Key Features
- Single row of elements (typical: 64-256 elements)
- Electronic beam steering and focusing in azimuth (x-z plane)
- Fixed elevation focus (mechanical lens)
- Compact and cost-effective

### Usage
```python
from transducer import Transducer

tx = Transducer(
    n_elements=128,           # Number of elements
    pitch=0.0003,             # Element spacing (m)
    element_width=0.00028,    # Element width (m)
    element_height=0.010,     # Element height (m) - fixed
    center_freq=5e6           # Center frequency (Hz)
)

# Compute focusing delays
focus = (0.0, 0.03)  # (x, z) in meters
delays = tx.delays_for_focus(focus)
```

### Applications
- General purpose 2D imaging
- Vascular imaging
- Musculoskeletal imaging
- Obstetrics

---

## 2. 1.5D Transducer

### Description
A 1.5D array has multiple rows (typically 3-7, but can be more) in the elevation direction. Rows can have different heights, allowing for electronic elevation focusing while maintaining good sensitivity.

### Key Features
- Multiple rows (3-7 typical, but flexible)
- Variable row heights for optimized beam profile
- Electronic elevation focusing (no mechanical lens needed)
- Fewer elements than full 2D matrix (cost-effective)
- Good compromise between 1D and 2D arrays

### Usage

**Method 1: Provide array of heights (n_rows inferred/validated automatically)**
```python
from transducer import Transducer1p5D

# If n_rows is not provided, it is inferred from len(row_heights).
# If n_rows is provided, it must match len(row_heights) or an error is raised.
row_heights = [0.0003, 0.0004, 0.0005, 0.0004, 0.0003]  # meters (5 rows)

tx = Transducer1p5D(
    n_elements_per_row=32,
    row_heights=row_heights,  # n_rows inferred as 5 unless explicitly provided
    pitch=0.0003,
    row_pitch=0.0004,
    center_freq=5e6
)
```

**Method 2: Specify n_rows for uniform heights**
```python
# All rows will have uniform default height (0.4mm)
tx = Transducer1p5D(
    n_elements_per_row=32,
    n_rows=7,                 # 7 rows with uniform heights
    pitch=0.0003,
    row_pitch=0.0004,
    center_freq=5e6
)
```

**Method 3: Single height value with n_rows**
```python
# All rows will have the specified height
tx = Transducer1p5D(
    n_elements_per_row=32,
    n_rows=5,
    row_heights=0.0005,       # Single value: all rows get 0.5mm height
    pitch=0.0003,
    row_pitch=0.0004,
    center_freq=5e6
)
```

**Using the transducer:**
```python
# Compute 3D focusing delays
focus = (0.0, 0.001, 0.03)  # (x, y, z) in meters
delays = tx.delays_for_focus_3d(focus)

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
tx = Transducer1p5D(
    n_elements_per_row=64,
    row_heights=row_heights  # [0.3, 0.382, 0.5, 0.382, 0.3] mm
)

# For more rows, just create a longer array
row_heights_10 = np.linspace(0.0003, 0.0005, 10)  # 10 rows
tx_large = Transducer1p5D(
    n_elements_per_row=64,
    row_heights=row_heights_10  # n_rows = 10 automatically
)
```

---

## 3. 2D Matrix Transducer

### Description
A 2D matrix array has a large grid of elements (tens to hundreds of rows and columns) allowing full 3D electronic beam steering and focusing without mechanical scanning.

### Key Features
- Large 2D element grid (typical: 16×16 to 128×128)
- Uniform element dimensions
- Full 3D electronic beam steering
- 3D volumetric imaging
- No mechanical scanning required

### Usage
```python
from transducer import MatrixTransducer

tx = MatrixTransducer(
    n_elements_x=32,          # Elements in X (lateral)
    n_elements_y=32,          # Elements in Y (elevation)
    pitch=0.0003,             # Spacing in both directions (m)
    element_width=0.00028,    # Element width (m)
    element_height=0.00028,   # Element height (m)
    center_freq=5e6           # Center frequency (Hz)
)

# Total elements
print(f"Total elements: {tx.n_elements}")  # 32 × 32 = 1024

# 3D focusing
focus = (0.002, 0.001, 0.025)  # (x, y, z) in meters
delays_focus = tx.delays_for_focus_3d(focus)

# 3D beam steering
steering = (np.deg2rad(15), np.deg2rad(10))  # (θx, θy) in radians
delays_steer = tx.delays_for_steering_3d(steering)

# Get element row/column
row, col = tx.get_element_row_col(element_idx=500)
```

### Element Numbering
Elements are numbered in row-major order:
- Elements 0 to (n_elements_x - 1): Row 0
- Elements n_elements_x to (2 × n_elements_x - 1): Row 1
- Element at position (row, col): index = row × n_elements_x + col

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
delays = tx.delays_for_focus_3d(focus_point)
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

## Comparison Table

| Feature | Linear (1D) | 1.5D | 2D Matrix |
|---------|-------------|------|-----------|
| **Elements** | 64-256 | 96-448 (typically 3-7 rows) | 256-16,384 |
| **Focusing** | X-Z plane only | X-Z with elevation | Full 3D |
| **Steering** | Azimuth only | Azimuth + limited elevation | Full 3D |
| **Elevation** | Fixed (lens) | Electronic | Electronic |
| **Complexity** | Low | Medium | High |
| **Cost** | Low | Medium | High |
| **Imaging** | 2D | Enhanced 2D | 3D/4D |
| **Frame Rate** | High | High | Lower (volume) |
| **Applications** | General 2D | Cardiac, enhanced 2D | Volumetric, 4D |

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
# 1.5D transducer simulation
python examples/run_1p5d_probe.py

# 2D matrix transducer simulation
python examples/run_2d_matrix_probe.py

# Linear transducer simulation
python examples/run_linear_probe.py

# Convex array (specialized 1D)
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
