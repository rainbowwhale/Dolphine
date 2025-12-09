# Dolphine — FDTD Ultrasound Simulation Framework

Lightweight, modular FDTD-based ultrasound RF data generator using Python + CuPy (RawKernel).

## Structure:
- `grid.py`, `medium.py`, `transducer.py`, `source.py`, `moving_window.py`, `solver_core.py`
- `utils/visualizer.py` for envelope detection, log compression and plotting
- `examples/` contains three probe demos and saves beampattern images to `results/`

## Features:
- **Full 3D FDTD solver** with staggered-grid acoustic wave propagation
- **2D compatibility** when ny=1 for faster 2D-only simulations
- **GPU acceleration** via CuPy RawKernel with mixed precision (fp16 host maps → fp32 compute)
- Supports multiple transducer geometries:
  - **Single-row linear arrays** (traditional 1D arrays)
  - **Multi-row arrays** (1.5D/1.75D arrays with elevation control)
  - **2D matrix arrays** (full 3D beam steering)
  - **Convex/curved arrays** (custom element positioning)
- **Element height information** for physical transducer dimensions
- **3D element positioning** (x, y, z) with backward compatible 2D focus points

## Notes:
- The examples are intentionally small so they can run on limited GPUs. Increase sizes for realistic runs.
- For 3D simulations, use small grids (e.g., 64³) and fp16 medium maps to reduce memory usage.

## CUDA Setup (Multiple CUDA Versions):
If you have multiple CUDA versions installed on your system, CuPy may have difficulty finding the correct CUDA libraries. This project includes automatic CUDA path detection in `src/cuda_setup.py` that runs before importing CuPy.

The automatic detection will:
1. Check if `CUDA_PATH` or `CUDA_HOME` environment variables are already set
2. Search common CUDA installation directories (`/usr/local/cuda*`, `/opt/cuda*`)
3. Select the newest CUDA version available
4. Set appropriate environment variables (`CUDA_PATH`, `CUDA_HOME`, `LD_LIBRARY_PATH`)

### Manual Override:
If you want to use a specific CUDA version, set the `CUDA_PATH` environment variable before running your script:

```bash
export CUDA_PATH=/usr/local/cuda-11.8
python examples/run_linear_probe.py
```

Or set it inline:
```bash
CUDA_PATH=/usr/local/cuda-11.8 python examples/run_linear_probe.py
```

## Run an example:
```bash
# 2D linear probe (nx=128, ny=1, nz=256)
python examples/run_linear_probe.py

# 2D convex probe
python examples/run_convex_probe.py

# 3D matrix probe (nx=64, ny=64, nz=64)
python examples/run_matrix_probe.py

# Multi-row transducer demonstration
python examples/run_multirow_demo.py
```

## Transducer Types:
The `Transducer` class now supports various array configurations:
- **Single-row** (`n_rows=1`): Traditional linear arrays with elements along x-axis
- **Multi-row** (`n_rows>1`): 1.5D or 1.75D arrays with elevation control
  - Supports per-row element heights via `row_heights` parameter (e.g., `[1mm, 3mm, 5mm, 3mm, 1mm]`)
- **Convex/curved** (`radius!=None`): Curved arrays with elements along an arc
  - Specify `radius` (curvature radius in meters) and `angle_span` (angular span in degrees)
- **2D matrix**: Full 2D element grid for volumetric imaging
- **Custom geometry**: Override `element_positions` for specialized arrays

All transducers store 3D element positions (x, y, z) and include `element_height` attribute for physical dimensions.

### Example: Multi-row array with varying heights
```python
# 5-row array where center row is tallest (1, 3, 5, 3, 1 mm)
tx = Transducer(
    n_elements=100, 
    n_rows=5, 
    row_heights=[0.001, 0.003, 0.005, 0.003, 0.001]
)
```

### Example: Convex array
```python
# Convex array with 45mm radius and 60 degree span
tx = Transducer(
    n_elements=32,
    center_freq=3e6,
    radius=0.045,
    angle_span=60.0
)
```

# Dolphine
FDTD simulator for medical ultrasound imaging with full 3D support
