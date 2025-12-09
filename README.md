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
- Supports linear, convex, and matrix array transducers

## Notes:
- The examples are intentionally small so they can run on limited GPUs. Increase sizes for realistic runs.
- For 3D simulations, use small grids (e.g., 64³) and fp16 medium maps to reduce memory usage.

## Run an example:
```bash
# 2D linear probe (nx=128, ny=1, nz=256)
python examples/run_linear_probe.py

# 2D convex probe
python examples/run_convex_probe.py

# 3D matrix probe (nx=64, ny=64, nz=64)
python examples/run_matrix_probe.py
```

# Dolphine
FDTD simulator for medical ultrasound imaging with full 3D support
