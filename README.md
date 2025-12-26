Dolphine — FDTD Ultrasound Simulation Framework

Lightweight, modular FDTD-based ultrasound RF data generator using Python + CuPy (RawKernel).

Structure:
- `grid.py`, `medium.py`, `transducer.py`, `source.py`, `moving_window.py`, `solver_core.py`
- `utils/visualizer.py` for envelope detection, log compression and plotting
- `examples/` contains probe demos and saves beampattern images to `results/`

Features:
- GPU acceleration via CuPy RawKernel with mixed precision (fp16 host maps -> fp32 compute)
- Band-limited interpolation for transducer masks (reference: https://doi.org/10.1121/1.5116132)
  - Rectangular element geometry with configurable dimensions
  - Surface point sampling for accurate element representation
  - Support for both normal and staggered grid configurations
  - Sinc-based interpolation for smooth spatial distribution
- Multiple transducer array types:
  - **Linear (1D)**: Single row of elements for 2D imaging
  - **1.5D**: Multiple rows (typically 3-7) with variable heights for elevation focusing
  - **2D Matrix**: Large 2D grids (10s-100s of elements) for 3D volumetric imaging

Notes:
- The examples are intentionally small so they can run on limited GPUs. Increase sizes for realistic runs.

Run examples:
```bash
# Linear (1D) transducer
python examples/run_linear_probe.py

# 1.5D transducer with elevation focusing
python examples/run_1p5d_probe.py

# 2D matrix transducer for 3D imaging
python examples/run_2d_matrix_probe.py

# Convex array
python examples/run_convex_probe.py
```

Test implementations:
```bash
# Test BLI implementation
python examples/test_corrected_bli.py

# Test 1.5D transducer
python examples/test_1p5d_transducer.py

# Test 2D matrix transducer
python examples/test_matrix_transducer.py
```

Documentation:
- See `docs/TRANSDUCER_TYPES.md` for detailed transducer array documentation
- See `docs/TRANSDUCER_MASK_API.md` for BLI mask generation API
- See `docs/BLI_OVERHAUL_SUMMARY.md` for BLI implementation details

# Dolphine
fdtd simulator for medical ultrasound image
