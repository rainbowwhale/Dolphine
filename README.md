# Dolphine — FDTD Ultrasound Simulation Framework

Lightweight, modular FDTD-based ultrasound RF data generator using Python + CuPy (RawKernel).

## Structure

- `src/grid.py` — Simulation grid and time-step manager
- `src/medium.py` — Acoustic medium properties
- `src/transducer.py` — Transducer array with acoustic lens support
- `src/source.py` — Source signal generation
- `src/moving_window.py` — Moving window support
- `src/solver_core.py` — FDTD solver core
- `src/utils/` — Utilities (visualization, etc.)
- `examples/` — Example scripts
- `docs/` — Documentation

## Features

### Transducer Module

The transducer module provides comprehensive ultrasound array modeling:

- **N × M array structure**: Flexible `n_cols` × `n_rows` configuration
- **Element properties**: 3D position, normal angle, and size for each element
- **Array curvature**: Radius of curvature (ROC) for convex arrays
- **Multi-layer acoustic lens**: 
  - Elevational ROC (convex or concave)
  - Multiple layer support
  - Per-layer material properties
- **Array dimensions**: Width, height, min/max dimension properties
- **Plotting functions**: 3D array view, 2D layout, lens cross-section
- **Band-limited interpolation (BLI)**: Accurate source injection

#### Quick Example

```python
from transducer import Transducer, AcousticLens, LensLayer

# Create a 64×8 element transducer array
tx = Transducer(
    n_cols=64,           # 64 columns (lateral)
    n_rows=8,            # 8 rows (elevation)
    pitch=0.0003,        # 0.3mm pitch
    roc=0,               # 0 = flat, >0 = curved
    center_freq=5e6      # 5 MHz
)

# Access element properties
pos = tx.get_element_position(0)      # (x, y, z) in meters
angle = tx.get_element_angle(0)       # (θx, θy) in radians
size = tx.get_element_size()          # (width, height) in meters

# Beamforming
delays = tx.delays_for_focus((0.0, 0.0, 0.03))  # Focus at 30mm depth

# Plot the array
tx.plot_array()
tx.plot_array_2d()
```

#### Acoustic Lens Support

```python
# Create multi-layer acoustic lens
lens = AcousticLens()
lens.add_layer(LensLayer(elevational_roc=0.020, max_thickness=0.001, name="Focus"))
lens.add_layer(LensLayer(elevational_roc=-0.030, max_thickness=0.0006, name="Diverging"))
lens.add_layer(LensLayer(elevational_roc=0, max_thickness=0.0004, name="Matching"))

# Create transducer with lens
tx = Transducer(n_cols=64, n_rows=5, lens=lens)

# Plot lens cross-section
tx.plot_lens()
```

### Other Features

- GPU acceleration via CuPy RawKernel with mixed precision (fp16 host maps → fp32 compute)
- Band-limited interpolation for transducer masks ([DOI: 10.1121/1.5116132](https://doi.org/10.1121/1.5116132))
- Sparse BLI mask representation for memory efficiency

## Installation

```bash
pip install numpy matplotlib scipy
# Optional for GPU acceleration:
pip install cupy
```

## Run Examples

```bash
# Comprehensive transducer demonstration
python examples/transducer_demo.py
```

## Documentation

See `docs/TRANSDUCER_MODULE.md` for detailed transducer module documentation.

## Notes

- Examples are intentionally small to run on limited GPUs. Increase sizes for realistic simulations.
