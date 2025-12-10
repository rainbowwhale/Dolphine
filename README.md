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

Notes:
- The examples are intentionally small so they can run on limited GPUs. Increase sizes for realistic runs.

Run an example:
```bash
python examples/run_linear_probe.py
```

Test transducer mask generation:
```bash
python examples/test_transducer_mask.py
```

Demo transducer masks:
```bash
python examples/demo_transducer_mask.py
```

# Dolphine
fdtd simulator for medical ultrasound image
