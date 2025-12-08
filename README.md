Dolphine — FDTD Ultrasound Simulation Framework

Lightweight, modular FDTD-based ultrasound RF data generator using Python + CuPy (RawKernel).

Structure:
- `grid.py`, `medium.py`, `transducer.py`, `source.py`, `moving_window.py`, `solver_core.py`
- `utils/visualizer.py` for envelope detection, log compression and plotting
- `examples/` contains three probe demos and saves beampattern images to `results/`

Notes:
- Designed for GPU acceleration via CuPy RawKernel. Mixed precision (fp16 host maps -> fp32 compute) is used.
- The examples are intentionally small so they can run on limited GPUs. Increase sizes for realistic runs.

Run an example:
```
python examples/run_linear_probe.py
```
# Dolphine
fdtd simulator for medical ultrasound image
