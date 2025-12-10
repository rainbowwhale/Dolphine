"""Matrix probe 3D demo (small grid) showing slice visualizations.
This example uses a very small 3D grid to keep memory low.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from medium import Medium
from transducer import Transducer
from source import tone_burst
from solver_core import SolverCore
from utils.visualizer import envelope, log_compress, save_mip


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    # small 3D grid (note: solver currently implemented in 2D form; this demonstrates structure)
    dx = 2e-4
    grid = Grid(nx=64, ny=64, nz=64, dx=dx)

    med = Medium((64, 64, 64), dtype=np.float16)

    # small 2D transducer grid placeholder
    tx = Transducer(n_elements=16, pitch=0.0005, center_freq=3e6)

    fs = 1.0 / grid.dt
    sig, sr = tone_burst(center_freq=tx.center_freq, sampling_rate=int(fs), n_cycles=2)

    # For 3D the SolverCore would need full 3D kernels; here we show recommended constraints
    print("Matrix probe demo: this example outlines usage; full 3D kernels are required for production.")
    print("Keep grid small (e.g., 64^3) and use fp16 medium maps to reduce memory.")


if __name__ == '__main__':
    main()
