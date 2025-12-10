"""Example: run a tiny linear-probe simulation and save beam intensity image.

This demo is intentionally small so it can run on limited GPUs/CI.
"""
import time
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from medium import Medium
from transducer import Transducer
from source import tone_burst
from solver_core import SolverCore
from utils.visualizer import envelope, log_compress, save_mip


def build_phantom(shape, dx):
    med = Medium(shape, dtype=np.float16, default_c=1540.0)
    # simple inclusion
    cx = shape[0] // 2
    cz = int(0.03 / dx)
    med.set_region((slice(cx-5, cx+5), slice(None), slice(cz-5, cz+5)), c=1600.0)
    return med


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    # grid: (nx, ny, nz) with ny=1 for 2D slice
    dx = 1e-4
    grid = Grid(nx=128, ny=1, nz=256, dx=dx)

    med = build_phantom((128, 1, 256), dx)

    # transducer
    tx = Transducer(n_elements=32, pitch=0.0003, center_freq=5e6)
    delays = tx.delays_for_focus((0.0, 0.03))

    # source waveform
    fs = 1.0 / grid.dt
    sig, sr = tone_burst(center_freq=5e6, sampling_rate=int(fs), n_cycles=2)

    # pick element center grid indices for injection
    elem_idx = tx.map_to_grid(grid, z0=0.0)
    # convert (ix,iy,iz) to linear indices for solver's flattened layout (nx x nz)
    src_positions = []
    for (ix, _, iz) in elem_idx[::4]:  # use every 4th element to reduce source count
        src_positions.append(iz * grid.nx + ix)

    # get ROI from medium for small window (use whole for simplicity)
    rho, c, alpha = med.rho, med.c, med.alpha

    solver = SolverCore(grid, (rho, c, alpha), dtype=np.float16)

    n_steps = 200
    t0 = time.time()
    elapsed = solver.run(n_steps, source_positions=src_positions, source_signal=sig)
    tot = time.time() - t0

    print(f"Simulation elapsed (kernel): {elapsed:.3f}s, total wall: {tot:.3f}s")

    # simple RF gather: use pressure field snapshot and compute envelope
    p_np = solver.p.get()
    # collapse into (lateral x depth) frame for visualization (max across y)
    img = np.max(np.abs(p_np.reshape(grid.nx, grid.nz).T), axis=2) if False else np.abs(p_np.reshape(grid.nx, grid.nz).T)
    # compute envelope along time/depth axis; here depth acts like time
    env = envelope(img)
    db = log_compress(env)
    out_path = os.path.join(out_dir, 'linear_probe_mip.png')
    save_mip(db, out_path, extent=[-grid.nx*dx*1e3/2, grid.nx*dx*1e3/2, grid.nz*dx*1e3, 0])
    print(f"Saved beam image to {out_path}")


if __name__ == '__main__':
    main()
