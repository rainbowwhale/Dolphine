"""Convex (curved) probe example: build arc of elements and compute delays.
This demo generates a small synthetic run and a beam visualization tip.
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

    dx = 1e-4
    grid = Grid(nx=128, ny=1, nz=256, dx=dx)

    med = Medium((128, 1, 256), dtype=np.float16)

    # Create convex transducer with native support
    tx = Transducer(n_elements=32, pitch=0.0004, center_freq=3e6, radius=0.045, angle_span=60.0)

    # compute delays to a focal distance
    focus = (0.0, 0.05)
    delays = tx.delays_for_focus(focus)

    fs = 1.0 / grid.dt
    sig, sr = tone_burst(center_freq=tx.center_freq, sampling_rate=int(fs), n_cycles=2)

    elem_idx = tx.map_to_grid(grid, z0=0.0)
    src_positions = [grid.to_linear_index(ix, iy, iz) for (ix, iy, iz) in elem_idx[::4]]

    solver = SolverCore(grid, (med.rho, med.c, med.alpha), dtype=np.float16)

    elapsed = solver.run(180, source_positions=src_positions, source_signal=sig)
    print(f"Convex simulation finished in {elapsed:.3f}s")

    p_np = solver.p.get()
    img = np.max(np.abs(p_np.reshape(grid.nx, grid.ny, grid.nz)), axis=1).T
    env = envelope(img)
    db = log_compress(env)
    out_path = os.path.join(out_dir, 'convex_probe_mip.png')
    save_mip(db, out_path, extent=[-grid.nx*dx*1e3/2, grid.nx*dx*1e3/2, grid.nz*dx*1e3, 0])
    print(f"Saved convex beam image to {out_path}")


if __name__ == '__main__':
    main()
