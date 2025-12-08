"""Matrix probe 3D demo (small grid) showing slice visualizations.
This example uses a very small 3D grid to keep memory low.
"""
import os
import sys
import time
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

    # small 3D grid
    dx = 2e-4
    grid = Grid(nx=64, ny=64, nz=64, dx=dx)

    med = Medium((64, 64, 64), dtype=np.float16)
    # Add a small spherical inclusion for visualization
    cx, cy, cz = 32, 32, 32
    for i in range(64):
        for j in range(64):
            for k in range(64):
                dist = np.sqrt((i-cx)**2 + (j-cy)**2 + (k-cz)**2)
                if dist < 8:
                    med.c[i, j, k] = 1600.0

    # small 2D transducer grid placeholder (use central elements)
    tx = Transducer(n_elements=16, pitch=0.0005, center_freq=3e6)

    fs = 1.0 / grid.dt
    sig, sr = tone_burst(center_freq=tx.center_freq, sampling_rate=int(fs), n_cycles=2)

    # Map transducer elements to grid
    elem_idx = tx.map_to_grid(grid, z0=0.0)
    src_positions = []
    for (ix, iy, iz) in elem_idx[::2]:  # use every 2nd element
        src_positions.append((iz * grid.ny + iy) * grid.nx + ix)

    # Get medium properties
    rho, c, alpha = med.rho, med.c, med.alpha

    solver = SolverCore(grid, (rho, c, alpha), dtype=np.float16)

    n_steps = 150
    t0 = time.time()
    elapsed = solver.run(n_steps, source_positions=src_positions, source_signal=sig)
    tot = time.time() - t0

    print(f"3D simulation elapsed (kernel): {elapsed:.3f}s, total wall: {tot:.3f}s")

    # Visualize pressure field - XZ slice at center Y
    p_np = solver.p.get()
    p_3d = p_np.reshape(grid.nx, grid.ny, grid.nz)
    
    # XZ slice at center Y
    xz_slice = np.abs(p_3d[:, grid.ny//2, :]).T
    env = envelope(xz_slice)
    db = log_compress(env)
    out_path = os.path.join(out_dir, 'matrix_probe_xz_slice.png')
    save_mip(db, out_path, extent=[-grid.nx*dx*1e3/2, grid.nx*dx*1e3/2, grid.nz*dx*1e3, 0])
    print(f"Saved XZ slice image to {out_path}")

    # YZ slice at center X
    yz_slice = np.abs(p_3d[grid.nx//2, :, :]).T
    env_yz = envelope(yz_slice)
    db_yz = log_compress(env_yz)
    out_path_yz = os.path.join(out_dir, 'matrix_probe_yz_slice.png')
    save_mip(db_yz, out_path_yz, extent=[-grid.ny*dx*1e3/2, grid.ny*dx*1e3/2, grid.nz*dx*1e3, 0])
    print(f"Saved YZ slice image to {out_path_yz}")

    print("Matrix probe demo complete: full 3D kernels are now implemented.")


if __name__ == '__main__':
    main()
