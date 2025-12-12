"""Example: 1.5D transducer probe simulation with elevation focusing.

This demo shows how to use a 1.5D transducer array with multiple rows
for elevation focusing capabilities. The 1.5D array allows electronic
control in the elevation direction while maintaining mechanical focus
in the lateral direction.
"""
import time
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from medium import Medium
from transducer import Transducer1p5D
from source import tone_burst


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("1.5D TRANSDUCER PROBE SIMULATION")
    print("=" * 70)
    
    # Create 3D grid with moderate resolution
    # For 1.5D arrays, we need proper Y dimension for elevation control
    dx = 1.5e-4
    grid = Grid(nx=128, ny=64, nz=256, dx=dx)
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {grid.nx} × {grid.ny} × {grid.nz}")
    print(f"  Resolution: {dx*1e3:.3f}mm")
    print(f"  Physical size: {grid.nx*dx*1e3:.1f}mm × {grid.ny*dx*1e3:.1f}mm × {grid.nz*dx*1e3:.1f}mm")
    
    # Create medium
    med = Medium((128, 64, 256), dtype=np.float16, default_c=1540.0)
    
    # Add simple scatterer at depth
    cx = grid.nx // 2
    cy = grid.ny // 2
    cz = int(0.025 / dx)  # 25mm depth
    med.set_region((slice(cx-3, cx+3), slice(cy-2, cy+2), slice(cz-3, cz+3)), c=1600.0)
    
    print(f"\nMedium:")
    print(f"  Background c: {med.c.mean():.0f} m/s")
    print(f"  Scatterer at depth ~{cz*dx*1e3:.1f}mm")
    
    # Create 1.5D transducer
    # 5 rows with variable heights for improved elevation focusing
    row_heights = np.array([0.0003, 0.0004, 0.0005, 0.0004, 0.0003])  # Gaussian-like profile
    tx = Transducer1p5D(
        n_elements_per_row=32,
        n_rows=5,
        pitch=0.0003,
        row_pitch=0.0004,
        row_heights=row_heights,
        center_freq=5e6
    )
    
    print(f"\n1.5D Transducer configuration:")
    print(f"  Elements per row: {tx.n_elements_per_row}")
    print(f"  Number of rows: {tx.n_rows}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  Pitch (lateral): {tx.pitch*1e3:.3f}mm")
    print(f"  Row pitch (elevation): {tx.row_pitch*1e3:.3f}mm")
    print(f"  Row heights: {row_heights*1e3} mm")
    print(f"  Center frequency: {tx.center_freq/1e6:.1f}MHz")
    
    # Compute 3D focusing delays
    # Focus at 25mm depth, centered
    focus_point = (0.0, 0.0, 0.025)
    delays = tx.delays_for_focus_3d(focus_point, speed_of_sound=1540.0)
    
    print(f"\n3D Focusing:")
    print(f"  Focus point: x={focus_point[0]*1e3:.1f}mm, y={focus_point[1]*1e3:.1f}mm, z={focus_point[2]*1e3:.1f}mm")
    print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    print(f"  Max delay: {delays.max()*1e6:.3f}µs")
    
    # Apply apodization to reduce side lobes
    apod = np.hanning(tx.n_elements)
    
    # Generate source waveform
    fs = 1.0 / grid.dt
    sig, sr = tone_burst(center_freq=tx.center_freq, sampling_rate=int(fs), n_cycles=2)
    
    print(f"\nSource signal:")
    print(f"  Waveform: {len(sig)} samples")
    print(f"  Sampling rate: {fs/1e6:.1f}MHz")
    print(f"  Duration: {len(sig)/fs*1e6:.2f}µs")
    
    # For demonstration, use a subset of elements
    # In a full simulation, you would use all elements with BLI masks
    elem_idx = tx.map_to_grid(grid, z0=0.0)
    
    # Use elements with stride for demonstration
    stride = 8  # Use every 8th element to keep it manageable
    active_elements = list(range(0, tx.n_elements, stride))
    
    print(f"\nSimulation setup:")
    print(f"  Using {len(active_elements)} out of {tx.n_elements} elements (stride={stride})")
    print(f"  Active rows: {set([tx.get_element_row_col(i)[0] for i in active_elements])}")
    
    # Map active elements to grid positions
    src_positions = []
    for elem in active_elements:
        ix, iy, iz = elem_idx[elem]
        # Convert to linear index for 3D grid: idx = (iz * ny + iy) * nx + ix
        linear_idx = (iz * grid.ny + iy) * grid.nx + ix
        src_positions.append(linear_idx)
    
    print(f"  Source positions mapped to grid")
    
    print(f"\n{'='*70}")
    print("SIMULATION NOTES:")
    print("This is a demonstration of 1.5D transducer configuration.")
    print("For full simulation:")
    print("  1. Use SolverCore with 3D grid support")
    print("  2. Generate BLI masks for all elements: tx.create_all_element_masks()")
    print("  3. Apply time delays and apodization")
    print("  4. Run simulation with proper source injection")
    print(f"{'='*70}")
    
    print(f"\n✓ 1.5D transducer configuration complete")
    print(f"\nKey advantages of 1.5D arrays:")
    print(f"  • Electronic elevation focusing (vs. fixed lens)")
    print(f"  • Variable row heights for optimized beam profile")
    print(f"  • Fewer elements than full 2D matrix (cost-effective)")
    print(f"  • Good compromise between 1D and 2D arrays")


if __name__ == '__main__':
    main()
