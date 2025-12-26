"""Example: 2D matrix transducer simulation with 3D beam steering.

This demo shows how to use a 2D matrix transducer array for full 3D
beam steering and focusing. Matrix arrays provide complete electronic
control in all dimensions, enabling volumetric imaging.
"""
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from medium import Medium
from transducer import Transducer
from source import tone_burst


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("2D MATRIX TRANSDUCER PROBE SIMULATION")
    print("=" * 70)
    
    # Create 3D grid
    # Matrix arrays require proper 3D grid
    dx = 2e-4
    grid = Grid(nx=128, ny=128, nz=256, dx=dx)
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {grid.nx} × {grid.ny} × {grid.nz}")
    print(f"  Resolution: {dx*1e3:.3f}mm")
    print(f"  Physical size: {grid.nx*dx*1e3:.1f}mm × {grid.ny*dx*1e3:.1f}mm × {grid.nz*dx*1e3:.1f}mm")
    
    # Create medium
    med = Medium((128, 128, 256), dtype=np.float16, default_c=1540.0)
    
    # Add multiple scatterers at different positions
    scatterers = [
        (grid.nx//2, grid.ny//2, int(0.025/dx), 1600.0),      # Center
        (grid.nx//2 + 15, grid.ny//2 + 10, int(0.03/dx), 1580.0),  # Off-axis
        (grid.nx//2 - 10, grid.ny//2 - 5, int(0.02/dx), 1620.0),   # Another off-axis
    ]
    
    for cx, cy, cz, speed in scatterers:
        med.set_region((slice(cx-2, cx+2), slice(cy-2, cy+2), slice(cz-2, cz+2)), c=speed)
    
    print(f"\nMedium:")
    print(f"  Background c: {med.c.mean():.0f} m/s")
    print(f"  Scatterers: {len(scatterers)}")
    for i, (cx, cy, cz, speed) in enumerate(scatterers):
        print(f"    {i+1}. Position: ({cx*dx*1e3:.1f}, {cy*dx*1e3:.1f}, {cz*dx*1e3:.1f})mm, c={speed:.0f}m/s")
    
    # Create 2D matrix transducer
    # Start with moderate size for demonstration
    tx = Transducer(
        n_elements_x=32,
        n_elements_y=32,
        pitch=0.0003,
        element_width=0.00028,
        element_height=0.00028,
        center_freq=5e6
    )
    
    print(f"\n2D Matrix Transducer configuration:")
    print(f"  Elements: {tx.n_elements_x} × {tx.n_elements_y} = {tx.n_elements} total")
    print(f"  Pitch: {tx.pitch*1e3:.3f}mm")
    print(f"  Element size: {tx.element_width*1e3:.3f}mm × {tx.element_height*1e3:.3f}mm")
    print(f"  Array size: {(tx.n_elements_x-1)*tx.pitch*1e3:.2f}mm × {(tx.n_elements_y-1)*tx.pitch*1e3:.2f}mm")
    print(f"  Center frequency: {tx.center_freq/1e6:.1f}MHz")
    
    # Demonstrate different beamforming scenarios
    
    # Scenario 1: 3D focusing at center
    print(f"\n{'='*70}")
    print("SCENARIO 1: Center Focus")
    print(f"{'='*70}")
    focus_center = (0.0, 0.0, 0.025)
    delays_center = tx.delays_for_focus_3d(focus_center, speed_of_sound=1540.0)
    
    print(f"  Focus point: x={focus_center[0]*1e3:.1f}mm, y={focus_center[1]*1e3:.1f}mm, z={focus_center[2]*1e3:.1f}mm")
    print(f"  Delay range: [{delays_center.min()*1e6:.3f}, {delays_center.max()*1e6:.3f}]µs")
    print(f"  Max delay: {delays_center.max()*1e6:.3f}µs")
    
    # Scenario 2: 3D focusing off-axis
    print(f"\n{'='*70}")
    print("SCENARIO 2: Off-Axis Focus")
    print(f"{'='*70}")
    focus_offset = (0.003, 0.002, 0.025)
    delays_offset = tx.delays_for_focus_3d(focus_offset, speed_of_sound=1540.0)
    
    print(f"  Focus point: x={focus_offset[0]*1e3:.1f}mm, y={focus_offset[1]*1e3:.1f}mm, z={focus_offset[2]*1e3:.1f}mm")
    print(f"  Delay range: [{delays_offset.min()*1e6:.3f}, {delays_offset.max()*1e6:.3f}]µs")
    print(f"  Max delay: {delays_offset.max()*1e6:.3f}µs")
    
    # Scenario 3: Beam steering
    print(f"\n{'='*70}")
    print("SCENARIO 3: 3D Beam Steering")
    print(f"{'='*70}")
    steering_angles = (np.deg2rad(15), np.deg2rad(10))  # 15° in X-Z, 10° in Y-Z
    delays_steering = tx.delays_for_steering_3d(steering_angles, speed_of_sound=1540.0)
    
    print(f"  Steering angles: θx={np.rad2deg(steering_angles[0]):.1f}°, θy={np.rad2deg(steering_angles[1]):.1f}°")
    print(f"  Delay range: [{delays_steering.min()*1e6:.3f}, {delays_steering.max()*1e6:.3f}]µs")
    print(f"  Max delay: {delays_steering.max()*1e6:.3f}µs")
    
    # Generate source waveform
    fs = 1.0 / grid.dt
    sig, sr = tone_burst(center_freq=tx.center_freq, sampling_rate=int(fs), n_cycles=2)
    
    print(f"\nSource signal:")
    print(f"  Waveform: {len(sig)} samples")
    print(f"  Sampling rate: {fs/1e6:.1f}MHz")
    print(f"  Duration: {len(sig)/fs*1e6:.2f}µs")
    
    # Map elements to grid
    elem_idx = tx.map_to_grid(grid, z0=0.0)
    
    # For demonstration, use a subset of elements
    # In practice, matrix arrays require careful element selection
    stride_x = 4
    stride_y = 4
    active_elements = []
    for row in range(0, tx.n_elements_y, stride_y):
        for col in range(0, tx.n_elements_x, stride_x):
            elem = row * tx.n_elements_x + col
            active_elements.append(elem)
    
    print(f"\nSimulation setup:")
    print(f"  Using {len(active_elements)} out of {tx.n_elements} elements")
    print(f"  Stride: {stride_x} × {stride_y}")
    print(f"  Coverage: {len(active_elements)/(tx.n_elements)*100:.1f}% of array")
    
    # Map active elements to grid positions
    src_positions = []
    for elem in active_elements:
        ix, iy, iz = elem_idx[elem]
        # Convert to linear index for 3D grid
        linear_idx = (iz * grid.ny + iy) * grid.nx + ix
        src_positions.append(linear_idx)
    
    print(f"  Source positions mapped to grid")
    
    # BLI mask statistics
    print(f"\nBLI Mask Generation:")
    print(f"  For accurate simulation, generate BLI masks:")
    print(f"    masks = tx.create_all_element_masks(grid, n_points_x=3, n_points_y=3)")
    print(f"  Memory estimate per element: ~700 sparse entries × 16 bytes ≈ 11KB")
    print(f"  Total for all {tx.n_elements} elements: ≈ {tx.n_elements * 11 / 1024:.1f}MB")
    
    print(f"\n{'='*70}")
    print("SIMULATION NOTES:")
    print("This is a demonstration of 2D matrix transducer configuration.")
    print("For full volumetric imaging:")
    print("  1. Use proper 3D solver implementation")
    print("  2. Generate BLI masks for active elements")
    print("  3. Implement parallel receive beamforming")
    print("  4. Process multiple steering angles for volume reconstruction")
    print(f"{'='*70}")
    
    print(f"\n✓ 2D matrix transducer configuration complete")
    print(f"\nKey advantages of 2D matrix arrays:")
    print(f"  • Full 3D electronic beam steering")
    print(f"  • No mechanical scanning required")
    print(f"  • Enables real-time volumetric imaging")
    print(f"  • Arbitrary scan geometries")
    print(f"  • Adaptive focusing at all depths")
    
    print(f"\nApplications:")
    print(f"  • 3D/4D cardiac imaging")
    print(f"  • Volumetric blood flow imaging")
    print(f"  • Intracardiac echocardiography (ICE)")
    print(f"  • Transesophageal echocardiography (TEE)")


if __name__ == '__main__':
    main()
