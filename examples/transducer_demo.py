"""
Comprehensive Transducer Array Demonstration

This example demonstrates all transducer array types supported by Dolphine:
- 1D Linear Array (roc=0)
- 1D Convex Array (roc>0)
- 1.5D Linear Array (multiple rows, roc=0)
- 2D Matrix Array (large grid, roc=0)

Shows the unified interface with n_cols, n_rows, and roc parameters.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from medium import Medium
from transducer import Transducer
from source import tone_burst


def demo_1d_linear():
    """Demonstrate 1D linear array transducer."""
    print("\n" + "=" * 70)
    print("1D LINEAR ARRAY TRANSDUCER")
    print("=" * 70)
    
    tx = Transducer(
        n_cols=64,           # 64 elements in lateral direction
        n_rows=1,            # Single row
        pitch=0.0003,        # 0.3mm pitch
        roc=0,               # Linear (no curvature)
        center_freq=5e6      # 5MHz center frequency
    )
    
    print(f"\nConfiguration:")
    print(f"  n_cols: {tx.n_cols}")
    print(f"  n_rows: {tx.n_rows}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  ROC: {tx.roc}m (0 = linear)")
    print(f"  Pitch: {tx.pitch*1e3:.3f}mm")
    print(f"  Array width: {(tx.n_cols-1)*tx.pitch*1e3:.2f}mm")
    
    # Element positions
    x_positions = tx.element_positions[:, 0]
    z_positions = tx.element_positions[:, 2]
    print(f"\nElement positions:")
    print(f"  X range: [{x_positions.min()*1e3:.2f}, {x_positions.max()*1e3:.2f}]mm")
    print(f"  Z range: [{z_positions.min()*1e3:.4f}, {z_positions.max()*1e3:.4f}]mm (flat)")
    
    # Focusing
    focus_point = (0.0, 0.0, 0.03)  # 30mm depth
    delays = tx.delays_for_focus(focus_point)
    print(f"\nFocusing at {focus_point[2]*1e3:.1f}mm depth:")
    print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    
    print(f"\nApplications:")
    print(f"  • General purpose 2D imaging")
    print(f"  • Vascular imaging")
    print(f"  • Musculoskeletal imaging")


def demo_1d_convex():
    """Demonstrate 1D convex array transducer."""
    print("\n" + "=" * 70)
    print("1D CONVEX ARRAY TRANSDUCER")
    print("=" * 70)
    
    tx = Transducer(
        n_cols=64,           # 64 elements
        n_rows=1,            # Single row
        pitch=0.0004,        # 0.4mm pitch
        roc=0.05,            # 50mm radius of curvature
        center_freq=3.5e6    # 3.5MHz center frequency
    )
    
    print(f"\nConfiguration:")
    print(f"  n_cols: {tx.n_cols}")
    print(f"  n_rows: {tx.n_rows}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  ROC: {tx.roc*1e3:.1f}mm (>0 = convex)")
    print(f"  Pitch: {tx.pitch*1e3:.3f}mm")
    
    # Element positions
    x_positions = tx.element_positions[:, 0]
    z_positions = tx.element_positions[:, 2]
    print(f"\nElement positions:")
    print(f"  X range: [{x_positions.min()*1e3:.2f}, {x_positions.max()*1e3:.2f}]mm")
    print(f"  Z range: [{z_positions.min()*1e3:.4f}, {z_positions.max()*1e3:.4f}]mm (curved)")
    
    # Calculate angular span
    arc_length = (tx.n_cols - 1) * tx.pitch
    theta_span = arc_length / tx.roc
    print(f"  Angular span: {np.rad2deg(theta_span):.1f}°")
    
    # Focusing
    focus_point = (0.0, 0.0, 0.05)  # 50mm depth
    delays = tx.delays_for_focus(focus_point)
    print(f"\nFocusing at {focus_point[2]*1e3:.1f}mm depth:")
    print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    
    print(f"\nApplications:")
    print(f"  • Abdominal imaging")
    print(f"  • Cardiac imaging")
    print(f"  • Deep tissue imaging")
    print(f"  • Wider field of view")


def demo_1p5d_linear():
    """Demonstrate 1.5D linear array transducer."""
    print("\n" + "=" * 70)
    print("1.5D LINEAR ARRAY TRANSDUCER")
    print("=" * 70)
    
    # Create with variable row heights (Gaussian-like profile)
    row_heights = np.array([0.0003, 0.0004, 0.0005, 0.0004, 0.0003])  # mm
    
    tx = Transducer(
        n_cols=64,           # 64 elements per row
        row_heights=row_heights,  # 5 rows with variable heights
        pitch=0.0003,        # 0.3mm lateral pitch
        row_pitch=0.0004,    # 0.4mm elevation pitch
        roc=0,               # Linear
        center_freq=5e6      # 5MHz
    )
    
    print(f"\nConfiguration:")
    print(f"  n_cols: {tx.n_cols}")
    print(f"  n_rows: {tx.n_rows}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  ROC: {tx.roc}m (0 = linear)")
    print(f"  Pitch (lateral): {tx.pitch*1e3:.3f}mm")
    print(f"  Row pitch (elevation): {tx.row_pitch*1e3:.3f}mm")
    print(f"  Row heights: {row_heights*1e3}mm")
    
    # Array dimensions
    lateral_width = (tx.n_cols - 1) * tx.pitch
    elevation_height = (tx.n_rows - 1) * tx.row_pitch
    print(f"\nArray dimensions:")
    print(f"  Lateral: {lateral_width*1e3:.2f}mm")
    print(f"  Elevation: {elevation_height*1e3:.2f}mm")
    
    # 3D focusing
    focus_point = (0.0, 0.001, 0.03)  # Offset 1mm in elevation
    delays = tx.delays_for_focus(focus_point)
    print(f"\nFocusing at ({focus_point[0]*1e3:.1f}, {focus_point[1]*1e3:.1f}, {focus_point[2]*1e3:.1f})mm:")
    print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    
    # Element indexing
    print(f"\nElement indexing (row-major order):")
    for i in [0, tx.n_cols, tx.n_elements-1]:
        row, col = tx.get_element_row_col(i)
        print(f"  Element {i:3d}: row={row}, col={col:2d}")
    
    print(f"\nApplications:")
    print(f"  • Cardiac imaging with elevation control")
    print(f"  • Enhanced 2D imaging")
    print(f"  • Cost-effective alternative to 2D matrix")
    print(f"  • Electronic elevation focusing")


def demo_2d_matrix():
    """Demonstrate 2D matrix array transducer."""
    print("\n" + "=" * 70)
    print("2D MATRIX ARRAY TRANSDUCER")
    print("=" * 70)
    
    tx = Transducer(
        n_cols=32,           # 32 elements in X
        n_rows=32,           # 32 elements in Y
        pitch=0.0003,        # 0.3mm pitch (both directions)
        element_width=0.00028,   # 0.28mm element width
        element_height=0.00028,  # 0.28mm element height
        roc=0,               # Linear
        center_freq=5e6      # 5MHz
    )
    
    print(f"\nConfiguration:")
    print(f"  n_cols: {tx.n_cols}")
    print(f"  n_rows: {tx.n_rows}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  ROC: {tx.roc}m (0 = linear)")
    print(f"  Pitch: {tx.pitch*1e3:.3f}mm")
    print(f"  Element size: {tx.element_width*1e3:.2f}mm × {tx.element_height*1e3:.2f}mm")
    
    # Array dimensions
    array_width = (tx.n_cols - 1) * tx.pitch
    array_height = (tx.n_rows - 1) * tx.pitch
    print(f"\nArray dimensions:")
    print(f"  Width: {array_width*1e3:.2f}mm")
    print(f"  Height: {array_height*1e3:.2f}mm")
    
    # 3D focusing
    focus_point = (0.002, 0.001, 0.025)  # Off-axis focus
    delays = tx.delays_for_focus(focus_point)
    print(f"\nFocusing at ({focus_point[0]*1e3:.1f}, {focus_point[1]*1e3:.1f}, {focus_point[2]*1e3:.1f})mm:")
    print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    
    # 3D beam steering
    steering_angles = (np.deg2rad(15), np.deg2rad(10))
    delays_steer = tx.delays_for_steering_3d(steering_angles)
    print(f"\nBeam steering at ({np.rad2deg(steering_angles[0]):.1f}°, {np.rad2deg(steering_angles[1]):.1f}°):")
    print(f"  Delay range: [{delays_steer.min()*1e6:.3f}, {delays_steer.max()*1e6:.3f}]µs")
    
    # Element indexing
    print(f"\nElement indexing (row-major order):")
    for i in [0, tx.n_cols, tx.n_elements//2, tx.n_elements-1]:
        row, col = tx.get_element_row_col(i)
        print(f"  Element {i:4d}: row={row:2d}, col={col:2d}")
    
    print(f"\nApplications:")
    print(f"  • Real-time 3D/4D cardiac imaging")
    print(f"  • Volumetric blood flow imaging")
    print(f"  • 3D fetal imaging")
    print(f"  • Full electronic beam steering")


def demo_bli_masks():
    """Demonstrate BLI mask generation for different array types."""
    print("\n" + "=" * 70)
    print("BAND-LIMITED INTERPOLATION (BLI) MASKS")
    print("=" * 70)
    
    # Create a small grid for demonstration
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    
    print(f"\nGrid configuration:")
    print(f"  Size: {grid.nx} × {grid.ny} × {grid.nz}")
    print(f"  Resolution: {grid.dx*1e3:.3f}mm")
    
    # Test different array types
    array_configs = [
        ("1D Linear", Transducer(n_cols=32, n_rows=1, roc=0)),
        ("1D Convex", Transducer(n_cols=32, n_rows=1, roc=0.05)),
        ("1.5D", Transducer(n_cols=16, n_rows=5, roc=0)),
        ("2D Matrix", Transducer(n_cols=16, n_rows=16, roc=0)),
    ]
    
    print(f"\nBLI mask statistics:")
    for name, tx in array_configs:
        # Create mask for center element
        elem_idx = tx.n_elements // 2
        indices, weights = tx.create_element_mask(
            grid, elem_idx,
            n_points_x=5,
            n_points_y=5,
            kernel_radius=3,
            tolerance=1e-3
        )
        
        print(f"\n  {name}:")
        print(f"    Element {elem_idx}: {len(weights)} sparse entries")
        print(f"    Weight sum: {weights.sum():.6f}")
        print(f"    Memory: ~{len(weights)*16/1024:.2f}KB (vs {grid.nx*grid.ny*grid.nz*4/1024**2:.1f}MB dense)")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("DOLPHINE TRANSDUCER ARRAY DEMONSTRATION")
    print("="*70)
    print("\nUnified interface for all transducer types:")
    print("  • n_cols: Number of columns (lateral direction)")
    print("  • n_rows: Number of rows (elevation direction)")
    print("  • roc: Radius of curvature (0 for linear, >0 for convex)")
    
    # Run demonstrations
    demo_1d_linear()
    demo_1d_convex()
    demo_1p5d_linear()
    demo_2d_matrix()
    demo_bli_masks()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nArray type configurations:")
    print("  1D Linear:  n_cols=N, n_rows=1, roc=0")
    print("  1D Convex:  n_cols=N, n_rows=1, roc>0")
    print("  1.5D:       n_cols=N, n_rows=3-7, roc=0")
    print("  2D Matrix:  n_cols=N, n_rows=N, roc=0")
    
    print("\nKey features:")
    print("  ✓ Unified transducer class for all array types")
    print("  ✓ Straightforward n_cols/n_rows interface")
    print("  ✓ ROC parameter for convex arrays")
    print("  ✓ Automatic position generation")
    print("  ✓ 3D focusing and beam steering")
    print("  ✓ Band-limited interpolation masks")
    print("  ✓ Backward compatibility maintained")
    
    print("\n" + "="*70)
    print("✓ Demonstration complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
