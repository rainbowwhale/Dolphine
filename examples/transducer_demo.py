"""
Comprehensive Transducer Array Demonstration

This example demonstrates the new transducer module capabilities:
- N x M array structure (n_cols x n_rows)
- Element properties: position, angle, size
- Array properties: ROC, dimensions
- Multi-layer acoustic lens support
- Plotting functions for array and lens visualization

Shows the clean interface with n_cols, n_rows, roc, and acoustic lens parameters.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from transducer import Transducer, AcousticLens, LensLayer


def demo_flat_array():
    """Demonstrate flat transducer array."""
    print("\n" + "=" * 70)
    print("FLAT TRANSDUCER ARRAY (N x M)")
    print("=" * 70)
    
    tx = Transducer(
        n_cols=64,           # 64 columns (lateral direction)
        n_rows=8,            # 8 rows (elevation direction)
        pitch=0.0003,        # 0.3mm lateral pitch
        row_pitch=0.0004,    # 0.4mm elevation pitch
        roc=0,               # Flat array
        center_freq=5e6      # 5MHz center frequency
    )
    
    print(f"\n{tx}")
    
    # Element information
    print(f"\nElement 0:")
    print(f"  Position: {tx.get_element_position(0) * 1e3} mm")
    print(f"  Angle: {tx.get_element_angle(0)} rad")
    print(f"  Size: {tx.get_element_size()[0]*1e3:.3f}mm × {tx.get_element_size()[1]*1e3:.3f}mm")
    
    # Focusing
    focus = (0.0, 0.0, 0.03)
    delays = tx.delays_for_focus(focus)
    print(f"\nFocusing at {focus[2]*1e3:.0f}mm depth:")
    print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    
    return tx


def demo_curved_array():
    """Demonstrate curved (convex) transducer array."""
    print("\n" + "=" * 70)
    print("CURVED TRANSDUCER ARRAY")
    print("=" * 70)
    
    tx = Transducer(
        n_cols=64,           # 64 columns
        n_rows=1,            # Single row
        pitch=0.0004,        # 0.4mm pitch
        roc=0.05,            # 50mm radius of curvature
        center_freq=3.5e6    # 3.5MHz
    )
    
    print(f"\n{tx}")
    
    # Show curvature in positions
    print(f"\nElement positions showing curvature:")
    print(f"  X range: [{tx.x_positions.min()*1e3:.2f}, {tx.x_positions.max()*1e3:.2f}]mm")
    print(f"  Z range: [{tx.z_positions.min()*1e3:.4f}, {tx.z_positions.max()*1e3:.4f}]mm")
    
    # Element angles
    print(f"\nElement angles (normal directions):")
    print(f"  Center element: θx={np.rad2deg(tx.element_angles[32, 0]):.2f}°")
    print(f"  Edge element: θx={np.rad2deg(tx.element_angles[0, 0]):.2f}°")
    
    return tx


def demo_acoustic_lens():
    """Demonstrate acoustic lens modeling."""
    print("\n" + "=" * 70)
    print("ACOUSTIC LENS (Multi-Layer)")
    print("=" * 70)
    
    # Create a multi-layer acoustic lens
    lens = AcousticLens()
    
    # Layer 1: Convex focusing layer
    lens.add_layer(LensLayer(
        elevational_roc=0.020,      # 20mm convex ROC
        max_thickness=0.001,        # 1mm max thickness
        speed_of_sound=1000.0,      # 1000 m/s (slower than tissue for focusing)
        density=1100.0,
        name="Focus layer"
    ))
    
    # Layer 2: Acoustic matching layer
    lens.add_layer(LensLayer(
        elevational_roc=0,          # Flat (infinite ROC)
        max_thickness=0.0005,       # 0.5mm thickness
        speed_of_sound=1500.0,      # 1500 m/s
        density=1050.0,
        name="Matching layer"
    ))
    
    print(f"\nLens configuration:")
    print(f"  Number of layers: {lens.n_layers}")
    print(f"  Total max thickness: {lens.total_max_thickness*1e3:.2f}mm")
    
    for i, layer in enumerate(lens.layers):
        print(f"\n  Layer {i+1}: {layer.name}")
        print(f"    Elevational ROC: {layer.elevational_roc*1e3:.1f}mm " +
              f"({'convex' if layer.is_convex else 'concave' if layer.is_concave else 'flat'})")
        print(f"    Max thickness: {layer.max_thickness*1e3:.2f}mm")
        print(f"    Speed of sound: {layer.speed_of_sound:.0f} m/s")
    
    # Create transducer with lens
    tx = Transducer(
        n_cols=64,
        n_rows=5,
        pitch=0.0003,
        row_pitch=0.0004,
        lens=lens,
        center_freq=5e6
    )
    
    print(f"\nTransducer with lens:")
    print(tx)
    
    # Lens thickness profile
    y = np.linspace(-0.002, 0.002, 11)
    thickness = lens.get_total_thickness_profile(y)
    print(f"\nLens thickness profile (at elevation positions):")
    for i in range(0, len(y), 2):
        print(f"  y={y[i]*1e3:+.2f}mm: thickness={thickness[i]*1e3:.3f}mm")
    
    return tx


def demo_concave_lens():
    """Demonstrate concave (diverging) lens."""
    print("\n" + "=" * 70)
    print("CONCAVE LENS (Negative ROC)")
    print("=" * 70)
    
    # Create a concave lens (negative ROC)
    lens = AcousticLens()
    lens.add_layer(LensLayer(
        elevational_roc=-0.030,     # -30mm (concave)
        max_thickness=0.0008,       # 0.8mm max thickness
        speed_of_sound=1000.0,
        name="Diverging layer"
    ))
    
    print(f"\nConcave lens configuration:")
    print(f"  Elevational ROC: {lens.layers[0].elevational_roc*1e3:.1f}mm (negative = concave)")
    print(f"  Is concave: {lens.layers[0].is_concave}")
    
    # Thickness profile (thinner in center, thicker at edges)
    y = np.linspace(-0.002, 0.002, 11)
    thickness = lens.get_total_thickness_profile(y)
    print(f"\nConcave lens thickness profile:")
    print(f"  Center (y=0): {thickness[5]*1e3:.3f}mm")
    print(f"  Edge (y=±2mm): {thickness[0]*1e3:.3f}mm")
    
    return lens


def demo_element_indexing():
    """Demonstrate element indexing and access."""
    print("\n" + "=" * 70)
    print("ELEMENT INDEXING AND ACCESS")
    print("=" * 70)
    
    tx = Transducer(n_cols=16, n_rows=8, pitch=0.0003)
    
    print(f"\nArray: {tx.n_cols} × {tx.n_rows} = {tx.n_elements} elements")
    print(f"\nRow-major element ordering:")
    
    # Show element indexing
    indices_to_show = [0, 15, 16, 31, 64, tx.n_elements-1]
    for idx in indices_to_show:
        row, col = tx.get_element_row_col(idx)
        pos = tx.get_element_position(idx) * 1e3
        print(f"  Element {idx:3d}: row={row}, col={col:2d}, pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f})mm")
    
    # Reverse lookup
    print(f"\nReverse lookup (row, col) -> index:")
    for row, col in [(0, 0), (0, 15), (4, 8), (7, 15)]:
        idx = tx.get_element_index(row, col)
        print(f"  ({row}, {col:2d}) -> Element {idx}")


def demo_array_properties():
    """Demonstrate array dimension properties."""
    print("\n" + "=" * 70)
    print("ARRAY PROPERTIES")
    print("=" * 70)
    
    tx = Transducer(
        n_cols=64,
        n_rows=16,
        pitch=0.0003,
        row_pitch=0.0004,
        element_width=0.00028,
        element_height=0.00038,
        kerf=0.00002,
        center_freq=5e6
    )
    
    print(f"\n{tx}")
    
    print(f"\nDimension properties:")
    print(f"  Array size: {tx.array_size[0]*1e3:.2f}mm × {tx.array_size[1]*1e3:.2f}mm")
    print(f"  Array width: {tx.array_width*1e3:.2f}mm")
    print(f"  Array height: {tx.array_height*1e3:.2f}mm")
    print(f"  Min dimension: {tx.min_dimension*1e3:.2f}mm")
    print(f"  Max dimension: {tx.max_dimension*1e3:.2f}mm")
    print(f"  Wavelength: {tx.wavelength*1e3:.3f}mm")


def demo_beamforming():
    """Demonstrate beamforming calculations."""
    print("\n" + "=" * 70)
    print("BEAMFORMING (Focusing and Steering)")
    print("=" * 70)
    
    tx = Transducer(n_cols=64, n_rows=8, pitch=0.0003)
    
    # Focusing
    focus_2d = (0.0, 0.02)  # (x, z) 2D focus
    focus_3d = (0.005, 0.002, 0.03)  # (x, y, z) 3D focus
    
    delays_2d = tx.delays_for_focus(focus_2d)
    delays_3d = tx.delays_for_focus(focus_3d)
    
    print(f"\nFocusing delays:")
    print(f"  2D focus at (x=0, z=20mm):")
    print(f"    Delay range: [{delays_2d.min()*1e6:.3f}, {delays_2d.max()*1e6:.3f}]µs")
    print(f"  3D focus at (x=5mm, y=2mm, z=30mm):")
    print(f"    Delay range: [{delays_3d.min()*1e6:.3f}, {delays_3d.max()*1e6:.3f}]µs")
    
    # Steering
    steering = (np.deg2rad(15), np.deg2rad(10))
    delays_steer = tx.delays_for_steering(steering)
    
    print(f"\nBeam steering delays:")
    print(f"  Steering angles: θx={np.rad2deg(steering[0]):.1f}°, θy={np.rad2deg(steering[1]):.1f}°")
    print(f"  Delay range: [{delays_steer.min()*1e6:.3f}, {delays_steer.max()*1e6:.3f}]µs")
    
    # Apodization
    apod = tx.apodization_hanning()
    print(f"\nHanning apodization:")
    print(f"  Shape: {apod.shape}")
    print(f"  Range: [{apod.min():.4f}, {apod.max():.4f}]")


def demo_plotting():
    """Demonstrate plotting capabilities."""
    print("\n" + "=" * 70)
    print("PLOTTING CAPABILITIES")
    print("=" * 70)
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping plotting demo.")
        return
    
    # Create transducer with lens for comprehensive demo
    lens = AcousticLens()
    lens.add_layer(LensLayer(elevational_roc=0.020, max_thickness=0.001, name="Focus layer"))
    lens.add_layer(LensLayer(elevational_roc=-0.050, max_thickness=0.0006, name="Diverging layer"))
    lens.add_layer(LensLayer(elevational_roc=0, max_thickness=0.0004, name="Matching layer"))
    
    tx = Transducer(
        n_cols=32,
        n_rows=8,
        pitch=0.0003,
        row_pitch=0.0004,
        lens=lens,
        center_freq=5e6
    )
    
    print(f"\nCreating plots for: {tx.n_cols}×{tx.n_rows} array with {lens.n_layers}-layer lens")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: 3D array view
    ax1 = fig.add_subplot(131, projection='3d')
    tx.plot_array(ax=ax1)
    ax1.set_title(f'3D Array View\n({tx.n_cols}×{tx.n_rows} elements)')
    
    # Plot 2: 2D array view
    ax2 = fig.add_subplot(132)
    tx.plot_array_2d(ax=ax2)
    ax2.set_title('2D Array View (Top-down)')
    
    # Plot 3: Lens cross-section
    ax3 = fig.add_subplot(133)
    tx.plot_lens(ax=ax3)
    ax3.set_title(f'Lens Cross-Section\n({lens.n_layers} layers)')
    
    plt.tight_layout()
    
    # Save to results directory
    os.makedirs('results', exist_ok=True)
    output_path = 'results/transducer_demo.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nPlots saved to: {output_path}")
    
    plt.close()


def demo_bli_masks():
    """Demonstrate BLI mask generation."""
    print("\n" + "=" * 70)
    print("BLI (Band-Limited Interpolation) MASKS")
    print("=" * 70)
    
    # Import Grid class
    try:
        from grid import Grid
    except ImportError:
        print("Grid class not available. Skipping BLI demo.")
        return
    
    # Create small grid and transducer
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_cols=16, n_rows=4, pitch=0.0003)
    
    print(f"\nGrid: {grid.nx}×{grid.ny}×{grid.nz}, resolution={grid.dx*1e3:.3f}mm")
    print(f"Transducer: {tx.n_cols}×{tx.n_rows} = {tx.n_elements} elements")
    
    # Generate mask for one element
    elem_idx = tx.n_elements // 2
    indices, weights = tx.create_element_mask(
        grid, elem_idx,
        n_points_x=5,
        n_points_y=5,
        kernel_radius=3,
        tolerance=1e-3
    )
    
    print(f"\nElement {elem_idx} BLI mask:")
    print(f"  Sparse entries: {len(weights)}")
    print(f"  Weight sum: {weights.sum():.6f}")
    print(f"  Memory: ~{len(weights)*16/1024:.2f}KB (sparse)")
    print(f"  vs dense: {grid.nx*grid.ny*grid.nz*4/1024**2:.1f}MB")
    print(f"  Savings: {(1 - len(weights)*16/(grid.nx*grid.ny*grid.nz*4))*100:.1f}%")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("DOLPHINE NEW TRANSDUCER MODULE DEMONSTRATION")
    print("=" * 70)
    print("\nThis module provides:")
    print("  • N × M array structure (n_cols × n_rows)")
    print("  • Element properties: position, angle, size")
    print("  • Array curvature (ROC)")
    print("  • Multi-layer acoustic lens support")
    print("  • Plotting functions for visualization")
    
    # Run all demos
    demo_flat_array()
    demo_curved_array()
    demo_acoustic_lens()
    demo_concave_lens()
    demo_element_indexing()
    demo_array_properties()
    demo_beamforming()
    demo_bli_masks()
    demo_plotting()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nNew Transducer module features:")
    print("  ✓ Clean N×M array interface (n_cols, n_rows)")
    print("  ✓ Element position, angle, and size access")
    print("  ✓ Array ROC for curved transducers")
    print("  ✓ Multi-layer acoustic lens (AcousticLens, LensLayer)")
    print("  ✓ Convex and concave lens layers (positive/negative ROC)")
    print("  ✓ Array dimension properties (size, min/max)")
    print("  ✓ Beamforming (focusing, steering, apodization)")
    print("  ✓ Plotting (3D array, 2D array, lens cross-section)")
    print("  ✓ BLI mask generation for simulation")
    
    print("\n" + "=" * 70)
    print("✓ Demonstration complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
