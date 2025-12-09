"""Example: Using transducer masks with band-limited interpolation in a simulation.

This example demonstrates how to use the new transducer mask functionality
to create more accurate source distributions for ultrasound simulations.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from medium import Medium
from transducer import Transducer
from source import tone_burst


def demo_mask_vs_point_source():
    """Compare mask-based source injection vs. point source injection."""
    print("=" * 70)
    print("Demonstration: Transducer Mask vs. Point Source")
    print("=" * 70)
    print()
    
    # Create grid
    dx = 1e-4
    grid = Grid(nx=128, ny=64, nz=128, dx=dx)
    print(f"Grid: {grid.nx} x {grid.ny} x {grid.nz}, dx={dx*1e3:.2f}mm")
    
    # Create transducer
    tx = Transducer(n_elements=8, pitch=0.0003, element_width=0.00028, 
                    element_height=0.002, center_freq=5e6)
    print(f"Transducer: {tx.n_elements} elements, pitch={tx.pitch*1e3:.2f}mm")
    print()
    
    # Generate masks for all elements (normal grid)
    print("Generating masks with band-limited interpolation...")
    masks_normal = tx.create_element_masks(grid, z0=0.0, n_points_x=5, n_points_y=5, 
                                          staggered=False)
    print(f"✓ Created {len(masks_normal)} masks for normal grid")
    
    # Generate masks for staggered grid
    masks_staggered = tx.create_element_masks(grid, z0=0.0, n_points_x=5, n_points_y=5, 
                                              staggered=True)
    print(f"✓ Created {len(masks_staggered)} masks for staggered grid")
    print()
    
    # Compare mask vs point source
    print("Mask properties:")
    for i, (mask_n, mask_s) in enumerate(zip(masks_normal[:3], masks_staggered[:3])):
        print(f"  Element {i}:")
        print(f"    Normal grid   - sum: {np.sum(mask_n):.4f}, spread: {np.count_nonzero(mask_n)} cells")
        print(f"    Staggered grid - sum: {np.sum(mask_s):.4f}, spread: {np.count_nonzero(mask_s)} cells")
    print()
    
    # Show element overlap
    combined_mask = np.sum(masks_normal, axis=0)
    max_overlap = np.max(combined_mask)
    print(f"Maximum overlap (combined mask): {max_overlap:.4f}")
    print(f"  This indicates how much energy from adjacent elements overlaps")
    print()
    
    # Demonstrate usage in source injection
    print("Usage example for source injection:")
    print("  # Instead of injecting at a single point:")
    print("  #   p[ix, iy, iz] += source_signal[step]")
    print("  #")
    print("  # Use mask-weighted injection:")
    print("  #   for elem_idx in active_elements:")
    print("  #       mask = masks[elem_idx]")
    print("  #       p += mask * source_signal[step] * apodization[elem_idx]")
    print()
    
    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    # Visualize mask comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Transducer Masks: Band-Limited Interpolation', fontsize=14)
    
    for idx in range(min(3, len(masks_normal))):
        # Normal grid mask
        ax = axes[0, idx]
        mask_xz = np.sum(masks_normal[idx], axis=1)  # Sum over y
        extent = [-grid.nx*dx*1e3/2, grid.nx*dx*1e3/2, grid.nz*dx*1e3, 0]
        im = ax.imshow(mask_xz.T, aspect='auto', extent=extent, cmap='hot')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'Element {idx} - Normal Grid')
        plt.colorbar(im, ax=ax)
        
        # Staggered grid mask
        ax = axes[1, idx]
        mask_xz = np.sum(masks_staggered[idx], axis=1)  # Sum over y
        im = ax.imshow(mask_xz.T, aspect='auto', extent=extent, cmap='hot')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'Element {idx} - Staggered Grid')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'mask_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison visualization to {out_path}")
    plt.close()
    
    # Show profile comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Surface profiles (normal grid)
    z_idx = grid.nz // 2
    for i in range(min(4, len(masks_normal))):
        profile = np.sum(masks_normal[i][:, :, z_idx], axis=1)
        x_coords = (np.arange(grid.nx) - grid.nx/2) * grid.dx * 1e3
        ax1.plot(x_coords, profile, label=f'Element {i}')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Mask Weight')
    ax1.set_title('Element Masks - Normal Grid (at z=center)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Surface profiles (staggered grid)
    for i in range(min(4, len(masks_staggered))):
        profile = np.sum(masks_staggered[i][:, :, z_idx], axis=1)
        x_coords = (np.arange(grid.nx) - grid.nx/2) * grid.dx * 1e3
        ax2.plot(x_coords, profile, label=f'Element {i}')
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Mask Weight')
    ax2.set_title('Element Masks - Staggered Grid (at z=center)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'mask_profiles.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved profile comparison to {out_path}")
    plt.close()
    
    print()
    print("=" * 70)
    print("Demonstration Complete")
    print("=" * 70)


def main():
    """Run the demonstration."""
    try:
        demo_mask_vs_point_source()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
