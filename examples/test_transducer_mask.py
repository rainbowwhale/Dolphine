"""Test script for transducer mask generation with band-limited interpolation.

This script demonstrates the new transducer mask functionality, including:
1. Generating surface points on rectangular elements
2. Creating masks using band-limited interpolation
3. Testing both normal and staggered grid configurations
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from transducer import Transducer


def test_surface_point_generation():
    """Test generation of surface points on rectangular elements."""
    print("=" * 60)
    print("Test 1: Surface Point Generation")
    print("=" * 60)
    
    # Create a simple transducer
    tx = Transducer(n_elements=8, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Generate points for first element
    element_idx = 0
    points = tx.generate_element_surface_points(element_idx, n_points_x=5, n_points_y=5)
    
    print(f"Generated {len(points)} surface points for element {element_idx}")
    print(f"Point coordinates shape: {points.shape}")
    print(f"Sample points (first 5):")
    for i in range(min(5, len(points))):
        print(f"  Point {i}: x={points[i,0]*1e3:.4f}mm, y={points[i,1]*1e3:.4f}mm, z={points[i,2]*1e3:.4f}mm")
    
    # Verify points are within element boundaries
    element_x = tx.element_positions[element_idx, 0]
    x_min = element_x - tx.element_width / 2
    x_max = element_x + tx.element_width / 2
    y_min = -tx.element_height / 2
    y_max = tx.element_height / 2
    
    assert np.all(points[:, 0] >= x_min - 1e-10), "Points outside element x-bounds"
    assert np.all(points[:, 0] <= x_max + 1e-10), "Points outside element x-bounds"
    assert np.all(points[:, 1] >= y_min - 1e-10), "Points outside element y-bounds"
    assert np.all(points[:, 1] <= y_max + 1e-10), "Points outside element y-bounds"
    
    print("✓ All points are within element boundaries")
    print()


def test_all_elements_points():
    """Test generation of surface points for all elements."""
    print("=" * 60)
    print("Test 2: All Elements Surface Points")
    print("=" * 60)
    
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028)
    points_list, element_indices = tx.generate_all_elements_surface_points(n_points_x=3, n_points_y=3)
    
    print(f"Generated points for {len(points_list)} elements")
    for i, (points, elem_idx) in enumerate(zip(points_list, element_indices)):
        print(f"  Element {elem_idx}: {len(points)} points")
    
    assert len(points_list) == tx.n_elements, "Point list length mismatch"
    print("✓ Successfully generated points for all elements")
    print()


def test_mask_generation_normal_grid():
    """Test mask generation on normal grid."""
    print("=" * 60)
    print("Test 3: Mask Generation (Normal Grid)")
    print("=" * 60)
    
    # Create small grid for testing - make sure grid is large enough for element
    # Grid should be centered and span enough to cover transducer elements
    grid = Grid(nx=64, ny=128, nz=64, dx=1e-4)
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Generate mask for first element
    element_idx = 0
    mask = tx.band_limited_interpolation_mask(
        grid, element_idx, n_points_x=3, n_points_y=3, z0=0.0, staggered=False
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum: {np.sum(mask):.6f} (should be close to 1.0)")
    print(f"Mask max: {np.max(mask):.6f}")
    print(f"Mask min: {np.min(mask):.6f}")
    print(f"Non-zero elements: {np.count_nonzero(mask)}")
    
    # Verify mask properties
    assert mask.shape == (grid.nx, grid.ny, grid.nz), "Mask shape mismatch"
    assert np.abs(np.sum(mask) - 1.0) < 0.2, f"Mask sum {np.sum(mask)} far from 1.0"
    assert np.min(mask) >= -0.1, "Mask contains large negative values"
    
    print("✓ Mask properties validated")
    print()
    
    return mask, grid


def test_mask_generation_staggered_grid():
    """Test mask generation on staggered grid."""
    print("=" * 60)
    print("Test 4: Mask Generation (Staggered Grid)")
    print("=" * 60)
    
    # Create small grid for testing - make sure grid is large enough for element
    grid = Grid(nx=64, ny=128, nz=64, dx=1e-4)
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Generate mask for first element with staggered grid
    element_idx = 0
    mask = tx.band_limited_interpolation_mask(
        grid, element_idx, n_points_x=3, n_points_y=3, z0=0.0, staggered=True
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum: {np.sum(mask):.6f} (should be close to 1.0)")
    print(f"Mask max: {np.max(mask):.6f}")
    print(f"Non-zero elements: {np.count_nonzero(mask)}")
    
    # Verify mask properties
    assert mask.shape == (grid.nx, grid.ny, grid.nz), "Mask shape mismatch"
    assert np.abs(np.sum(mask) - 1.0) < 0.2, f"Mask sum {np.sum(mask)} far from 1.0"
    assert np.min(mask) >= -0.1, "Mask contains large negative values"
    
    print("✓ Staggered mask properties validated")
    print()
    
    return mask, grid


def test_all_element_masks():
    """Test creating masks for all elements."""
    print("=" * 60)
    print("Test 5: All Element Masks")
    print("=" * 60)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Generate masks for all elements
    masks = tx.create_element_masks(grid, z0=0.0, n_points_x=3, n_points_y=3, staggered=False)
    
    print(f"Created {len(masks)} masks")
    for i, mask in enumerate(masks):
        print(f"  Element {i}: sum={np.sum(mask):.6f}, non-zero={np.count_nonzero(mask)}")
    
    assert len(masks) == tx.n_elements, "Number of masks mismatch"
    for mask in masks:
        assert mask.shape == (grid.nx, grid.ny, grid.nz), "Mask shape mismatch"
    
    print("✓ All element masks created successfully")
    print()
    
    return masks, grid, tx


def visualize_masks(masks, grid, tx):
    """Visualize the generated masks."""
    print("=" * 60)
    print("Test 6: Visualization")
    print("=" * 60)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    # Visualize masks in x-z plane (collapse y dimension)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Transducer Element Masks (Band-Limited Interpolation)')
    
    for idx in range(min(4, len(masks))):
        ax = axes[idx // 2, idx % 2]
        mask_xz = np.sum(masks[idx], axis=1)  # Sum over y
        
        extent = [0, grid.nx * grid.dx * 1e3,  # x in mm
                 grid.nz * grid.dx * 1e3, 0]     # z in mm
        
        im = ax.imshow(mask_xz.T, aspect='auto', extent=extent, cmap='hot')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'Element {idx} Mask (x-z plane)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'transducer_masks.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {out_path}")
    plt.close()
    
    # Create a combined view showing all elements
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sum all masks
    combined_mask = np.sum(masks, axis=0)
    mask_xz_combined = np.sum(combined_mask, axis=1)
    
    extent = [0, grid.nx * grid.dx * 1e3, grid.nz * grid.dx * 1e3, 0]
    
    im1 = ax1.imshow(mask_xz_combined.T, aspect='auto', extent=extent, cmap='hot')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title('Combined Mask (All Elements)')
    plt.colorbar(im1, ax=ax1)
    
    # Show individual element contributions along x-axis at z=0
    z_idx = 0
    profiles = []
    for mask in masks:
        profile = np.sum(mask[:, :, z_idx], axis=1)
        profiles.append(profile)
    
    x_coords = np.arange(grid.nx) * grid.dx * 1e3
    for i, profile in enumerate(profiles):
        ax2.plot(x_coords, profile, label=f'Element {i}')
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Mask Weight')
    ax2.set_title('Element Masks at Surface (z=0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'transducer_masks_combined.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved combined visualization to {out_path}")
    plt.close()
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TRANSDUCER MASK GENERATION TEST SUITE")
    print("Reference: https://doi.org/10.1121/1.5116132")
    print("=" * 60 + "\n")
    
    try:
        # Run tests
        test_surface_point_generation()
        test_all_elements_points()
        test_mask_generation_normal_grid()
        test_mask_generation_staggered_grid()
        masks, grid, tx = test_all_element_masks()
        visualize_masks(masks, grid, tx)
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
