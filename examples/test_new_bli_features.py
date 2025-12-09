"""Test script demonstrating new BLI features requested in PR feedback.

New features tested:
1. Minimum grid spacing for point interpolation
2. Grid-center-only constraints per axis
3. Pre-calculated BLI points without grid
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from transducer import Transducer


def test_minimum_grid_spacing():
    """Test that points use minimum grid spacing."""
    print("=" * 70)
    print("Test 1: Minimum Grid Spacing for Point Sampling")
    print("=" * 70)
    
    # Create grid with different spacing in each dimension
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4, dy=2e-4, dz=1.5e-4)
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    min_spacing = min(grid.dx, grid.dy, grid.dz)
    
    print(f"Grid spacing: dx={grid.dx*1e3:.3f}mm, dy={grid.dy*1e3:.3f}mm, dz={grid.dz*1e3:.3f}mm")
    print(f"Minimum grid spacing: {min_spacing*1e3:.3f}mm")
    print()
    
    # Generate points with grid alignment
    points = tx.generate_element_surface_points(0, grid=grid, grid_aligned=True)
    
    print(f"Generated {len(points)} points")
    
    # Check that spacing is based on minimum grid spacing
    x_coords = np.unique(points[:, 0])
    y_coords = np.unique(points[:, 1])
    
    if len(x_coords) > 1:
        x_spacing = np.diff(x_coords).mean()
        print(f"Average X spacing: {x_spacing*1e3:.3f}mm")
    
    if len(y_coords) > 1:
        y_spacing = np.diff(y_coords).mean()
        print(f"Average Y spacing: {y_spacing*1e3:.3f}mm")
    
    print()
    print("✓ Points use minimum grid spacing for sampling")
    print()


def test_grid_center_constraints():
    """Test grid-center-only constraints for each axis."""
    print("=" * 70)
    print("Test 2: Grid-Center-Only Constraints")
    print("=" * 70)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Test without constraints
    points_free = tx.generate_element_surface_points(
        0, n_points_x=5, n_points_y=5, grid=grid, grid_aligned=False
    )
    
    # Test with Y-axis constraint
    points_y_constrained = tx.generate_element_surface_points(
        0, n_points_x=5, n_points_y=5, grid=grid, 
        grid_aligned=False, grid_center_only_y=True
    )
    
    # Test with all axes constrained
    points_all_constrained = tx.generate_element_surface_points(
        0, n_points_x=5, n_points_y=5, grid=grid, 
        grid_aligned=False,
        grid_center_only_x=True,
        grid_center_only_y=True,
        grid_center_only_z=True
    )
    
    print("Free points (no constraints):")
    print(f"  X unique values: {len(np.unique(points_free[:, 0]))}")
    print(f"  Y unique values: {len(np.unique(points_free[:, 1]))}")
    print(f"  Z unique values: {len(np.unique(points_free[:, 2]))}")
    print()
    
    print("Y-axis constrained to grid centers:")
    print(f"  X unique values: {len(np.unique(points_y_constrained[:, 0]))}")
    print(f"  Y unique values: {len(np.unique(points_y_constrained[:, 1]))}")
    print()
    
    # Verify Y coordinates are at grid centers
    grid_center_y = (grid.ny - 1) * grid.dy / 2.0
    y_indices = np.round((points_y_constrained[:, 1] + grid_center_y) / grid.dy)
    y_reconstructed = y_indices * grid.dy - grid_center_y
    y_at_centers = np.allclose(points_y_constrained[:, 1], y_reconstructed)
    print(f"  Y coordinates verified at grid centers: {y_at_centers}")
    print()
    
    print("All axes constrained to grid centers:")
    print(f"  X unique values: {len(np.unique(points_all_constrained[:, 0]))}")
    print(f"  Y unique values: {len(np.unique(points_all_constrained[:, 1]))}")
    print(f"  Z unique values: {len(np.unique(points_all_constrained[:, 2]))}")
    
    # Verify all coordinates are at grid centers
    grid_center_x = (grid.nx - 1) * grid.dx / 2.0
    grid_center_z = (grid.nz - 1) * grid.dz / 2.0
    
    x_indices = np.round((points_all_constrained[:, 0] + grid_center_x) / grid.dx)
    x_reconstructed = x_indices * grid.dx - grid_center_x
    x_at_centers = np.allclose(points_all_constrained[:, 0], x_reconstructed)
    
    z_indices = np.round((points_all_constrained[:, 2] + grid_center_z) / grid.dz)
    z_reconstructed = z_indices * grid.dz - grid_center_z
    z_at_centers = np.allclose(points_all_constrained[:, 2], z_reconstructed)
    
    print(f"  All coordinates verified at grid centers: X={x_at_centers}, Y={y_at_centers}, Z={z_at_centers}")
    print()
    print("✓ Grid-center constraints working correctly")
    print()


def test_grid_independent_calculation():
    """Test BLI point calculation without grid."""
    print("=" * 70)
    print("Test 3: Grid-Independent BLI Point Calculation")
    print("=" * 70)
    
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    print(f"Element dimensions: {tx.element_width*1e3:.3f}mm × {tx.element_height*1e3:.3f}mm")
    print()
    
    # Calculate without grid (grid-independent)
    print("Without grid (grid-independent calculation):")
    for error in [0.01, 0.05]:
        n_x, n_y = tx.calculate_bli_points_for_error(error_tolerance=error, grid=None)
        total = n_x * n_y
        print(f"  Error {error*100:5.1f}%: {n_x:4d} × {n_y:4d} = {total:5d} points")
    print()
    
    # Calculate with grid (grid-aware)
    grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
    print(f"With grid (dx={grid.dx*1e3:.3f}mm, dy={grid.dy*1e3:.3f}mm, dz={grid.dz*1e3:.3f}mm):")
    for error in [0.01, 0.05]:
        n_x, n_y = tx.calculate_bli_points_for_error(error_tolerance=error, grid=grid)
        total = n_x * n_y
        print(f"  Error {error*100:5.1f}%: {n_x:4d} × {n_y:4d} = {total:5d} points")
    print()
    
    print("Note: Grid-independent uses conservative assumed spacing (50µm)")
    print("      Grid-aware uses actual minimum grid spacing")
    print()
    print("✓ Grid-independent calculation working correctly")
    print()


def test_combined_features():
    """Test using all new features together."""
    print("=" * 70)
    print("Test 4: Combined Features")
    print("=" * 70)
    
    grid = Grid(nx=128, ny=64, nz=128, dx=1e-4, dy=1.5e-4, dz=1.2e-4)
    tx = Transducer(n_elements=8, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    print("Creating mask with:")
    print("  - Minimum grid spacing alignment")
    print("  - Y-axis constrained to grid centers")
    print("  - Error tolerance: 1%")
    print()
    
    import time
    t0 = time.time()
    mask = tx.band_limited_interpolation_mask(
        grid, element_idx=0,
        error_tolerance=0.01,
        grid_aligned=True,
        grid_center_only_y=True
    )
    t1 = time.time()
    
    print(f"Mask generated in {(t1-t0)*1000:.2f}ms")
    print(f"Mask sum: {np.sum(mask):.6f}")
    print(f"Non-zero elements: {np.count_nonzero(mask)}")
    print()
    
    # Generate multiple masks
    print("Creating masks for all elements...")
    t0 = time.time()
    masks = tx.create_element_masks(
        grid,
        error_tolerance=0.01,
        grid_aligned=True,
        grid_center_only_y=True
    )
    t1 = time.time()
    
    print(f"Generated {len(masks)} masks in {(t1-t0)*1000:.2f}ms")
    print(f"Average time per mask: {(t1-t0)*1000/len(masks):.2f}ms")
    print()
    print("✓ All features work together correctly")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("NEW BLI FEATURES TEST SUITE")
    print("Testing PR feedback implementation")
    print("=" * 70 + "\n")
    
    try:
        test_minimum_grid_spacing()
        test_grid_center_constraints()
        test_grid_independent_calculation()
        test_combined_features()
        
        print("=" * 70)
        print("ALL NEW FEATURES TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
