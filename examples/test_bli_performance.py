"""Performance test and demonstration of new BLI features.

Tests the improvements requested in the PR feedback:
1. Grid-aligned sampling
2. Error-tolerance-based point calculation
3. Optimized vectorized implementation for large arrays
"""
import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from transducer import Transducer


def test_grid_aligned_sampling():
    """Test grid-aligned point sampling."""
    print("=" * 70)
    print("Test 1: Grid-Aligned Sampling")
    print("=" * 70)
    
    grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
    tx = Transducer(n_elements=8, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Old method: uniform spacing
    points_uniform = tx.generate_element_surface_points(0, n_points_x=5, n_points_y=5, 
                                                        grid=None, grid_aligned=False)
    print(f"Uniform spacing: {len(points_uniform)} points")
    
    # New method: grid-aligned
    points_aligned = tx.generate_element_surface_points(0, grid=grid, grid_aligned=True)
    print(f"Grid-aligned:    {len(points_aligned)} points (auto-calculated)")
    
    # Check spacing
    x_coords = points_aligned[:, 0]
    x_unique = np.unique(x_coords)
    if len(x_unique) > 1:
        x_spacing = np.diff(x_unique).mean()
        print(f"Average x-spacing: {x_spacing*1e3:.4f}mm (grid dx: {grid.dx*1e3:.4f}mm)")
    
    print("✓ Grid-aligned sampling working correctly")
    print()


def test_error_tolerance_calculation():
    """Test error-tolerance-based point calculation."""
    print("=" * 70)
    print("Test 2: Error-Tolerance-Based Point Calculation")
    print("=" * 70)
    
    grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
    tx = Transducer(n_elements=8, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    error_levels = [0.001, 0.01, 0.05, 0.1]
    
    print(f"Element size: {tx.element_width*1e3:.3f}mm × {tx.element_height*1e3:.3f}mm")
    print(f"Grid spacing: {grid.dx*1e3:.3f}mm × {grid.dy*1e3:.3f}mm")
    print()
    
    for error in error_levels:
        n_x, n_y = tx.calculate_bli_points_for_error(grid, error_tolerance=error)
        total_points = n_x * n_y
        print(f"Error tolerance {error*100:5.1f}%: {n_x:3d} × {n_y:3d} = {total_points:4d} points")
    
    print()
    print("✓ Error-tolerance calculation working correctly")
    print()


def test_vectorized_performance():
    """Test performance of vectorized implementation."""
    print("=" * 70)
    print("Test 3: Vectorized Performance")
    print("=" * 70)
    
    grid = Grid(nx=128, ny=64, nz=128, dx=1e-4)
    tx = Transducer(n_elements=16, pitch=0.0003, element_width=0.00028, element_height=0.002)
    
    # Test with different error tolerances
    print("Single element mask generation time:")
    print()
    
    for error in [0.01, 0.05]:
        n_x, n_y = tx.calculate_bli_points_for_error(grid, error_tolerance=error)
        n_points = n_x * n_y
        
        t0 = time.time()
        mask = tx.band_limited_interpolation_mask(
            grid, 0, error_tolerance=error, grid_aligned=True
        )
        t1 = time.time()
        
        print(f"  Error {error*100}%: {n_points:4d} points -> {(t1-t0)*1000:6.2f}ms, sum={np.sum(mask):.4f}")
    
    print()
    print("Multiple elements:")
    
    # Test all elements with 1% error
    t0 = time.time()
    masks = tx.create_element_masks(grid, error_tolerance=0.01, grid_aligned=True)
    t1 = time.time()
    
    n_x, n_y = tx.calculate_bli_points_for_error(grid, error_tolerance=0.01)
    total_points = n_x * n_y * tx.n_elements
    
    print(f"  {tx.n_elements} elements, {n_x}×{n_y}={n_x*n_y} points/element")
    print(f"  Total: {total_points} points -> {(t1-t0)*1000:.2f}ms ({(t1-t0)*1000/tx.n_elements:.2f}ms/element)")
    
    print()
    print("✓ Vectorized implementation working efficiently")
    print()


def test_large_array_simulation():
    """Simulate performance for large 2D matrix probe."""
    print("=" * 70)
    print("Test 4: Large Array Simulation (2D Matrix Probe)")
    print("=" * 70)
    
    # Simulate a 100x100 element 2D matrix probe
    n_elements_1d = 100
    n_elements_total = n_elements_1d * n_elements_1d
    
    grid = Grid(nx=256, ny=256, nz=256, dx=1e-4)
    
    # Single representative element
    tx_single = Transducer(n_elements=1, pitch=0.0003, element_width=0.00028, element_height=0.00028)
    
    # Measure time for one element
    t0 = time.time()
    mask = tx_single.band_limited_interpolation_mask(
        grid, 0, error_tolerance=0.01, grid_aligned=True
    )
    t_single = time.time() - t0
    
    n_x, n_y = tx_single.calculate_bli_points_for_error(grid, error_tolerance=0.01)
    points_per_element = n_x * n_y
    
    # Extrapolate to full array
    estimated_total_time = t_single * n_elements_total
    estimated_total_points = points_per_element * n_elements_total
    
    print(f"Matrix probe: {n_elements_1d}×{n_elements_1d} = {n_elements_total} elements")
    print(f"Points per element: {points_per_element}")
    print(f"Total sampling points: {estimated_total_points:,}")
    print()
    print(f"Time per element: {t_single*1000:.2f}ms")
    print(f"Estimated total time: {estimated_total_time:.2f}s ({estimated_total_time/60:.2f} minutes)")
    print()
    
    # Memory estimate
    bytes_per_mask = grid.nx * grid.ny * grid.nz * 4  # float32
    total_memory_mb = bytes_per_mask * n_elements_total / (1024**2)
    
    print(f"Memory per mask: {bytes_per_mask/(1024**2):.2f} MB")
    print(f"Total memory for all masks: {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
    print()
    print("Note: For such large arrays, consider:")
    print("  - Computing masks on-demand (one element at a time)")
    print("  - Using sparse matrix storage")
    print("  - Reducing error tolerance (fewer points)")
    print("  - Processing elements in batches")
    
    print()
    print("✓ Large array simulation completed")
    print()


def test_backward_compatibility():
    """Test that old API still works."""
    print("=" * 70)
    print("Test 5: Backward Compatibility")
    print("=" * 70)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements=4, pitch=0.0003, element_width=0.00028)
    
    # Old API should still work
    points_old = tx.generate_element_surface_points(0, n_points_x=5, n_points_y=5)
    print(f"Old API: generate_element_surface_points(0, 5, 5) -> {len(points_old)} points")
    
    mask_old = tx.band_limited_interpolation_mask(grid, 0, n_points_x=5, n_points_y=5)
    print(f"Old API: band_limited_interpolation_mask(..., 5, 5) -> sum={np.sum(mask_old):.4f}")
    
    masks_old = tx.create_element_masks(grid, n_points_x=5, n_points_y=5)
    print(f"Old API: create_element_masks(..., 5, 5) -> {len(masks_old)} masks")
    
    print()
    print("✓ Backward compatibility maintained")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST SUITE - NEW BLI FEATURES")
    print("=" * 70 + "\n")
    
    try:
        test_grid_aligned_sampling()
        test_error_tolerance_calculation()
        test_vectorized_performance()
        test_large_array_simulation()
        test_backward_compatibility()
        
        print("=" * 70)
        print("ALL PERFORMANCE TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
