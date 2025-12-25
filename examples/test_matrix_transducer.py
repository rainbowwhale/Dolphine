"""
Test suite for 2D Matrix Transducer implementation

Tests the 2D matrix transducer with uniform element grid:
1. Large 2D element arrays (10s-100s of elements)
2. Uniform element dimensions
3. 3D focusing and steering
4. BLI mask generation
5. Element geometry
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from grid import Grid
from transducer import MatrixTransducer


def test_basic_creation():
    """Test basic 2D matrix transducer creation."""
    print("=" * 70)
    print("Test 1: Basic 2D Matrix Transducer Creation")
    print("=" * 70)
    
    # Create small matrix transducer with square elements
    tx = MatrixTransducer(n_elements_x=16, n_elements_y=16, element_height=0.00028)
    
    print(f"Created 2D matrix transducer:")
    print(f"  Elements in X: {tx.n_elements_x}")
    print(f"  Elements in Y: {tx.n_elements_y}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  Element size: {tx.element_width*1e3:.3f}mm × {tx.element_height*1e3:.3f}mm")
    
    assert tx.n_elements == 16 * 16, "Total elements should be 256"
    assert tx.element_width == tx.element_height, "Elements should be square (element_width should equal element_height)"
    
    print("\n✓ Basic creation OK")
    print()


def test_large_arrays():
    """Test creation of large 2D arrays."""
    print("=" * 70)
    print("Test 2: Large Array Support")
    print("=" * 70)
    
    # Test various array sizes
    array_sizes = [
        (16, 16, "Small"),
        (32, 32, "Medium"),
        (64, 64, "Large"),
        (128, 128, "Very Large"),
    ]
    
    for nx, ny, label in array_sizes:
        tx = MatrixTransducer(n_elements_x=nx, n_elements_y=ny)
        print(f"{label} array {nx}×{ny}:")
        print(f"  Total elements: {tx.n_elements}")
        print(f"  Array size: {(nx-1)*tx.pitch*1e3:.2f}mm × {(ny-1)*tx.pitch*1e3:.2f}mm")
        
        assert tx.n_elements == nx * ny, f"Should have {nx*ny} elements"
    
    print("\n✓ Large array support OK")
    print()


def test_uniform_elements():
    """Test uniform element dimensions."""
    print("=" * 70)
    print("Test 3: Uniform Element Dimensions")
    print("=" * 70)
    
    element_size = 0.0002  # 0.2mm
    tx = MatrixTransducer(n_elements_x=16, n_elements_y=16,
                         element_width=element_size,
                         element_height=element_size)
    
    print(f"Element dimensions:")
    print(f"  Width: {tx.element_width*1e3:.4f}mm")
    print(f"  Height: {tx.element_height*1e3:.4f}mm")
    
    # Verify all elements have same size
    assert tx.element_heights is None, "Should use uniform height (not per-element)"
    
    # Generate points for different elements and check size consistency
    for elem_idx in [0, 50, 100, 255]:
        points = tx.generate_element_surface_points(elem_idx, n_points_x=3, n_points_y=3)
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        
        assert abs(x_range - element_size) < 1e-9, f"Element {elem_idx} X range incorrect"
        assert abs(y_range - element_size) < 1e-9, f"Element {elem_idx} Y range incorrect"
    
    print(f"  All elements have uniform size ✓")
    
    print("\n✓ Uniform element dimensions OK")
    print()


def test_element_positioning():
    """Test 2D element grid positioning."""
    print("=" * 70)
    print("Test 4: 2D Element Grid Positioning")
    print("=" * 70)
    
    tx = MatrixTransducer(n_elements_x=8, n_elements_y=8, pitch=0.0003)
    
    print(f"Element positions (corners and center):")
    test_elements = [
        (0, "Top-left corner"),
        (7, "Top-right corner"),
        (56, "Bottom-left corner"),
        (63, "Bottom-right corner"),
        (27, "Near center"),
    ]
    
    for elem_idx, label in test_elements:
        row, col = tx.get_element_row_col(elem_idx)
        x, y, z = tx.element_positions[elem_idx]
        print(f"  Element {elem_idx:2d} ({label}): row={row}, col={col}, "
              f"x={x*1e3:6.3f}mm, y={y*1e3:6.3f}mm")
    
    # Check grid structure
    # First element should be at row 0, col 0
    row, col = tx.get_element_row_col(0)
    assert row == 0 and col == 0, "Element 0 should be at (0, 0)"
    
    # Last element of first row
    row, col = tx.get_element_row_col(7)
    assert row == 0 and col == 7, "Element 7 should be at (0, 7)"
    
    # First element of second row
    row, col = tx.get_element_row_col(8)
    assert row == 1 and col == 0, "Element 8 should be at (1, 0)"
    
    # Last element
    row, col = tx.get_element_row_col(63)
    assert row == 7 and col == 7, "Element 63 should be at (7, 7)"
    
    # Check positions are centered (element_positions is now 3D)
    x_positions = tx.element_positions[:, 0]
    y_positions = tx.element_positions[:, 1]
    
    print(f"\nPosition ranges:")
    print(f"  X: [{x_positions.min()*1e3:.3f}, {x_positions.max()*1e3:.3f}]mm")
    print(f"  Y: [{y_positions.min()*1e3:.3f}, {y_positions.max()*1e3:.3f}]mm")
    
    # Check centering
    assert abs(x_positions.mean()) < 1e-10, "X positions should be centered"
    assert abs(y_positions.mean()) < 1e-10, "Y positions should be centered"
    
    print("\n✓ Element grid positioning OK")
    print()


def test_3d_focusing():
    """Test 3D focusing capabilities."""
    print("=" * 70)
    print("Test 5: 3D Focusing")
    print("=" * 70)
    
    tx = MatrixTransducer(n_elements_x=16, n_elements_y=16)
    
    # Test various focus points
    focus_points = [
        ((0.0, 0.0, 0.03), "Center, 30mm depth"),
        ((0.001, 0.0, 0.03), "1mm X offset"),
        ((0.0, 0.001, 0.03), "1mm Y offset"),
        ((0.001, 0.001, 0.03), "1mm diagonal offset"),
    ]
    
    for focus_point, label in focus_points:
        delays = tx.delays_for_focus_3d(focus_point)
        print(f"{label}:")
        print(f"  Focus: {focus_point}")
        print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
        print(f"  Max delay: {delays.max()*1e6:.3f}µs")
        
        # Check normalization
        assert abs(delays.min()) < 1e-10, "Minimum delay should be 0"
        assert len(delays) == tx.n_elements, "Should have delays for all elements"
    
    print("\n✓ 3D focusing OK")
    print()


def test_3d_steering():
    """Test 3D beam steering."""
    print("=" * 70)
    print("Test 6: 3D Beam Steering")
    print("=" * 70)
    
    tx = MatrixTransducer(n_elements_x=16, n_elements_y=16)
    
    # Test steering angles
    steering_angles = [
        ((0.0, 0.0), "No steering (boresight)"),
        ((np.deg2rad(10), 0.0), "10° in X-Z plane"),
        ((0.0, np.deg2rad(10)), "10° in Y-Z plane"),
        ((np.deg2rad(10), np.deg2rad(10)), "10° diagonal"),
    ]
    
    for angles, label in steering_angles:
        delays = tx.delays_for_steering_3d(angles)
        print(f"{label}:")
        print(f"  Angles: ({np.rad2deg(angles[0]):.1f}°, {np.rad2deg(angles[1]):.1f}°)")
        print(f"  Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
        
        # Check normalization
        assert abs(delays.min()) < 1e-10, "Minimum delay should be 0"
        assert len(delays) == tx.n_elements, "Should have delays for all elements"
    
    print("\n✓ 3D beam steering OK")
    print()


def test_bli_mask_generation():
    """Test BLI mask generation for matrix transducer."""
    print("=" * 70)
    print("Test 7: BLI Mask Generation")
    print("=" * 70)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = MatrixTransducer(n_elements_x=8, n_elements_y=8)
    
    # Create mask for one element
    elem_idx = 36  # Center element (row 4, col 4)
    indices, weights = tx.create_element_mask(grid, elem_idx, n_points_x=3, n_points_y=3)
    
    row, col = tx.get_element_row_col(elem_idx)
    print(f"BLI mask for element {elem_idx} (row={row}, col={col}):")
    print(f"  Sparse entries: {len(weights)}")
    print(f"  Weight sum: {weights.sum():.6f} (should be ~1.0)")
    print(f"  Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    # Check normalization
    assert abs(weights.sum() - 1.0) < 1e-5, "Weights should sum to 1.0"
    
    # Test memory efficiency for larger arrays
    print(f"\nMemory efficiency test:")
    dense_size = grid.nx * grid.ny * grid.nz * 4 / (1024**2)  # MB
    sparse_size = len(weights) * (4 + 3*4) / (1024**2)  # MB
    
    print(f"  Dense representation: {dense_size:.3f} MB")
    print(f"  Sparse representation: {sparse_size:.6f} MB")
    print(f"  Memory savings: {100*(1-sparse_size/dense_size):.1f}%")
    
    print("\n✓ BLI mask generation OK")
    print()


def test_grid_mapping():
    """Test mapping elements to grid."""
    print("=" * 70)
    print("Test 8: Grid Mapping")
    print("=" * 70)
    
    grid = Grid(nx=128, ny=128, nz=128, dx=1e-4)
    tx = MatrixTransducer(n_elements_x=16, n_elements_y=16)
    
    # Map elements to grid
    elem_indices = tx.map_to_grid(grid)
    
    print(f"Mapped {len(elem_indices)} elements to grid")
    print(f"Sample grid indices:")
    for i in [0, 8, 16, 128, 255]:
        ix, iy, iz = elem_indices[i]
        row, col = tx.get_element_row_col(i)
        print(f"  Element {i:3d} (row={row:2d}, col={col:2d}): grid[{ix}, {iy}, {iz}]")
    
    # Check that elements map to 2D grid structure
    # Elements in same row should have similar Y index
    first_row_indices = [elem_indices[i] for i in range(tx.n_elements_x)]
    first_row_y = [idx[1] for idx in first_row_indices]
    
    print(f"\nFirst row Y indices: {set(first_row_y)}")
    assert len(set(first_row_y)) == 1, "Elements in same row should have same Y index"
    
    # Elements in same column should have similar X index
    first_col_indices = [elem_indices[i*tx.n_elements_x] for i in range(tx.n_elements_y)]
    first_col_x = [idx[0] for idx in first_col_indices]
    
    print(f"First column X indices: {set(first_col_x)}")
    assert len(set(first_col_x)) == 1, "Elements in same column should have same X index"
    
    print("\n✓ Grid mapping OK")
    print()


def test_performance_scaling():
    """Test performance with different array sizes."""
    print("=" * 70)
    print("Test 9: Performance Scaling")
    print("=" * 70)
    
    import time
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    array_sizes = [(8, 8), (16, 16), (32, 32)]
    
    print(f"Mask generation timing:")
    for nx, ny in array_sizes:
        tx = MatrixTransducer(n_elements_x=nx, n_elements_y=ny)
        
        # Time single element mask generation
        start = time.time()
        indices, weights = tx.create_element_mask(grid, 0, n_points_x=3, n_points_y=3)
        elapsed = time.time() - start
        
        print(f"  {nx}×{ny} array ({tx.n_elements} elements):")
        print(f"    Single element: {elapsed*1000:.2f}ms, {len(weights)} entries")
        print(f"    Estimated for all: {elapsed*tx.n_elements:.2f}s")
    
    print("\n✓ Performance scaling OK")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("2D MATRIX TRANSDUCER TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_basic_creation()
        test_large_arrays()
        test_uniform_elements()
        test_element_positioning()
        test_3d_focusing()
        test_3d_steering()
        test_bli_mask_generation()
        test_grid_mapping()
        test_performance_scaling()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nKey features verified:")
        print("  ✓ Large 2D arrays (10s-100s of elements)")
        print("  ✓ Uniform element dimensions")
        print("  ✓ 2D element grid positioning")
        print("  ✓ 3D focusing capabilities")
        print("  ✓ 3D beam steering")
        print("  ✓ BLI mask generation for 2D arrays")
        print("  ✓ Efficient grid mapping")
        print("  ✓ Performance scaling")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
