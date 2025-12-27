"""
Test suite for 1.5D Transducer implementation

Tests the 1.5D transducer with multiple rows and variable element heights:
1. Element positioning in 2D grid
2. Variable row heights
3. 3D focusing capabilities
4. BLI mask generation
5. Element geometry
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from grid import Grid
from transducer import Transducer


def test_basic_creation():
    """Test basic 1.5D transducer creation."""
    print("=" * 70)
    print("Test 1: Basic 1.5D Transducer Creation")
    print("=" * 70)
    
    # Create 1.5D transducer with default parameters
    tx = Transducer(n_elements_per_row=32, n_rows=5)
    
    print(f"Created 1.5D transducer:")
    print(f"  Elements per row: {tx.n_elements_per_row}")
    print(f"  Number of rows: {tx.n_rows}")
    print(f"  Total elements: {tx.n_elements}")
    print(f"  Row pitch: {tx.row_pitch*1e3:.3f}mm")
    
    assert tx.n_elements == 32 * 5, "Total elements should be 160"
    assert len(tx.element_heights) == 160, "Should have height for each element"
    assert len(tx.row_heights) == 5, "Should have 5 row heights"
    
    print("\n✓ Basic creation OK")
    print()


def test_variable_row_heights():
    """Test 1.5D transducer with variable row heights."""
    print("=" * 70)
    print("Test 2: Variable Row Heights")
    print("=" * 70)
    
    # Create with custom row heights (Gaussian-like distribution)
    row_heights = np.array([0.0003, 0.0004, 0.0005, 0.0004, 0.0003])
    tx = Transducer(n_elements_per_row=16, n_rows=5, row_heights=row_heights)
    
    print(f"Custom row heights (mm): {row_heights * 1e3}")
    print(f"Element heights per row (mm):")
    
    for row_idx in range(tx.n_rows):
        row_start = row_idx * tx.n_elements_per_row
        row_end = row_start + tx.n_elements_per_row
        row_elem_heights = tx.element_heights[row_start:row_end]
        print(f"  Row {row_idx}: {row_elem_heights[0]*1e3:.4f}mm (all {tx.n_elements_per_row} elements)")
        assert np.allclose(row_elem_heights, row_heights[row_idx]), f"Row {row_idx} heights incorrect"
    
    print("\n✓ Variable row heights OK")
    print()


def test_element_positioning():
    """Test 2D element positioning."""
    print("=" * 70)
    print("Test 3: 2D Element Positioning")
    print("=" * 70)
    
    tx = Transducer(n_elements_per_row=8, n_rows=3, pitch=0.0003, row_pitch=0.0004)
    
    print(f"Element positions (first 5):")
    for i in range(min(5, tx.n_elements)):
        row, col = tx.get_element_row_col(i)
        x, y, z = tx.element_positions[i]
        print(f"  Element {i}: row={row}, col={col}, x={x*1e3:.3f}mm, y={y*1e3:.3f}mm, z={z*1e3:.3f}mm")
    
    # Check first element (row 0, col 0)
    row, col = tx.get_element_row_col(0)
    assert row == 0 and col == 0, "Element 0 should be at row 0, col 0"
    
    # Check last element of first row
    row, col = tx.get_element_row_col(7)
    assert row == 0 and col == 7, "Element 7 should be at row 0, col 7"
    
    # Check first element of second row
    row, col = tx.get_element_row_col(8)
    assert row == 1 and col == 0, "Element 8 should be at row 1, col 0"
    
    # Check positions are centered (using 3D element_positions)
    x_positions = tx.element_positions[:, 0]
    y_positions = tx.element_positions[:, 1]
    
    print(f"\nPosition ranges:")
    print(f"  X: [{x_positions.min()*1e3:.3f}, {x_positions.max()*1e3:.3f}]mm")
    print(f"  Y: [{y_positions.min()*1e3:.3f}, {y_positions.max()*1e3:.3f}]mm")
    
    # Check centering (should be symmetric around 0)
    assert abs(x_positions.mean()) < 1e-10, "X positions should be centered"
    assert abs(y_positions.mean()) < 1e-10, "Y positions should be centered"
    
    print("\n✓ Element positioning OK")
    print()


def test_3d_focusing():
    """Test 3D focusing capabilities."""
    print("=" * 70)
    print("Test 4: 3D Focusing")
    print("=" * 70)
    
    tx = Transducer(n_elements_per_row=16, n_rows=5)
    
    # Test focus at center
    focus_center = (0.0, 0.0, 0.03)  # 30mm depth, centered
    delays_center = tx.delays_for_focus_3d(focus_center)
    
    print(f"Focus at {focus_center}:")
    print(f"  Delay range: [{delays_center.min()*1e6:.3f}, {delays_center.max()*1e6:.3f}]µs")
    print(f"  Max delay: {delays_center.max()*1e6:.3f}µs")
    
    # Test focus offset in elevation
    focus_offset = (0.0, 0.001, 0.03)  # 30mm depth, 1mm offset in y
    delays_offset = tx.delays_for_focus_3d(focus_offset)
    
    print(f"\nFocus at {focus_offset}:")
    print(f"  Delay range: [{delays_offset.min()*1e6:.3f}, {delays_offset.max()*1e6:.3f}]µs")
    print(f"  Max delay: {delays_offset.max()*1e6:.3f}µs")
    
    # Delays should be different for offset focus
    assert not np.allclose(delays_center, delays_offset), "Delays should differ for different focus points"
    
    # Check normalization (minimum delay should be 0)
    assert abs(delays_center.min()) < 1e-10, "Minimum delay should be 0"
    assert abs(delays_offset.min()) < 1e-10, "Minimum delay should be 0"
    
    print("\n✓ 3D focusing OK")
    print()


def test_surface_point_generation():
    """Test surface point generation for 1.5D elements."""
    print("=" * 70)
    print("Test 5: Surface Point Generation")
    print("=" * 70)
    
    # Create transducer with variable row heights
    row_heights = np.array([0.0003, 0.0005, 0.0003])
    tx = Transducer(n_elements_per_row=8, n_rows=3, row_heights=row_heights)
    
    # Generate points for element in middle row (should use height 0.0005)
    elem_idx = 8  # First element of second row
    points = tx.generate_element_surface_points(elem_idx, n_points_x=5, n_points_y=5)
    
    row, col = tx.get_element_row_col(elem_idx)
    expected_height = row_heights[row]
    
    print(f"Element {elem_idx} (row={row}, col={col}):")
    print(f"  Expected height: {expected_height*1e3:.4f}mm")
    print(f"  Generated {len(points)} points")
    
    # Check point distribution (element_positions is now 3D)
    y_rel = points[:, 1] - tx.element_positions[elem_idx, 1]
    print(f"  Y range: [{y_rel.min()*1e3:.4f}, {y_rel.max()*1e3:.4f}]mm")
    print(f"  Expected: [{-expected_height/2*1e3:.4f}, {expected_height/2*1e3:.4f}]mm")
    
    # Verify points span the correct height
    assert abs(y_rel.max() - expected_height/2) < 1e-6, "Points should reach upper bound"
    assert abs(y_rel.min() + expected_height/2) < 1e-6, "Points should reach lower bound"
    
    print("\n✓ Surface point generation OK")
    print()


def test_bli_mask_generation():
    """Test BLI mask generation for 1.5D transducer."""
    print("=" * 70)
    print("Test 6: BLI Mask Generation")
    print("=" * 70)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements_per_row=8, n_rows=3)
    
    # Create mask for one element
    elem_idx = 12  # Middle row, middle column
    indices, weights = tx.create_element_mask(grid, elem_idx, n_points_x=3, n_points_y=3)
    
    row, col = tx.get_element_row_col(elem_idx)
    print(f"BLI mask for element {elem_idx} (row={row}, col={col}):")
    print(f"  Sparse entries: {len(weights)}")
    print(f"  Weight sum: {weights.sum():.6f} (should be ~1.0)")
    print(f"  Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    # Check normalization
    assert abs(weights.sum() - 1.0) < 1e-5, "Weights should sum to 1.0"
    
    # Create masks for multiple elements
    print(f"\nCreating masks for all {tx.n_elements} elements...")
    masks = tx.create_all_element_masks(grid, n_points_x=3, n_points_y=3)
    
    print(f"Created {len(masks)} masks")
    print(f"Sample masks:")
    for i in [0, tx.n_elements_per_row, tx.n_elements-1]:
        indices, weights = masks[i]
        row, col = tx.get_element_row_col(i)
        print(f"  Element {i} (row={row}, col={col}): {len(weights)} entries, sum={weights.sum():.6f}")
    
    print("\n✓ BLI mask generation OK")
    print()


def test_grid_mapping():
    """Test mapping elements to grid."""
    print("=" * 70)
    print("Test 7: Grid Mapping")
    print("=" * 70)
    
    grid = Grid(nx=128, ny=128, nz=128, dx=1e-4)
    tx = Transducer(n_elements_per_row=8, n_rows=3, pitch=0.0003)
    
    # Map elements to grid
    elem_indices = tx.map_to_grid(grid)
    
    print(f"Mapped {len(elem_indices)} elements to grid")
    print(f"Sample grid indices:")
    for i in [0, 4, 8]:
        ix, iy, iz = elem_indices[i]
        row, col = tx.get_element_row_col(i)
        print(f"  Element {i} (row={row}, col={col}): grid[{ix}, {iy}, {iz}]")
    
    # Check that different rows map to different y indices
    first_row_indices = [elem_indices[i] for i in range(tx.n_elements_per_row)]
    second_row_indices = [elem_indices[i] for i in range(tx.n_elements_per_row, 2*tx.n_elements_per_row)]
    
    # Y indices should differ between rows
    first_row_y = [idx[1] for idx in first_row_indices]
    second_row_y = [idx[1] for idx in second_row_indices]
    
    print(f"\nFirst row Y indices: {set(first_row_y)}")
    print(f"Second row Y indices: {set(second_row_y)}")
    
    assert set(first_row_y) != set(second_row_y), "Different rows should map to different Y indices"
    
    print("\n✓ Grid mapping OK")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("1.5D TRANSDUCER TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_basic_creation()
        test_variable_row_heights()
        test_element_positioning()
        test_3d_focusing()
        test_surface_point_generation()
        test_bli_mask_generation()
        test_grid_mapping()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nKey features verified:")
        print("  ✓ Multiple rows (3-7) with configurable heights")
        print("  ✓ Variable element heights per row")
        print("  ✓ 2D element positioning (x, y)")
        print("  ✓ 3D focusing and beam control")
        print("  ✓ BLI mask generation for 2D arrays")
        print("  ✓ Grid mapping with elevation support")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
