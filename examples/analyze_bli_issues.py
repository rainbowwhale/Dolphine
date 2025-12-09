"""
Test to analyze current BLI implementation issues

This script helps identify problems with the current implementation
to guide the overhaul.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from grid import Grid
from transducer import Transducer


def test_point_distribution():
    """Check if points are truly evenly distributed."""
    print("=" * 70)
    print("Test 1: Point Distribution Analysis")
    print("=" * 70)
    
    tx = Transducer(n_elements=1, element_width=0.0003, element_height=0.002)
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    
    # Test with grid alignment
    points_aligned = tx.generate_element_surface_points(0, grid=grid, grid_aligned=True)
    
    # Test without grid alignment
    points_uniform = tx.generate_element_surface_points(0, n_points_x=5, n_points_y=5, 
                                                        grid_aligned=False)
    
    print(f"\nGrid-aligned points: {len(points_aligned)}")
    x_spacing_aligned = np.diff(np.sort(np.unique(points_aligned[:, 0])))
    print(f"  X spacing range: [{x_spacing_aligned.min()*1e3:.4f}, {x_spacing_aligned.max()*1e3:.4f}]mm")
    
    print(f"\nUniform points: {len(points_uniform)}")
    x_spacing_uniform = np.diff(np.sort(np.unique(points_uniform[:, 0])))
    print(f"  X spacing range: [{x_spacing_uniform.min()*1e3:.4f}, {x_spacing_uniform.max()*1e3:.4f}]mm")
    
    # Check if points form a regular grid
    x_coords = points_uniform[:, 0] - tx.element_positions[0, 0]
    y_coords = points_uniform[:, 1]
    
    print(f"\nPoint bounds check:")
    print(f"  Element: {-tx.element_width/2*1e3:.3f} to {tx.element_width/2*1e3:.3f}mm (X)")
    print(f"  Points:  {x_coords.min()*1e3:.3f} to {x_coords.max()*1e3:.3f}mm (X)")
    print(f"  Element: {-tx.element_height/2*1e3:.3f} to {tx.element_height/2*1e3:.3f}mm (Y)")
    print(f"  Points:  {y_coords.min()*1e3:.3f} to {y_coords.max()*1e3:.3f}mm (Y)")
    
    # Check if points are on boundaries
    on_x_boundary = (np.abs(x_coords + tx.element_width/2) < 1e-10) | \
                    (np.abs(x_coords - tx.element_width/2) < 1e-10)
    on_y_boundary = (np.abs(y_coords + tx.element_height/2) < 1e-10) | \
                    (np.abs(y_coords - tx.element_height/2) < 1e-10)
    
    print(f"\nPoints on boundaries:")
    print(f"  X boundary: {on_x_boundary.sum()}/{len(points_uniform)}")
    print(f"  Y boundary: {on_y_boundary.sum()}/{len(points_uniform)}")


def test_mask_properties():
    """Analyze mask properties."""
    print("\n" + "=" * 70)
    print("Test 2: Mask Properties Analysis")
    print("=" * 70)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements=1, element_width=0.0003, element_height=0.002)
    
    # Generate mask
    mask = tx.band_limited_interpolation_mask(grid, 0, n_points_x=5, n_points_y=5)
    
    print(f"\nMask statistics:")
    print(f"  Sum: {mask.sum():.6f} (should be ~1.0)")
    print(f"  Max: {mask.max():.6f}")
    print(f"  Min: {mask.min():.6f}")
    print(f"  Non-zero: {np.count_nonzero(mask)}")
    print(f"  Sparsity: {100*(1 - np.count_nonzero(mask)/mask.size):.2f}%")
    
    # Check normalization per source point
    n_source_points = 5 * 5
    print(f"\n  Expected weight per source: {1.0/n_source_points:.6f}")
    
    # Memory analysis
    dense_memory = mask.nbytes / (1024**2)
    sparse_nnz = np.count_nonzero(mask)
    sparse_memory = (sparse_nnz * 4 * 4) / (1024**2)  # 3 ints + 1 float
    
    print(f"\nMemory usage:")
    print(f"  Dense: {dense_memory:.2f} MB")
    print(f"  Sparse (estimated): {sparse_memory:.2f} MB")
    print(f"  Savings: {100*(1-sparse_memory/dense_memory):.1f}%")


def test_staggered_grid():
    """Test staggered grid behavior."""
    print("\n" + "=" * 70)
    print("Test 3: Staggered Grid Analysis")
    print("=" * 70)
    
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements=1, element_width=0.0003, element_height=0.002)
    
    mask_normal = tx.band_limited_interpolation_mask(grid, 0, n_points_x=3, n_points_y=3, 
                                                     staggered=False)
    mask_staggered = tx.band_limited_interpolation_mask(grid, 0, n_points_x=3, n_points_y=3,
                                                        staggered=True)
    
    print(f"\nNormal grid mask:")
    print(f"  Sum: {mask_normal.sum():.6f}")
    print(f"  Max at: {np.unravel_index(mask_normal.argmax(), mask_normal.shape)}")
    
    print(f"\nStaggered grid mask:")
    print(f"  Sum: {mask_staggered.sum():.6f}")
    print(f"  Max at: {np.unravel_index(mask_staggered.argmax(), mask_staggered.shape)}")
    
    print(f"\nDifference:")
    diff = mask_staggered - mask_normal
    print(f"  Max diff: {np.abs(diff).max():.6f}")
    print(f"  Mean diff: {np.abs(diff).mean():.6f}")
    
    print("\nCurrent implementation: Single mask with half-cell offset")
    print("Required: Separate masks for pressure and velocity components")


def main():
    print("\n" + "=" * 70)
    print("BLI IMPLEMENTATION ANALYSIS")
    print("Identifying issues for overhaul")
    print("=" * 70 + "\n")
    
    try:
        test_point_distribution()
        test_mask_properties()
        test_staggered_grid()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nKey findings:")
        print("1. Points currently include boundaries (may bunch up after clipping)")
        print("2. Mask is dense - sparse representation would save ~99% memory")
        print("3. Staggered grid: single mask vs needed: separate P, Vx, Vy, Vz masks")
        print("4. BLI normalization appears correct for current implementation")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
