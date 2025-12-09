"""Example: Multi-row transducer array demonstration.

This example shows how to use the Transducer class with multiple rows (n_rows > 1),
which is useful for 1.5D or 1.75D arrays that have elevation focus control.
"""
import time
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from grid import Grid
from transducer import Transducer


def main():
    print("=" * 60)
    print("Multi-row Transducer Array Demonstration")
    print("=" * 60)
    
    # Example 1: Single-row linear array (traditional)
    print("\n1. Single-row linear array (64 elements)")
    tx_1row = Transducer(n_elements=64, pitch=0.0003, element_height=0.00028, n_rows=1)
    print(f"   - Total elements: {tx_1row.n_elements}")
    print(f"   - Number of rows: {tx_1row.n_rows}")
    print(f"   - Element height: {tx_1row.element_height * 1000:.2f} mm")
    print(f"   - Position shape: {tx_1row.element_positions.shape}")
    print(f"   - Y position range: [{tx_1row.element_positions[:, 1].min():.6f}, {tx_1row.element_positions[:, 1].max():.6f}] m")
    
    # Example 2: Two-row array (1.5D array)
    print("\n2. Two-row array - 1.5D transducer (64 elements total)")
    tx_2row = Transducer(n_elements=64, pitch=0.0003, element_height=0.00028, n_rows=2)
    print(f"   - Total elements: {tx_2row.n_elements}")
    print(f"   - Number of rows: {tx_2row.n_rows}")
    print(f"   - Elements per row: {tx_2row.n_elements // tx_2row.n_rows}")
    print(f"   - Position shape: {tx_2row.element_positions.shape}")
    print(f"   - Y position range: [{tx_2row.element_positions[:, 1].min():.6f}, {tx_2row.element_positions[:, 1].max():.6f}] m")
    
    # Example 3: Four-row array
    print("\n3. Four-row array (64 elements total)")
    tx_4row = Transducer(n_elements=64, pitch=0.0003, element_height=0.00028, n_rows=4)
    print(f"   - Total elements: {tx_4row.n_elements}")
    print(f"   - Number of rows: {tx_4row.n_rows}")
    print(f"   - Elements per row: {tx_4row.n_elements // tx_4row.n_rows}")
    print(f"   - Position shape: {tx_4row.element_positions.shape}")
    print(f"   - Y position range: [{tx_4row.element_positions[:, 1].min():.6f}, {tx_4row.element_positions[:, 1].max():.6f}] m")
    
    # Example 4: 2D matrix array (8x8 = 64 elements)
    print("\n4. 2D Matrix array (8x8 = 64 elements)")
    tx_matrix = Transducer(n_elements=64, pitch=0.0005, element_height=0.0005, n_rows=8)
    print(f"   - Total elements: {tx_matrix.n_elements}")
    print(f"   - Number of rows: {tx_matrix.n_rows}")
    print(f"   - Elements per row: {tx_matrix.n_elements // tx_matrix.n_rows}")
    print(f"   - Position shape: {tx_matrix.element_positions.shape}")
    print(f"   - X position range: [{tx_matrix.element_positions[:, 0].min():.6f}, {tx_matrix.element_positions[:, 0].max():.6f}] m")
    print(f"   - Y position range: [{tx_matrix.element_positions[:, 1].min():.6f}, {tx_matrix.element_positions[:, 1].max():.6f}] m")
    
    # Example 5: Multi-row array with varying row heights (NEW FEATURE)
    print("\n5. Multi-row array with varying row heights (5 rows: 1, 3, 5, 3, 1 mm)")
    row_heights_mm = [1, 3, 5, 3, 1]  # Heights in mm
    row_heights_m = [h / 1000.0 for h in row_heights_mm]  # Convert to meters
    tx_varying = Transducer(n_elements=100, pitch=0.0003, n_rows=5, row_heights=row_heights_m)
    print(f"   - Total elements: {tx_varying.n_elements}")
    print(f"   - Number of rows: {tx_varying.n_rows}")
    print(f"   - Elements per row: {tx_varying.n_elements // tx_varying.n_rows}")
    print(f"   - Row heights: {[f'{h:.1f}' for h in (tx_varying.row_heights * 1000)]} mm")
    print(f"   - Mean element height: {tx_varying.element_height * 1000:.2f} mm")
    print(f"   - Position shape: {tx_varying.element_positions.shape}")
    
    # Example 6: Focusing with multi-row array
    print("\n6. Focusing demonstration")
    print("   2D focus point (x, z) - backward compatible:")
    focus_2d = (0.0, 0.03)  # 3 cm depth, centered
    delays_2d = tx_2row.delays_for_focus(focus_2d)
    print(f"   - Focus point: x={focus_2d[0]:.3f} m, z={focus_2d[1]:.3f} m")
    print(f"   - Delay range: [{delays_2d.min():.6e}, {delays_2d.max():.6e}] seconds")
    
    print("\n   3D focus point (x, y, z) - full control:")
    focus_3d = (0.005, 0.002, 0.03)  # 5mm lateral, 2mm elevation, 3cm depth
    delays_3d = tx_4row.delays_for_focus(focus_3d)
    print(f"   - Focus point: x={focus_3d[0]:.3f} m, y={focus_3d[1]:.3f} m, z={focus_3d[2]:.3f} m")
    print(f"   - Delay range: [{delays_3d.min():.6e}, {delays_3d.max():.6e}] seconds")
    
    # Example 7: Grid mapping
    print("\n7. Grid mapping demonstration")
    grid = Grid(nx=128, ny=64, nz=256, dx=1e-4)
    
    # Map single-row transducer
    idx_1row = tx_1row.map_to_grid(grid, z0=0.0)
    iy_values_1row = [idx[1] for idx in idx_1row]
    print(f"   Single-row transducer:")
    print(f"   - Mapped to {len(idx_1row)} grid positions")
    print(f"   - Y indices (unique): {sorted(set(iy_values_1row))}")
    
    # Map multi-row transducer
    idx_4row = tx_4row.map_to_grid(grid, z0=0.0)
    iy_values_4row = [idx[1] for idx in idx_4row]
    print(f"\n   Four-row transducer:")
    print(f"   - Mapped to {len(idx_4row)} grid positions")
    print(f"   - Y indices (unique): {sorted(set(iy_values_4row))}")
    
    print("\n" + "=" * 60)
    print("Key Features Demonstrated:")
    print("=" * 60)
    print("✓ Single-row arrays (n_rows=1) - traditional linear arrays")
    print("✓ Multi-row arrays (n_rows>1) - 1.5D/1.75D arrays")
    print("✓ 2D matrix arrays - full 3D beam control")
    print("✓ Element height attribute for physical dimensions")
    print("✓ Per-row element heights - each row can have different height")
    print("✓ 3D element positions (x, y, z) for all elements")
    print("✓ Backward compatible 2D focus points (x, z)")
    print("✓ Full 3D focus points (x, y, z) for elevation control")
    print("✓ Grid mapping with proper 3D indexing")
    print("=" * 60)


if __name__ == '__main__':
    main()
