"""
BLI Implementation Notes - Reference Analysis

Based on DOI: 10.1121/1.5116132
"Band-limited interpolation of simulation data in the time-space domain"

Key concepts from paper:
1. Band-limited interpolation uses sinc functions but with proper windowing
2. The interpolation should preserve band-limited signals  
3. Kaiser-Bessel window is commonly used for practical implementations
4. Normalization ensures energy conservation

Standard FDTD Staggered Grid (Yee Grid):
- Pressure P at cell centers: (i, j, k)
- Velocity Vx at x-faces: (i+1/2, j, k)
- Velocity Vy at y-faces: (i, j+1/2, k)
- Velocity Vz at z-faces: (i, j, k+1/2)

For proper staggered grid support, need 4 masks:
1. Pressure mask (cell centers)
2. Vx mask (x-face centers) 
3. Vy mask (y-face centers)
4. Vz mask (z-face centers)

Sparse Representation Benefits:
- Memory: O(nnz) instead of O(nx*ny*nz)
- Speed: Only compute/store non-zero weights
- For BLI with kernel radius 3: ~343 cells per source point max
- For 1000 source points: ~343K entries vs full grid

Point Distribution:
- Should be uniform within element bounds
- Regular grid is fine (linspace)
- Key: All points INSIDE element (not outside)

"""

# Proposed sparse format:
# Return: (indices, weights)
# where:
#   indices: np.array of shape (N, 3) with dtype=int, containing (i, j, k) for each non-zero
#   weights: np.array of shape (N,) with dtype=float32, containing corresponding weights
#
# This allows easy sparse matrix construction or direct indexing:
#   mask[indices[:, 0], indices[:, 1], indices[:, 2]] = weights
