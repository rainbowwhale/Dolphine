"""
Proposed BLI Overhaul Implementation Plan

Based on analysis and feedback, here's the implementation strategy:

## 1. Point Distribution Fix

Current issue: Points include boundaries (10/25 points on boundaries)
Solution: Use interior points only

```python
# Instead of:
x_samples = np.linspace(-width/2, width/2, n_points)  # includes boundaries

# Use:
x_samples = np.linspace(-width/2, width/2, n_points+2)[1:-1]  # excludes boundaries
# OR
dx = width / (n_points + 1)
x_samples = np.arange(1, n_points+1) * dx - width/2  # interior points only
```

## 2. Remove Grid Center Options

- Delete grid_center_only_x/y/z parameters
- Delete _snap_to_grid_center() method
- Update all method signatures
- Remove from documentation

## 3. Sparse Mask Representation

Current: Dense 3D array (1 MB for 64³ grid)
Proposed: (indices, weights) tuple

```python
def band_limited_interpolation_mask_sparse(self, grid, element_idx, ...):
    """
    Returns:
        indices: np.array of shape (N, 3), dtype=int32, grid indices (i, j, k)
        weights: np.array of shape (N,), dtype=float32, corresponding weights
    """
    indices_list = []
    weights_list = []
    
    for point in points:
        # ... compute local indices and weights ...
        indices_list.append(local_indices)
        weights_list.append(local_weights)
    
    indices = np.vstack(indices_list)
    weights = np.concatenate(weights_list)
    
    return indices, weights
```

Usage:
```python
indices, weights = tx.band_limited_interpolation_mask_sparse(grid, 0)
# To create dense mask:
mask = np.zeros((grid.nx, grid.ny, grid.nz))
mask[indices[:, 0], indices[:, 1], indices[:, 2]] = weights
```

## 4. Staggered Grid Rework

Instead of single mask with offset, create separate masks for each component:

```python
def create_staggered_masks(self, grid, element_idx, ...):
    """
    Returns dict with masks for each component:
        {
            'pressure': (indices_p, weights_p),    # at cell centers
            'vx': (indices_vx, weights_vx),        # at x-face centers  
            'vy': (indices_vy, weights_vy),        # at y-face centers
            'vz': (indices_vz, weights_vz),        # at z-face centers
        }
    """
    offsets = {
        'pressure': (0.0, 0.0, 0.0),
        'vx': (grid.dx/2, 0.0, 0.0),
        'vy': (0.0, grid.dy/2, 0.0),
        'vz': (0.0, 0.0, grid.dz/2),
    }
    
    masks = {}
    for component, offset in offsets.items():
        indices, weights = self._compute_mask_with_offset(
            grid, element_idx, offset, ...
        )
        masks[component] = (indices, weights)
    
    return masks
```

## 5. BLI Correction

Need to verify the sinc implementation matches the reference. The standard band-limited
interpolation uses:

```
w(x) = sinc(x) * window(x)
```

where sinc(x) = sin(πx)/(πx) and window is typically Kaiser-Bessel.

Current implementation uses bare sinc without windowing, which can cause:
- Gibbs phenomenon (ringing)
- Slow convergence
- Negative weights (as seen in analysis: min = -0.015653)

Proper implementation:
```python
def kaiser_bessel_window(x, radius, beta=8.6):
    """Kaiser-Bessel window for band-limited interpolation."""
    alpha = beta / radius
    x_norm = np.abs(x) / radius
    
    # Outside radius: zero
    window = np.zeros_like(x)
    inside = x_norm <= 1.0
    
    if np.any(inside):
        from scipy.special import i0  # modified Bessel function
        window[inside] = i0(alpha * np.sqrt(1 - x_norm[inside]**2)) / i0(alpha)
    
    return window

def bandlimited_weight(dist_x, dist_y, dist_z, radius):
    """Compute band-limited interpolation weight with Kaiser-Bessel window."""
    # Sinc functions
    sinc_x = np.sinc(dist_x)
    sinc_y = np.sinc(dist_y)
    sinc_z = np.sinc(dist_z)
    
    # Windows
    win_x = kaiser_bessel_window(dist_x, radius)
    win_y = kaiser_bessel_window(dist_y, radius)
    win_z = kaiser_bessel_window(dist_z, radius)
    
    # Combined weight
    return sinc_x * sinc_y * sinc_z * win_x * win_y * win_z
```

## 6. GPU Acceleration (Optional)

Use CuPy when available:
```python
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

def _get_array_module(use_gpu=False):
    if use_gpu and HAS_CUPY:
        return cp
    return np
```

## Testing Strategy

1. Verify points are interior-only
2. Check sparse representation memory savings
3. Validate staggered masks at correct positions
4. Compare windowed vs non-windowed sinc
5. Benchmark GPU vs CPU performance

## Migration Path

To maintain backward compatibility:
- Keep old methods with deprecation warnings
- Add new methods with "_sparse" suffix
- Provide conversion utilities

```python
def band_limited_interpolation_mask(self, ...):
    """Legacy method - returns dense mask."""
    import warnings
    warnings.warn("Dense masks deprecated, use band_limited_interpolation_mask_sparse",
                  DeprecationWarning)
    indices, weights = self.band_limited_interpolation_mask_sparse(...)
    mask = np.zeros((grid.nx, grid.ny, grid.nz))
    mask[indices[:, 0], indices[:, 1], indices[:, 2]] = weights
    return mask
```
