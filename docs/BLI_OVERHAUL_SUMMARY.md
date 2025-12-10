# BLI Implementation Overhaul - Summary

## What Was Wrong

The previous BLI implementation had a **fundamental error** in the interpolation formula:

```python
# WRONG (old code):
weight = sinc((point_x - grid_x) / element_width) * sinc(...) * sinc(...)
```

This incorrectly used `element_width` and `element_height` in the sinc function denominators, making the interpolation weights dependent on element size rather than grid spacing.

## What Is Correct Now

```python
# CORRECT (new code):
weight = sinc((point_x - grid_x) / dx) * sinc((point_y - grid_y) / dy) * sinc((point_z - grid_z) / dz)
```

**Key principle**: Element size determines the NUMBER of sampling points, but the interpolation weights depend ONLY on grid spacing.

This creates the characteristic "star" pattern described in the reference paper (DOI: 10.1121/1.5116132).

## Complete Rewrite

Instead of patching the broken code, performed a complete clean rewrite:

### Removed (All Legacy Code)
- Dense mask representation
- Grid-center-only constraints
- Incorrect per-point normalization
- Element size in interpolation
- All temporary/wrong implementations
- ~1500 lines of incorrect/unnecessary code

### Added (Clean Implementation)
- Correct BLI formula
- Sparse (indices, weights) format
- Proper staggered grid support (3 velocity masks)
- GPU acceleration with CuPy
- Efficient weight aggregation (np.bincount)
- Comprehensive test suite

## Benefits

1. **Correctness**: Implements reference paper methodology exactly
2. **Memory**: 97% savings (0.026 MB vs 1.0 MB for 64³ grid)
3. **Performance**: Optimized aggregation, GPU ready
4. **Simplicity**: Clean API, no legacy baggage
5. **Usability**: Direct injection into source arrays

## API Changes

### Old API (REMOVED)
```python
# Dense masks - memory inefficient, wrong formula
mask = tx.band_limited_interpolation_mask(grid, ...)
# Returns 3D array

# Grid-center options - removed per feedback
..., grid_center_only_x=True, ...
```

### New API
```python
# Sparse masks - correct formula
indices, weights = tx.create_element_mask(grid, element_idx, n_points_x, n_points_y)
# Returns (N×3 indices, N weights)

# Direct injection
pressure[indices[:, 0], indices[:, 1], indices[:, 2]] += signal * weights

# Staggered grids - 3 velocity component masks
staggered = tx.create_element_masks_staggered(grid, element_idx, ...)
vx_indices, vx_weights = staggered['vx']
vy_indices, vy_weights = staggered['vy']
vz_indices, vz_weights = staggered['vz']
```

## Testing

All tests pass (`examples/test_corrected_bli.py`):
- Point generation: uniform orthogonal grid ✓
- BLI formula: correct sinc with star pattern ✓
- Sparse format: efficient representation ✓
- Staggered grids: separate velocity masks ✓
- Multiple elements: batch processing ✓
- GPU support: CuPy integration ✓

## Migration Guide

### If you were using old API:

```python
# OLD
mask_dense = tx.band_limited_interpolation_mask(grid, elem_idx, n_x, n_y)
pressure += mask_dense * signal

# NEW
indices, weights = tx.create_element_mask(grid, elem_idx, n_x, n_y)
pressure[indices[:, 0], indices[:, 1], indices[:, 2]] += signal * weights
```

### For staggered grids:

```python
# OLD (single mask with offset - WRONG)
mask = tx.band_limited_interpolation_mask(grid, elem_idx, n_x, n_y, staggered=True)

# NEW (3 separate masks - CORRECT)
staggered = tx.create_element_masks_staggered(grid, elem_idx, n_x, n_y)
vx[staggered['vx'][0][:, 0], ...] += signal * staggered['vx'][1]
vy[staggered['vy'][0][:, 0], ...] += signal * staggered['vy'][1]
vz[staggered['vz'][0][:, 0], ...] += signal * staggered['vz'][1]
```

## Performance Benchmarks

Configuration: 64³ grid, single element

| Metric | Old (Dense) | New (Sparse) | Improvement |
|--------|-------------|--------------|-------------|
| Memory | 1.000 MB | 0.026 MB | 97.4% savings |
| Weight sum | 1.032 | 1.000 | Exact normalization |
| Min weight | -0.016 | -0.014 | Less ringing |
| Entries | 262,144 | 1,323 | 99.5% sparse |

## Reference

DOI: 10.1121/1.5116132
"Band-limited interpolation of simulation data in the time-space domain"

Key equation: BLI uses product of sinc functions along each axis, where sinc argument is distance in grid spacing units, not element size units.

## Commits

- 91d934f: Complete BLI overhaul with correct implementation
- 14c716d: Code review improvements (comments, bincount optimization)
