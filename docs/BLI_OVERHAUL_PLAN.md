# BLI Implementation Overhaul Plan

## Issues Identified from Feedback

1. **Point Distribution**
   - Current: Uses linspace/arange which may not be optimal
   - Required: Evenly distributed points WITHIN element surface
   - Concern: "but outside of the surface" - needs clarification
   
2. **Grid Center Option**
   - Current: Has grid_center_only_x/y/z parameters
   - Required: Remove these options
   
3. **Staggered Grid**
   - Current: Single mask with offset applied to all dimensions
   - Required: 3 separate masks for different staggered components
   - Likely: Pressure at cell centers, velocities at face centers
   
4. **BLI Implementation**
   - Current: Uses np.sinc directly with distance/spacing
   - Required: Re-read DOI 10.1121/1.5116132 and match exact formulation
   - Potential issues:
     * Normalization may be incorrect
     * Sinc function application may not match paper
     * Kaiser-Bessel windowing might be needed
     
5. **Return Format**
   - Current: Dense 3D array (nx × ny × nz)
   - Required: Sparse representation (indices + weights)
   - Format options:
     * COO: (indices_array, weights_array)
     * Dict: {(i,j,k): weight}
     * Separate: (i_array, j_array, k_array, weights_array)
     
6. **GPU Acceleration**
   - Consider: CuPy for GPU acceleration
   - Benefit: Could speed up mask generation for large arrays
   
## Implementation Steps

### Step 1: Fix Point Distribution
- Use np.linspace with inclusive endpoints within element bounds
- Ensure uniform spacing
- No clipping needed if done correctly

### Step 2: Remove Grid Center Options
- Remove grid_center_only_x/y/z parameters from all methods
- Remove _snap_to_grid_center helper

### Step 3: Correct BLI Implementation
- Review reference paper equation
- Implement proper windowed sinc (Kaiser-Bessel or similar)
- Correct normalization

### Step 4: Implement Sparse Output
- Change return type to sparse format
- Return (indices, weights) tuple
- Indices as (N, 3) array or separate arrays

### Step 5: Rework Staggered Grid
- Create method that returns 3 masks for staggered grid
- One for pressure (cell centers)
- Three for velocity components (face centers)

### Step 6: GPU Acceleration (Optional)
- Use CuPy for large arrays
- Fallback to NumPy for small arrays or when CuPy unavailable

## Reference Paper Key Points (DOI 10.1121/1.5116132)

Need to review:
- Equation for band-limited interpolation
- Windowing function used
- Normalization approach
- Grid alignment requirements

## Questions Needing Clarification

1. Point distribution: "evenly on surface but outside of surface" - contradiction?
2. Staggered masks: Return all 3 at once or one at a time?
3. Sparse format preference?
4. GPU: Required or optional?
