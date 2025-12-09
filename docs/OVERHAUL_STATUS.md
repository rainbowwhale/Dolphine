# BLI Overhaul - Status and Next Steps

## Status: Awaiting Clarification

I've asked for clarification on the following points in comment 3560073617:

1. **Point distribution method**: Uniform interior vs other distribution (Fibonacci, jittered, etc.)
2. **Staggered grid format**: Return all 3-4 masks at once, or one at a time?
3. **Sparse format preference**: COO, dict, or separate arrays?
4. **GPU requirement**: Required or optional?

## Analysis Completed

### Issues Identified

1. **Point Distribution**
   - Current: 10/25 points on boundaries after linspace+clip
   - Fix: Use interior points only (n+1 intervals, take interior n points)

2. **Memory Efficiency**
   - Current: Dense 1 MB for 64³ grid
   - Sparse: ~0.01 MB (98.7% savings)
   - Solution: Return (indices, weights) tuple

3. **Staggered Grid**
   - Current: Single mask with uniform offset
   - Needed: Separate masks for P, Vx, Vy, Vz at different grid positions

4. **BLI Implementation**
   - Current: Bare sinc without windowing
   - Issue: Causes negative weights (-0.016), Gibbs phenomenon
   - Solution: Add Kaiser-Bessel windowing

5. **Grid Center Options**
   - Current: Has grid_center_only_x/y/z parameters
   - Requested: Remove these

### Prepared Components

#### Documentation
- `docs/BLI_OVERHAUL_PLAN.md`: Overall plan
- `docs/BLI_REFERENCE_NOTES.md`: Reference paper notes
- `docs/IMPLEMENTATION_APPROACH.md`: Detailed implementation strategy

#### Analysis Tools
- `examples/analyze_bli_issues.py`: Script analyzing current issues
- Results confirm: 99% sparsity, points on boundaries, single staggered mask

#### Draft Utilities
- Interior point generation (tested ✓)
- Kaiser-Bessel window function (tested ✓)
- Sparse/dense conversion (tested ✓)
- Band-limited weight computation (ready)

## Implementation Plan (Once Clarified)

### Phase 1: Point Distribution Fix
1. Replace linspace with interior point generation
2. Remove clipping logic
3. Verify points stay within bounds
4. Test: No points on boundaries

### Phase 2: Remove Grid Center Options
1. Delete grid_center_only_x/y/z from all method signatures
2. Remove _snap_to_grid_center() helper
3. Update documentation
4. Update tests

### Phase 3: Sparse Representation
1. Create band_limited_interpolation_mask_sparse()
2. Return (indices, weights) tuple
3. Test memory savings
4. Keep dense version for backward compatibility (deprecated)

### Phase 4: Correct BLI Implementation
1. Add Kaiser-Bessel windowing
2. Fix normalization if needed
3. Verify weights sum to ~1.0
4. Test: No large negative weights

### Phase 5: Staggered Grid Rework
1. Create create_staggered_masks() returning dict
2. Generate separate masks for P, Vx, Vy, Vz
3. Apply correct offsets for each component
4. Test: Masks at correct grid positions

### Phase 6: GPU Support (If Requested)
1. Add CuPy support with numpy fallback
2. Benchmark performance gains
3. Document GPU requirements

## Testing Strategy

1. Unit tests for each component
2. Memory profiling (confirm savings)
3. Accuracy tests (compare with reference)
4. Performance benchmarks
5. Backward compatibility checks

## Migration Path

- Keep old methods with deprecation warnings
- Add new _sparse suffix methods
- Provide conversion utilities
- Update examples gradually
- Document breaking changes clearly

## Files Ready for Changes

Once clarification received:
- `src/transducer.py`: Main implementation
- `examples/*.py`: Update test files
- `docs/TRANSDUCER_MASK_API.md`: Update documentation
- `README.md`: Update feature list

## Estimated Effort

- Phase 1-2: ~2 hours (straightforward)
- Phase 3-4: ~4 hours (moderate complexity)
- Phase 5: ~3 hours (requires careful testing)
- Phase 6: ~2 hours (if needed)
- Testing/Documentation: ~3 hours

Total: 12-14 hours of development time

## Next Action

Waiting for user response to clarification questions before proceeding with implementation.
