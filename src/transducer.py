"""
Clean BLI Implementation - Correct Version

Based on feedback and reference DOI: 10.1121/1.5116132

Key corrections:
1. BLI formula: sinc((point_pos - grid_pos) / grid_spacing) per axis, multiply together
2. Element size only determines number of sample points, NOT interpolation calculation
3. Returns sparse format (indices, weights) for direct use in source injection
4. Supports staggered grids: 3 separate velocity component masks
5. GPU acceleration with CuPy
"""
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


class Transducer:
    """Transducer element geometry and beamforming utilities.
    
    Includes band-limited interpolation (BLI) for distributed source injection.
    Reference: https://doi.org/10.1121/1.5116132
    """
    
    def __init__(self, n_elements=64, pitch=0.0003, element_width=0.00028, kerf=0.00002,
                 center_freq=5e6, c=1540.0, element_height=0.010):
        """
        Args:
            n_elements: Number of transducer elements
            pitch: Center-to-center spacing between elements (m)
            element_width: Width of each element (lateral, m)
            kerf: Gap between elements (m)
            center_freq: Center frequency (Hz)
            c: Speed of sound (m/s)
            element_height: Height of element (elevation, m)
        """
        self.n_elements = n_elements
        self.pitch = pitch
        self.element_width = element_width
        self.kerf = kerf
        self.center_freq = center_freq
        self.c = c
        self.element_height = element_height
        
        # Generate element center positions along x-axis centered at zero
        x_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch
        self.element_positions = np.stack([x_positions, np.zeros_like(x_positions)], axis=1)  # (x, z=0)

    def delays_for_focus(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for each element to focus at `focus_point` (x,z) in meters."""
        if speed_of_sound is None:
            c = self.c
        else:
            c = speed_of_sound
        pos = self.element_positions
        dx = pos[:, 0] - focus_point[0]
        dz = focus_point[1] - 0.0
        distances = np.sqrt(dx ** 2 + dz ** 2)
        delays = distances / c
        delays -= delays.min()
        return delays

    def apodization_hanning(self):
        """Return Hanning apodization weights across elements."""
        return np.hanning(self.n_elements)

    def map_to_grid(self, grid, z0=0.0):
        """Map element centers to grid indices (ix, iy, iz) using `Grid` object."""
        idx = []
        for (x, _) in self.element_positions:
            ix, iy, iz = grid.world_to_index(x, 0.0, z0)
            idx.append((ix, iy, iz))
        return idx

    def generate_element_surface_points(self, element_idx, n_points_x, n_points_y):
        """
        Generate uniformly distributed points on rectangular element surface.
        
        Points are distributed on a regular orthogonal grid within element boundaries.
        
        Args:
            element_idx: Index of the element (0 to n_elements-1)
            n_points_x: Number of sample points along element width (lateral)
            n_points_y: Number of sample points along element height (elevation)
            
        Returns:
            points: Array of shape (n_points_x * n_points_y, 3) with (x, y, z) coordinates
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center position
        center_x = self.element_positions[element_idx, 0]
        center_z = self.element_positions[element_idx, 1]
        
        # Generate uniform grid of points within element bounds
        # Use linspace with inclusive endpoints for uniform coverage
        x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
        y_samples = np.linspace(-self.element_height/2, self.element_height/2, n_points_y)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_samples, y_samples, indexing='xy')
        
        # Flatten and offset by element center
        x_coords = center_x + xx.flatten()
        y_coords = yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        return points

    def band_limited_interpolation_weights(self, grid, points, z0=0.0, kernel_radius=3,
                                          staggered_component=None, use_gpu=False):
        """
        Compute BLI weights for source points on grid using vectorized calculation.
        
        Implements correct BLI formula from reference:
        weight(grid_node) = sinc((px - gx)/dx) * sinc((py - gy)/dy) * sinc((pz - gz)/dz)
        
        Fully vectorized implementation as per feedback:
        - px = (points[:,0] - grid.x_vec[0]) / grid.dx
        - ix = floor(px)
        - rx = px - ix
        - sx = sinc(rx + bli_x) for bli_x in range(-kernel_radius, kernel_radius+1)
        
        This creates a "star" pattern as described in the reference paper.
        
        Args:
            grid: Grid object with axis vectors (x_vec, y_vec, z_vec)
            points: Array of (x, y, z) source point coordinates
            z0: Z-position of transducer surface (m)
            kernel_radius: Sinc kernel radius in grid cells (e.g., 3 means -3 to +3)
            staggered_component: None for pressure (centered), 'x', 'y', or 'z' for velocity
            use_gpu: Use CuPy for GPU acceleration if available
            
        Returns:
            indices: (N, 3) array of grid indices (i, j, k)
            weights: (N,) array of corresponding weights
        """
        # Select array module (for GPU or CPU)
        if use_gpu and HAS_CUPY:
            xp = cp
            points = cp.asarray(points)
            grid_x_vec = cp.asarray(grid.x_vec)
            grid_y_vec = cp.asarray(grid.y_vec)
            grid_z_vec = cp.asarray(grid.z_vec)
        else:
            xp = np
            grid_x_vec = grid.x_vec
            grid_y_vec = grid.y_vec
            grid_z_vec = grid.z_vec
        
        # Pre-compute reciprocals for better performance (per reviewer suggestion)
        inv_dx = 1.0 / grid.dx
        inv_dy = 1.0 / grid.dy
        inv_dz = 1.0 / grid.dz
        
        # Set z-coordinate
        points_with_z = points.copy()
        points_with_z[:, 2] = z0
        
        # Grid offsets for staggered components
        offset_x, offset_y, offset_z = 0.0, 0.0, 0.0
        if staggered_component == 'x':
            offset_x = grid.dx / 2.0
        elif staggered_component == 'y':
            offset_y = grid.dy / 2.0
        elif staggered_component == 'z':
            offset_z = grid.dz / 2.0
        
        # Vectorized calculation per axis
        # px = (points[:, 0] - grid.x_vec[0]) / grid.dx
        px = (points_with_z[:, 0] + offset_x - grid_x_vec[0]) * inv_dx  # Shape: (n_points,)
        py = (points_with_z[:, 1] + offset_y - grid_y_vec[0]) * inv_dy
        pz = (points_with_z[:, 2] + offset_z - grid_z_vec[0]) * inv_dz
        
        # ix = floor(px)
        ix = xp.floor(px).astype(int)  # Shape: (n_points,)
        iy = xp.floor(py).astype(int)
        iz = xp.floor(pz).astype(int)
        
        # rx = px - ix (fractional part)
        rx = px - ix  # Shape: (n_points,)
        ry = py - iy
        rz = pz - iz
        
        # Create BLI offset ranges: [-kernel_radius, ..., +kernel_radius]
        bli_range = xp.arange(-kernel_radius, kernel_radius + 1)  # Shape: (2*kernel_radius+1,)
        
        # Compute sinc for all combinations: sinc(rx + bli_x)
        # Broadcasting: (n_points, 1) + (1, bli_range) = (n_points, bli_range)
        sinc_x_all = xp.sinc(rx[:, None] + bli_range[None, :])  # Shape: (n_points, 2*kr+1)
        sinc_y_all = xp.sinc(ry[:, None] + bli_range[None, :])
        sinc_z_all = xp.sinc(rz[:, None] + bli_range[None, :])
        
        # Generate indices for all points and bli offsets
        # ix + bli_range: (n_points, 1) + (1, bli_range) = (n_points, bli_range)
        ix_all = ix[:, None] + bli_range[None, :]  # Shape: (n_points, 2*kr+1)
        iy_all = iy[:, None] + bli_range[None, :]
        iz_all = iz[:, None] + bli_range[None, :]
        
        # Filter out-of-bounds indices and compute weights
        indices_list = []
        weights_list = []
        
        n_points = points_with_z.shape[0]
        n_bli = len(bli_range)
        
        # For each point, compute 3D kernel weights
        for p_idx in range(n_points):
            # Get valid index ranges for this point
            ix_range = ix_all[p_idx]
            iy_range = iy_all[p_idx]
            iz_range = iz_all[p_idx]
            
            # Mask for valid indices
            valid_x = (ix_range >= 0) & (ix_range < grid.nx)
            valid_y = (iy_range >= 0) & (iy_range < grid.ny)
            valid_z = (iz_range >= 0) & (iz_range < grid.nz)
            
            # Get valid indices
            ix_valid = ix_range[valid_x]
            iy_valid = iy_range[valid_y]
            iz_valid = iz_range[valid_z]
            
            # Get corresponding sinc values
            sinc_x_valid = sinc_x_all[p_idx, valid_x]
            sinc_y_valid = sinc_y_all[p_idx, valid_y]
            sinc_z_valid = sinc_z_all[p_idx, valid_z]
            
            if len(ix_valid) == 0 or len(iy_valid) == 0 or len(iz_valid) == 0:
                continue
            
            # Create 3D weight grid using outer products (star pattern)
            # weights_3d = sinc_x[:, None, None] * sinc_y[None, :, None] * sinc_z[None, None, :]
            weights_3d = xp.einsum('i,j,k->ijk', sinc_x_valid, sinc_y_valid, sinc_z_valid)
            
            # Create index meshgrid
            ix_grid, iy_grid, iz_grid = xp.meshgrid(ix_valid, iy_valid, iz_valid, indexing='ij')
            
            # Flatten
            local_indices = xp.stack([ix_grid.flatten(), iy_grid.flatten(), iz_grid.flatten()], axis=1)
            local_weights = weights_3d.flatten()
            
            indices_list.append(local_indices)
            weights_list.append(local_weights)
        
        # Combine all points
        if len(indices_list) > 0:
            indices = xp.vstack(indices_list)
            weights = xp.concatenate(weights_list)
            
            # Convert back to numpy if using GPU
            if use_gpu and HAS_CUPY:
                indices = cp.asnumpy(indices)
                weights = cp.asnumpy(weights)
            
            # Aggregate weights for duplicate indices using linear indexing (per reviewer suggestion)
            # Convert 3D indices to linear indices for fast unique/aggregation
            linear_indices = np.ravel_multi_index(indices.T, (grid.nx, grid.ny, grid.nz))
            unique_linear_indices, inverse = np.unique(linear_indices, return_inverse=True)
            
            # Aggregate using bincount (efficient)
            aggregated_weights = np.bincount(inverse, weights=weights).astype(np.float32)
            
            # Convert back to 3D indices
            indices = np.vstack(np.unravel_index(unique_linear_indices, (grid.nx, grid.ny, grid.nz))).T
            weights = aggregated_weights
            
            # Normalize so total weight sums to 1.0
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                weights = weights / weight_sum
        else:
            indices = np.zeros((0, 3), dtype=np.int32)
            weights = np.zeros(0, dtype=np.float32)
        
        return indices.astype(np.int32), weights.astype(np.float32)

    def create_element_mask(self, grid, element_idx, n_points_x, n_points_y, z0=0.0,
                           kernel_radius=3, staggered_component=None, use_gpu=False):
        """
        Create BLI mask for a single element.
        
        Args:
            grid: Grid object
            element_idx: Element index
            n_points_x: Number of sample points along width
            n_points_y: Number of sample points along height
            z0: Z-position of transducer surface
            kernel_radius: Sinc kernel radius (grid cells)
            staggered_component: None, 'x', 'y', or 'z' for staggered grids
            use_gpu: Use GPU acceleration
            
        Returns:
            indices: (N, 3) array of grid indices
            weights: (N,) array of weights
        """
        # Generate surface points
        points = self.generate_element_surface_points(element_idx, n_points_x, n_points_y)
        
        # Compute BLI weights
        indices, weights = self.band_limited_interpolation_weights(
            grid, points, z0, kernel_radius, staggered_component, use_gpu
        )
        
        return indices, weights

    def create_element_masks_staggered(self, grid, element_idx, n_points_x, n_points_y,
                                      z0=0.0, kernel_radius=3, use_gpu=False):
        """
        Create staggered grid masks for velocity components.
        
        Returns 3 masks for velocity components Vx, Vy, Vz where each component
        is staggered along its corresponding axis.
        
        Args:
            grid: Grid object
            element_idx: Element index
            n_points_x: Number of sample points along width
            n_points_y: Number of sample points along height
            z0: Z-position of transducer surface
            kernel_radius: Sinc kernel radius
            use_gpu: Use GPU acceleration
            
        Returns:
            dict: {'vx': (indices, weights), 'vy': (indices, weights), 'vz': (indices, weights)}
        """
        # Generate surface points once
        points = self.generate_element_surface_points(element_idx, n_points_x, n_points_y)
        
        # Create mask for each velocity component
        masks = {}
        for component in ['x', 'y', 'z']:
            indices, weights = self.band_limited_interpolation_weights(
                grid, points, z0, kernel_radius, staggered_component=component, use_gpu=use_gpu
            )
            masks[f'v{component}'] = (indices, weights)
        
        return masks

    def create_all_element_masks(self, grid, n_points_x, n_points_y, z0=0.0,
                                 kernel_radius=3, staggered=False, use_gpu=False):
        """
        Create BLI masks for all elements.
        
        Args:
            grid: Grid object
            n_points_x: Number of sample points along width per element
            n_points_y: Number of sample points along height per element
            z0: Z-position of transducer surface
            kernel_radius: Sinc kernel radius
            staggered: If True, return staggered masks for velocity components
            use_gpu: Use GPU acceleration
            
        Returns:
            If staggered=False: list of (indices, weights) tuples
            If staggered=True: list of dicts with 'vx', 'vy', 'vz' keys
        """
        masks = []
        
        for elem_idx in range(self.n_elements):
            if staggered:
                mask = self.create_element_masks_staggered(
                    grid, elem_idx, n_points_x, n_points_y, z0, kernel_radius, use_gpu
                )
            else:
                mask = self.create_element_mask(
                    grid, elem_idx, n_points_x, n_points_y, z0, kernel_radius, None, use_gpu
                )
            masks.append(mask)
        
        return masks


# Test the implementation
if __name__ == '__main__':
    from grid import Grid
    
    print("Testing correct BLI implementation...")
    
    # Create test setup
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    tx = Transducer(n_elements=4, element_width=0.0003, element_height=0.002)
    
    # Test 1: Generate points
    points = tx.generate_element_surface_points(0, n_points_x=5, n_points_y=5)
    print(f"\nTest 1: Generated {len(points)} points")
    print(f"  Point range X: [{points[:, 0].min()*1e3:.3f}, {points[:, 0].max()*1e3:.3f}]mm")
    print(f"  Point range Y: [{points[:, 1].min()*1e3:.3f}, {points[:, 1].max()*1e3:.3f}]mm")
    
    # Test 2: Create BLI mask
    indices, weights = tx.create_element_mask(grid, 0, n_points_x=5, n_points_y=5)
    print(f"\nTest 2: BLI mask")
    print(f"  Sparse entries: {len(weights)}")
    print(f"  Weight sum: {weights.sum():.6f} (should be ~1.0)")
    print(f"  Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    # Test 3: Staggered masks
    staggered_masks = tx.create_element_masks_staggered(grid, 0, n_points_x=3, n_points_y=3)
    print(f"\nTest 3: Staggered masks")
    for comp, (idx, wgt) in staggered_masks.items():
        print(f"  {comp}: {len(wgt)} entries, sum={wgt.sum():.6f}")
    
    # Test 4: Memory comparison
    dense_memory = grid.nx * grid.ny * grid.nz * 4 / (1024**2)
    sparse_memory = len(weights) * (3 * 4 + 4) / (1024**2)
    print(f"\nTest 4: Memory usage")
    print(f"  Dense: {dense_memory:.3f} MB")
    print(f"  Sparse: {sparse_memory:.3f} MB")
    print(f"  Savings: {100*(1-sparse_memory/dense_memory):.1f}%")
    
    print("\n✓ All tests completed")
