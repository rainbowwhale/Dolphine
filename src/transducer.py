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
        
        # Cache for BLI star (reusable across elements)
        self._bli_star_cache = {}
        
        # Cache for weight grid (reusable across elements to avoid reallocating)
        self._weight_grid_cache = {}

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
        
        # Get element dimensions (support for variable height)
        if hasattr(self, 'element_heights') and self.element_heights is not None:
            elem_height = self.element_heights[element_idx]
        else:
            elem_height = self.element_height
        
        # Generate uniform grid of points within element bounds
        # Use linspace with inclusive endpoints for uniform coverage
        x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
        y_samples = np.linspace(-elem_height/2, elem_height/2, n_points_y)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_samples, y_samples, indexing='xy')
        
        # Flatten and offset by element center
        x_coords = center_x + xx.flatten()
        y_coords = yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        return points

    def _get_bli_star(self, kernel_radius, tolerance):
        """
        Compute and cache BLI star points based on tolerance.
        
        BLI star = meshgrid of bli_range for x, y, z dimensions.
        Points are selected where bli_level = 1/(bli_x * bli_y * bli_z) <= tolerance.
        
        This is expensive for large kernels but reusable across all elements.
        
        Args:
            kernel_radius: Sinc kernel radius in grid cells
            tolerance: Weight threshold for selecting BLI star points
            
        Returns:
            bli_star_x, bli_star_y, bli_star_z: 1D arrays of selected BLI offsets
        """
        cache_key = (kernel_radius, tolerance)
        if cache_key in self._bli_star_cache:
            return self._bli_star_cache[cache_key]
        
        # Create BLI range: [-kernel_radius, ..., +kernel_radius]
        bli_range = np.arange(-kernel_radius, kernel_radius + 1)
        
        # Create meshgrid for BLI star
        bli_star_x, bli_star_y, bli_star_z = np.meshgrid(bli_range, bli_range, bli_range, indexing='ij')
        
        # Compute BLI level: 1/(x * y * z)
        # Avoid division by zero: set zero entries to large value
        with np.errstate(divide='ignore', invalid='ignore'):
            bli_level_x = np.where(bli_star_x != 0, 1.0 / np.abs(bli_star_x), 1.0)
            bli_level_y = np.where(bli_star_y != 0, 1.0 / np.abs(bli_star_y), 1.0)
            bli_level_z = np.where(bli_star_z != 0, 1.0 / np.abs(bli_star_z), 1.0)
            bli_level = bli_level_x * bli_level_y * bli_level_z
        
        # Select points where bli_level <= tolerance
        bli_selected = bli_level >= tolerance
        
        # Extract selected offsets
        bli_star_x_selected = bli_star_x[bli_selected]
        bli_star_y_selected = bli_star_y[bli_selected]
        bli_star_z_selected = bli_star_z[bli_selected]
        
        # Cache result
        result = (bli_star_x_selected, bli_star_y_selected, bli_star_z_selected)
        self._bli_star_cache[cache_key] = result
        
        return result

    def band_limited_interpolation_weights(self, grid, points, z0=0.0, kernel_radius=3,
                                          staggered_component=None, tolerance=1e-3, use_gpu=False):
        """
        Compute BLI weights for source points on grid using vectorized calculation.
        
        Implements correct BLI formula from reference:
        weight(grid_node) = sinc((px - gx)/dx) * sinc((py - gy)/dy) * sinc((pz - gz)/dz)
        
        Fully vectorized implementation with BLI star pre-selection:
        - px = (points[:,0] - (grid.x_vec[0] + offset)) / grid.dx
        - ix = floor(px)
        - rx = px - ix
        - BLI star is pre-computed and cached for reuse across elements
        - Uses 3D array approach for efficient index/weight computation
        
        Args:
            grid: Grid object with axis vectors (x_vec, y_vec, z_vec)
            points: Array of (x, y, z) source point coordinates
            z0: Z-position of transducer surface (m)
            kernel_radius: Sinc kernel radius in grid cells (e.g., 3 means -3 to +3)
            staggered_component: None for pressure (centered), 'x', 'y', or 'z' for velocity
            tolerance: Weight threshold for selecting BLI star points (default: 1e-3)
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
        
        # Get BLI star (pre-computed and cached)
        bli_star_x, bli_star_y, bli_star_z = self._get_bli_star(kernel_radius, tolerance)
        n_bli_points = len(bli_star_x)
        
        if use_gpu and HAS_CUPY:
            bli_star_x = cp.asarray(bli_star_x)
            bli_star_y = cp.asarray(bli_star_y)
            bli_star_z = cp.asarray(bli_star_z)
        
        # Grid offsets for staggered components
        offset_x, offset_y, offset_z = 0.0, 0.0, 0.0
        if staggered_component == 'x':
            offset_x = grid.dx / 2.0
        elif staggered_component == 'y':
            offset_y = grid.dy / 2.0
        elif staggered_component == 'z':
            offset_z = grid.dz / 2.0
        
        # Vectorized calculation per axis
        # Shape: (n_points,) - vectorized coordinate transformation
        # grid_origin represents the world coordinate of grid index 0
        # For staggered grids, add offset to grid_origin (not to point position)
        grid_origin_x = grid_x_vec[0] + offset_x
        grid_origin_y = grid_y_vec[0] + offset_y
        grid_origin_z = grid_z_vec[0] + offset_z
        
        n_points = points.shape[0]
        
        # Use actual point coordinates (points[:, 2]), not z0 scalar
        # Each point has its own z-coordinate
        px = (points[:, 0] - grid_origin_x) * inv_dx
        py = (points[:, 1] - grid_origin_y) * inv_dy
        pz = (points[:, 2] - grid_origin_z) * inv_dz  # Use points[:, 2], not z0!
        
        # ix = floor(px)
        ix = xp.floor(px).astype(int)  # Shape: (n_points,)
        iy = xp.floor(py).astype(int)
        iz = xp.floor(pz).astype(int)
        
        # rx = px - ix (fractional part)
        rx = px - ix  # Shape: (n_points,)
        ry = py - iy
        rz = pz - iz
        
        # Compute sinc for all BLI star points
        # Broadcasting: (n_points, 1) + (1, n_bli_points) = (n_points, n_bli_points)
        sinc_x_all = xp.sinc(rx[:, None] + bli_star_x[None, :])  # Shape: (n_points, n_bli_points)
        sinc_y_all = xp.sinc(ry[:, None] + bli_star_y[None, :])
        sinc_z_all = xp.sinc(rz[:, None] + bli_star_z[None, :])
        
        # Compute 3D weights: sinc_x * sinc_y * sinc_z for each BLI star point
        # Shape: (n_points, n_bli_points)
        sinc_all = sinc_x_all * sinc_y_all * sinc_z_all
        
        # Generate indices for all points and BLI star offsets
        # Shape: (n_points, n_bli_points)
        ix_all = ix[:, None] + bli_star_x[None, :]
        iy_all = iy[:, None] + bli_star_y[None, :]
        iz_all = iz[:, None] + bli_star_z[None, :]
        
        # Flatten all arrays for vectorized assignment
        ix_flat = ix_all.flatten()
        iy_flat = iy_all.flatten()
        iz_flat = iz_all.flatten()
        sinc_flat = sinc_all.flatten()
        
        # Filter valid indices (within grid bounds)
        valid_mask = (
            (ix_flat >= 0) & (ix_flat < grid.nx) &
            (iy_flat >= 0) & (iy_flat < grid.ny) &
            (iz_flat >= 0) & (iz_flat < grid.nz)
        )
        
        ix_valid = ix_flat[valid_mask]
        iy_valid = iy_flat[valid_mask]
        iz_valid = iz_flat[valid_mask]
        sinc_valid = sinc_flat[valid_mask]
        
        # Use localized 3D array approach for memory efficiency
        # Compute bounding box around element position (max range: element + ~11-12 grid cells)
        # This avoids allocating full grid for large grids (e.g., 1000x1000x1000)
        
        if len(ix_valid) == 0:
            # No valid points, return empty result
            return np.array([], dtype=np.int32).reshape(0, 3), np.array([], dtype=np.float32)
        
        # Determine bounding box
        ix_min = int(ix_valid.min() if use_gpu and HAS_CUPY else np.min(ix_valid))
        ix_max = int(ix_valid.max() if use_gpu and HAS_CUPY else np.max(ix_valid))
        iy_min = int(iy_valid.min() if use_gpu and HAS_CUPY else np.min(iy_valid))
        iy_max = int(iy_valid.max() if use_gpu and HAS_CUPY else np.max(iy_valid))
        iz_min = int(iz_valid.min() if use_gpu and HAS_CUPY else np.min(iz_valid))
        iz_max = int(iz_valid.max() if use_gpu and HAS_CUPY else np.max(iz_valid))
        
        # Local subgrid dimensions
        subgrid_nx = ix_max - ix_min + 1
        subgrid_ny = iy_max - iy_min + 1
        subgrid_nz = iz_max - iz_min + 1
        
        # Check cache for reusable weight_grid
        cache_key = (subgrid_nx, subgrid_ny, subgrid_nz, use_gpu)
        if cache_key in self._weight_grid_cache:
            weight_grid_local = self._weight_grid_cache[cache_key]
            # Zero out for reuse
            weight_grid_local[:] = 0
        else:
            # Create new local weight grid
            weight_grid_local = xp.zeros((subgrid_nx, subgrid_ny, subgrid_nz), dtype=xp.float32)
            self._weight_grid_cache[cache_key] = weight_grid_local
        
        # Map global indices to local subgrid coordinates
        ix_local = ix_valid - ix_min
        iy_local = iy_valid - iy_min
        iz_local = iz_valid - iz_min
        
        # Accumulate weights in local 3D grid
        # Use advanced indexing with add.at for accumulation (handles duplicates)
        if use_gpu and HAS_CUPY:
            # CuPy doesn't have add.at, use scatter_add
            linear_indices = ix_local + iy_local * subgrid_nx + iz_local * subgrid_nx * subgrid_ny
            cp.scatter_add(weight_grid_local.flatten(), linear_indices, sinc_valid)
        else:
            np.add.at(weight_grid_local, (ix_local, iy_local, iz_local), sinc_valid)
        
        # Find non-zero cells in local grid
        nonzero_mask = weight_grid_local != 0
        if use_gpu and HAS_CUPY:
            indices_i_local, indices_j_local, indices_k_local = cp.where(nonzero_mask)
            weights = weight_grid_local[nonzero_mask]
            
            # Map back to global indices
            indices_i = cp.asnumpy(indices_i_local) + ix_min
            indices_j = cp.asnumpy(indices_j_local) + iy_min
            indices_k = cp.asnumpy(indices_k_local) + iz_min
            weights = cp.asnumpy(weights)
        else:
            indices_i_local, indices_j_local, indices_k_local = np.where(nonzero_mask)
            weights = weight_grid_local[nonzero_mask]
            
            # Map back to global indices
            indices_i = indices_i_local + ix_min
            indices_j = indices_j_local + iy_min
            indices_k = indices_k_local + iz_min
        
        indices = np.stack([indices_i, indices_j, indices_k], axis=1).astype(np.int32)
        weights = weights.astype(np.float32)
        
        # Normalize so total weight sums to 1.0
        weight_sum = weights.sum()
        if weight_sum > 1e-10:
            weights = weights / weight_sum
        
        return indices, weights

    def create_element_mask(self, grid, element_idx, n_points_x, n_points_y, z0=0.0,
                           kernel_radius=3, tolerance=1e-3, staggered_component=None, use_gpu=False):
        """
        Create BLI mask for a single element.
        
        Args:
            grid: Grid object
            element_idx: Element index
            n_points_x: Number of sample points along width
            n_points_y: Number of sample points along height
            z0: Z-position of transducer surface
            kernel_radius: Sinc kernel radius (grid cells)
            tolerance: Weight threshold for BLI star point selection
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
            grid, points, z0, kernel_radius, staggered_component, tolerance, use_gpu
        )
        
        return indices, weights

    def create_element_masks_staggered(self, grid, element_idx, n_points_x, n_points_y,
                                      z0=0.0, kernel_radius=3, tolerance=1e-3, use_gpu=False):
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
            tolerance: Weight threshold for BLI star point selection
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
                grid, points, z0, kernel_radius, staggered_component=component, 
                tolerance=tolerance, use_gpu=use_gpu
            )
            masks[f'v{component}'] = (indices, weights)
        
        return masks

    def create_all_element_masks(self, grid, n_points_x, n_points_y, z0=0.0,
                                 kernel_radius=3, tolerance=1e-3, staggered=False, use_gpu=False):
        """
        Create BLI masks for all elements.
        
        Args:
            grid: Grid object
            n_points_x: Number of sample points along width per element
            n_points_y: Number of sample points along height per element
            z0: Z-position of transducer surface
            kernel_radius: Sinc kernel radius
            tolerance: Weight threshold for BLI star point selection
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
                    grid, elem_idx, n_points_x, n_points_y, z0, kernel_radius, tolerance, use_gpu
                )
            else:
                mask = self.create_element_mask(
                    grid, elem_idx, n_points_x, n_points_y, z0, kernel_radius, tolerance, None, use_gpu
                )
            masks.append(mask)
        
        return masks


class Transducer1p5D(Transducer):
    """1.5D Transducer with multiple rows (3-7 typical) and variable element heights.
    
    A 1.5D array has multiple rows of elements arranged in the elevation direction,
    allowing electronic elevation focusing. Rows can have different heights for
    optimized beam control.
    
    Element numbering: Elements are numbered sequentially across rows.
    For a 3-row, 32-element-per-row array: elements 0-31 are row 0, 32-63 are row 1, etc.
    """
    
    def __init__(self, n_elements_per_row=32, n_rows=None, pitch=0.0003, row_pitch=0.0004,
                 element_width=0.00028, row_heights=None, kerf=0.00002,
                 center_freq=5e6, c=1540.0):
        """
        Args:
            n_elements_per_row: Number of elements in each row (lateral direction)
            n_rows: Number of rows (elevation direction), typically 3-7.
                   If row_heights is an array, n_rows is derived from its length.
                   If row_heights is a scalar, n_rows must be provided.
            pitch: Center-to-center spacing between elements in a row (lateral, m)
            row_pitch: Center-to-center spacing between rows (elevation, m)
            element_width: Width of each element (lateral, m)
            row_heights: Height(s) for row(s) (elevation, m).
                        Can be:
                        - None: uses uniform height of 0.4mm for n_rows rows
                        - Scalar (single value): uses this height uniformly for n_rows rows
                        - Array: n_rows is derived from array length, each row gets its specified height
            kerf: Gap between elements (m)
            center_freq: Center frequency (Hz)
            c: Speed of sound (m/s)
        """
        # Determine number of rows and row heights
        if row_heights is None:
            # No heights specified, use default uniform heights
            if n_rows is None:
                n_rows = 5  # Default to 5 rows
            DEFAULT_ROW_HEIGHT = 0.0004  # 0.4mm in meters
            self.row_heights = np.full(n_rows, DEFAULT_ROW_HEIGHT)
            self.n_rows = n_rows
        else:
            # Heights are specified
            row_heights_array = np.atleast_1d(row_heights)
            
            if row_heights_array.size == 1:
                # Single height value provided
                if n_rows is None:
                    raise ValueError("n_rows must be specified when row_heights is a single value")
                # Use the single height for all rows
                self.row_heights = np.full(n_rows, float(row_heights_array[0]))
                self.n_rows = n_rows
            else:
                # Array of heights provided - derive n_rows from length
                self.row_heights = np.array(row_heights_array)
                self.n_rows = len(self.row_heights)
                # If n_rows was also provided, verify consistency
                if n_rows is not None and n_rows != self.n_rows:
                    raise ValueError(f"n_rows ({n_rows}) does not match length of row_heights array ({self.n_rows})")
        
        # Total number of elements
        total_elements = n_elements_per_row * self.n_rows
        
        # Store configuration
        self.n_elements_per_row = n_elements_per_row
        self.row_pitch = row_pitch
        
        # Initialize base class with total elements and average height
        avg_height = np.mean(self.row_heights)
        super().__init__(
            n_elements=total_elements,
            pitch=pitch,
            element_width=element_width,
            kerf=kerf,
            center_freq=center_freq,
            c=c,
            element_height=avg_height
        )
        
        # Generate 2D element positions (x, y coordinates, z=0)
        x_positions = (np.arange(n_elements_per_row) - (n_elements_per_row - 1) / 2.0) * pitch
        y_positions = (np.arange(self.n_rows) - (self.n_rows - 1) / 2.0) * row_pitch
        
        # Create grid of positions
        xx, yy = np.meshgrid(x_positions, y_positions, indexing='xy')
        
        # Flatten to create element_positions array (N_total, 2) for (x, z)
        # Store as (x, y) in 3D space, z=0 for transducer surface
        self.element_positions_2d = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Update element_positions for compatibility (keep x, set z=0)
        self.element_positions = np.stack([xx.flatten(), np.zeros(total_elements)], axis=1)
        
        # Store per-element heights (map row index to height)
        self.element_heights = np.repeat(self.row_heights, n_elements_per_row)
    
    def get_element_row_col(self, element_idx):
        """Get the row and column indices for a given element index.
        
        Args:
            element_idx: Linear element index (0 to n_elements-1)
            
        Returns:
            (row_idx, col_idx): Row and column indices
        """
        row_idx = element_idx // self.n_elements_per_row
        col_idx = element_idx % self.n_elements_per_row
        return row_idx, col_idx
    
    def delays_for_focus_3d(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for 3D focus point (x, y, z) in meters.
        
        Args:
            focus_point: Tuple (x, y, z) of focus point in meters
            speed_of_sound: Speed of sound (m/s), uses self.c if None
            
        Returns:
            delays: Array of delays for each element (seconds)
        """
        if speed_of_sound is None:
            c = self.c
        else:
            c = speed_of_sound
        
        # Element positions in 3D (x, y, z=0)
        dx = self.element_positions_2d[:, 0] - focus_point[0]
        dy = self.element_positions_2d[:, 1] - focus_point[1]
        dz = focus_point[2] - 0.0
        
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        delays = distances / c
        delays -= delays.min()
        return delays
    
    def map_to_grid(self, grid, z0=0.0):
        """Map element centers to grid indices (ix, iy, iz) using Grid object."""
        idx = []
        for i in range(self.n_elements):
            x = self.element_positions_2d[i, 0]
            y = self.element_positions_2d[i, 1]
            ix, iy, iz = grid.world_to_index(x, y, z0)
            idx.append((ix, iy, iz))
        return idx
    
    def generate_element_surface_points(self, element_idx, n_points_x, n_points_y):
        """Generate surface points for 1.5D element with variable height.
        
        Overrides parent method to use element-specific height from row.
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center position in 3D
        center_x = self.element_positions_2d[element_idx, 0]
        center_y = self.element_positions_2d[element_idx, 1]
        center_z = 0.0
        
        # Get element-specific height
        elem_height = self.element_heights[element_idx]
        
        # Generate uniform grid of points within element bounds
        x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
        y_samples = np.linspace(-elem_height/2, elem_height/2, n_points_y)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_samples, y_samples, indexing='xy')
        
        # Flatten and offset by element center
        x_coords = center_x + xx.flatten()
        y_coords = center_y + yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        return points


class MatrixTransducer(Transducer):
    """2D Matrix Transducer with uniform rectangular grid of elements.
    
    A 2D matrix array has tens to hundreds of rows and columns of elements,
    allowing full 3D electronic beam steering and focusing. All elements have
    the same size.
    
    Element numbering: Elements are numbered row-major order.
    For a 16x16 array: elements 0-15 are row 0, 16-31 are row 1, etc.
    """
    
    def __init__(self, n_elements_x=16, n_elements_y=16, pitch=0.0003,
                 element_width=0.00028, element_height=0.00028, kerf=0.00002,
                 center_freq=5e6, c=1540.0):
        """
        Args:
            n_elements_x: Number of elements in x direction (lateral)
            n_elements_y: Number of elements in y direction (elevation)
            pitch: Center-to-center spacing between elements (m), same in both directions
            element_width: Width of each element (x direction, m)
            element_height: Height of each element (y direction, m)
            kerf: Gap between elements (m)
            center_freq: Center frequency (Hz)
            c: Speed of sound (m/s)
        """
        # Total number of elements
        total_elements = n_elements_x * n_elements_y
        
        # Store configuration
        self.n_elements_x = n_elements_x
        self.n_elements_y = n_elements_y
        
        # Initialize base class
        super().__init__(
            n_elements=total_elements,
            pitch=pitch,
            element_width=element_width,
            kerf=kerf,
            center_freq=center_freq,
            c=c,
            element_height=element_height
        )
        
        # Generate 2D element positions (x, y coordinates, z=0)
        x_positions = (np.arange(n_elements_x) - (n_elements_x - 1) / 2.0) * pitch
        y_positions = (np.arange(n_elements_y) - (n_elements_y - 1) / 2.0) * pitch
        
        # Create grid of positions
        xx, yy = np.meshgrid(x_positions, y_positions, indexing='xy')
        
        # Flatten to create element_positions array
        self.element_positions_2d = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Update element_positions for compatibility
        self.element_positions = np.stack([xx.flatten(), np.zeros(total_elements)], axis=1)
        
        # All elements have uniform height
        self.element_heights = None  # Uniform, use self.element_height
    
    def get_element_row_col(self, element_idx):
        """Get the row and column indices for a given element index.
        
        Args:
            element_idx: Linear element index (0 to n_elements-1)
            
        Returns:
            (row_idx, col_idx): Row and column indices
        """
        row_idx = element_idx // self.n_elements_x
        col_idx = element_idx % self.n_elements_x
        return row_idx, col_idx
    
    def delays_for_focus_3d(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for 3D focus point (x, y, z) in meters.
        
        Args:
            focus_point: Tuple (x, y, z) of focus point in meters
            speed_of_sound: Speed of sound (m/s), uses self.c if None
            
        Returns:
            delays: Array of delays for each element (seconds)
        """
        if speed_of_sound is None:
            c = self.c
        else:
            c = speed_of_sound
        
        # Element positions in 3D (x, y, z=0)
        dx = self.element_positions_2d[:, 0] - focus_point[0]
        dy = self.element_positions_2d[:, 1] - focus_point[1]
        dz = focus_point[2] - 0.0
        
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        delays = distances / c
        delays -= delays.min()
        return delays
    
    def delays_for_steering_3d(self, steering_angles, speed_of_sound=None):
        """Compute transmission delays for 3D beam steering.
        
        Args:
            steering_angles: Tuple (theta_x, theta_y) in radians
                           theta_x: steering angle in x-z plane
                           theta_y: steering angle in y-z plane
            speed_of_sound: Speed of sound (m/s), uses self.c if None
            
        Returns:
            delays: Array of delays for each element (seconds)
        """
        if speed_of_sound is None:
            c = self.c
        else:
            c = speed_of_sound
        
        theta_x, theta_y = steering_angles
        
        # Compute delays based on plane wave steering
        # delay = (x * sin(theta_x) + y * sin(theta_y)) / c
        delays = (
            self.element_positions_2d[:, 0] * np.sin(theta_x) +
            self.element_positions_2d[:, 1] * np.sin(theta_y)
        ) / c
        
        # Normalize to positive delays
        delays -= delays.min()
        return delays
    
    def map_to_grid(self, grid, z0=0.0):
        """Map element centers to grid indices (ix, iy, iz) using Grid object."""
        idx = []
        for i in range(self.n_elements):
            x = self.element_positions_2d[i, 0]
            y = self.element_positions_2d[i, 1]
            ix, iy, iz = grid.world_to_index(x, y, z0)
            idx.append((ix, iy, iz))
        return idx
    
    def generate_element_surface_points(self, element_idx, n_points_x, n_points_y):
        """Generate surface points for matrix element.
        
        Overrides parent method to use 2D positions.
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center position in 3D
        center_x = self.element_positions_2d[element_idx, 0]
        center_y = self.element_positions_2d[element_idx, 1]
        center_z = 0.0
        
        # Generate uniform grid of points within element bounds
        x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
        y_samples = np.linspace(-self.element_height/2, self.element_height/2, n_points_y)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_samples, y_samples, indexing='xy')
        
        # Flatten and offset by element center
        x_coords = center_x + xx.flatten()
        y_coords = center_y + yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        return points


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
