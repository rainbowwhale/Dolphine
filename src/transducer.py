"""
Clean BLI Implementation - Unified Transducer Class

Based on feedback and reference DOI: 10.1121/1.5116132

Key corrections:
1. BLI formula: sinc((point_pos - grid_pos) / grid_spacing) per axis, multiply together
2. Element size only determines number of sample points, NOT interpolation calculation
3. Returns sparse format (indices, weights) for direct use in source injection
4. Supports staggered grids: 3 separate velocity component masks
5. GPU acceleration with CuPy

Unified design:
- Single Transducer class supports all array types (1D, 1.5D, 2D matrix, curved/concave)
- 3D element positions (x, y, z) for concave transducer support
- Configurable element layout via element_positions parameter
"""
import numpy as np
import warnings

try:
    from scipy.signal import windows as signal_windows
    HAS_SCIPY_WINDOWS = True
except ImportError:
    HAS_SCIPY_WINDOWS = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


class Transducer:
    """Unified transducer element geometry and beamforming utilities.
    
    Supports all transducer array types:
    - Linear (1D): Single row of elements
    - 1.5D: Multiple rows with variable heights
    - 2D Matrix: Large 2D grid for volumetric imaging
    - Curved/Concave: Elements with non-zero z-coordinates
    
    Includes band-limited interpolation (BLI) for distributed source injection.
    Reference: https://doi.org/10.1121/1.5116132
    """
    # Default configuration constants
    DEFAULT_ROW_HEIGHT = 0.0004  # Default row height in meters (0.4mm) for 1.5D arrays
    
    def __init__(self, n_elements=64, pitch=0.0003, element_width=0.00028, kerf=0.00002,
                 center_freq=5e6, c=1540.0, element_height=0.010,
                 n_elements_x=None, n_elements_y=None, row_pitch=None,
                 element_positions=None, element_heights=None,
                 # Legacy 1.5D parameters for backward compatibility
                 n_elements_per_row=None, n_rows=None, row_heights=None):
        """
        Args:
            n_elements: Number of transducer elements (overridden if element_positions provided)
            pitch: Center-to-center spacing between elements (m)
            element_width: Width of each element (lateral, m)
            kerf: Gap between elements (m)
            center_freq: Center frequency (Hz)
            c: Speed of sound (m/s)
            element_height: Height of element (elevation, m) - default for all elements
            n_elements_x: Number of elements in x direction (for 2D arrays)
            n_elements_y: Number of elements in y direction (for 2D arrays)
            row_pitch: Center-to-center spacing between rows (elevation, m) - for 1.5D/2D arrays
            element_positions: Custom 3D positions array (N, 3) for (x, y, z) coordinates.
                              If provided, overrides automatic position generation.
                              Enables curved/concave transducer geometries.
            element_heights: Per-element heights array (N,) for variable height elements.
                            If None, uses uniform element_height for all elements.
            n_elements_per_row: Legacy alias for n_elements_x (1.5D compatibility)
            n_rows: Legacy alias for n_elements_y (1.5D compatibility)
            row_heights: Per-row heights array for 1.5D transducers. If provided,
                        element_heights is generated automatically.
        """
        self.pitch = pitch
        self.element_width = element_width
        self.kerf = kerf
        self.center_freq = center_freq
        self.c = c
        self.element_height = element_height
        
        # Handle legacy 1.5D parameters
        if n_elements_per_row is not None:
            n_elements_x = n_elements_per_row
            self.n_elements_per_row = n_elements_per_row  # Store for backward compatibility
        
        # Handle row_heights and n_rows for 1.5D transducers
        if row_heights is not None:
            row_heights = np.atleast_1d(row_heights)
            if row_heights.size == 1:
                # Single height value - need n_rows
                if n_rows is None:
                    n_rows = 5  # Default
                self.row_heights = np.full(n_rows, float(row_heights[0]))
            else:
                # Array of heights - derive n_rows
                self.row_heights = np.array(row_heights)
                if n_rows is not None and n_rows != len(self.row_heights):
                    raise ValueError(f"n_rows ({n_rows}) doesn't match row_heights length ({len(self.row_heights)})")
                n_rows = len(self.row_heights)
            n_elements_y = n_rows
            self.n_rows = n_rows
            # Generate per-element heights from row_heights.
            # Note: element_heights here represents heights derived from row_heights
            # (all elements in a row get the same height), not custom per-element heights.
            if n_elements_x is not None:
                element_heights = np.repeat(self.row_heights, n_elements_x)
        elif n_rows is not None:
            n_elements_y = n_rows
            self.n_rows = n_rows
            # Default row heights
            self.row_heights = np.full(n_rows, self.DEFAULT_ROW_HEIGHT)
            if n_elements_x is not None:
                element_heights = np.repeat(self.row_heights, n_elements_x)
        
        # Set row_pitch default
        if row_pitch is None:
            row_pitch = 0.0004 if (n_rows is not None or n_elements_per_row is not None) else pitch
        self.row_pitch = row_pitch
        
        # Handle custom element positions (for curved/concave transducers)
        if element_positions is not None:
            element_positions = np.asarray(element_positions)
            if element_positions.ndim != 2 or element_positions.shape[1] != 3:
                raise ValueError("element_positions must be shape (N, 3) for (x, y, z) coordinates")
            self.element_positions = element_positions
            self.n_elements = element_positions.shape[0]
        else:
            # Generate positions based on array configuration
            if n_elements_x is not None and n_elements_y is not None:
                # 2D array (matrix or 1.5D style)
                self.n_elements_x = n_elements_x
                self.n_elements_y = n_elements_y
                self.n_elements = n_elements_x * n_elements_y
                
                # Generate 2D element positions
                x_positions = (np.arange(n_elements_x) - (n_elements_x - 1) / 2.0) * pitch
                y_positions = (np.arange(n_elements_y) - (n_elements_y - 1) / 2.0) * self.row_pitch
                
                xx, yy = np.meshgrid(x_positions, y_positions, indexing='xy')
                
                # 3D positions with z=0 (flat transducer surface)
                self.element_positions = np.stack([
                    xx.flatten(),
                    yy.flatten(),
                    np.zeros(self.n_elements)
                ], axis=1)
            else:
                # 1D linear array
                self.n_elements = n_elements
                self.n_elements_x = n_elements
                self.n_elements_y = 1
                
                x_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch
                
                # 3D positions with y=0 and z=0
                self.element_positions = np.stack([
                    x_positions,
                    np.zeros_like(x_positions),
                    np.zeros_like(x_positions)
                ], axis=1)
        
        # Set grid layout if not already set
        if not hasattr(self, 'n_elements_x'):
            self.n_elements_x = self.n_elements
        if not hasattr(self, 'n_elements_y'):
            self.n_elements_y = 1
        
        # Store n_elements_per_row if not already set (backward compatibility)
        if not hasattr(self, 'n_elements_per_row'):
            self.n_elements_per_row = self.n_elements_x
        if not hasattr(self, 'n_rows'):
            self.n_rows = self.n_elements_y
        if not hasattr(self, 'row_heights'):
            self.row_heights = np.full(self.n_elements_y, self.element_height)
        
        # Per-element heights (None means uniform element_height for all)
        if element_heights is not None:
            element_heights = np.asarray(element_heights)
            if len(element_heights) != self.n_elements:
                raise ValueError(f"element_heights length ({len(element_heights)}) must match n_elements ({self.n_elements})")
            self.element_heights = element_heights
        else:
            self.element_heights = None
        
        # Cache for BLI star (reusable across elements)
        self._bli_star_cache = {}
        
        # Cache for weight grid (reusable across elements to avoid reallocating)
        self._weight_grid_cache = {}

    def get_element_row_col(self, element_idx):
        """Get the row and column indices for a given element index.
        
        For 1D arrays, row is always 0 and col equals the element index.
        For 2D arrays, elements are numbered in row-major order.
        
        Args:
            element_idx: Linear element index (0 to n_elements-1)
            
        Returns:
            (row_idx, col_idx): Row and column indices
        """
        row_idx = element_idx // self.n_elements_x
        col_idx = element_idx % self.n_elements_x
        return row_idx, col_idx

    def delays_for_focus(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for each element to focus at `focus_point`.
        
        Accepts either 2D (x, z) or 3D (x, y, z) focus point.
        
        Args:
            focus_point: Tuple (x, z) or (x, y, z) of focus point in meters
            speed_of_sound: Speed of sound (m/s), uses self.c if None
            
        Returns:
            delays: Array of delays for each element (seconds)
        """
        if speed_of_sound is None:
            c = self.c
        else:
            c = speed_of_sound
        
        # Handle both 2D and 3D focus points
        if len(focus_point) == 2:
            # 2D focus (x, z) - assume y=0
            focus_x, focus_z = focus_point
            focus_y = 0.0
        else:
            focus_x, focus_y, focus_z = focus_point
        
        # Compute distances using 3D element positions
        dx = self.element_positions[:, 0] - focus_x
        dy = self.element_positions[:, 1] - focus_y
        dz = self.element_positions[:, 2] - focus_z
        
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        delays = distances / c
        delays -= delays.min()
        return delays

    def delays_for_focus_3d(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for 3D focus point (x, y, z) in meters.
        
        .. deprecated::
            This method is deprecated and will be removed in a future version.
            Use delays_for_focus() instead, which accepts both 2D (x, z) and 3D (x, y, z) focus points.
        
        Args:
            focus_point: Tuple (x, y, z) of focus point in meters
            speed_of_sound: Speed of sound (m/s), uses self.c if None
            
        Returns:
            delays: Array of delays for each element (seconds)
        """
        warnings.warn(
            "delays_for_focus_3d() is deprecated. Use delays_for_focus() instead, "
            "which accepts both 2D and 3D focus points.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.delays_for_focus(focus_point, speed_of_sound)

    def delays_for_steering_3d(self, steering_angles, speed_of_sound=None):
        """Compute transmission delays for 3D beam steering.
        
        Works for all transducer types. For curved transducers, the steering
        is computed relative to each element's local position.
        
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
        
        # Compute delays based on plane wave steering using x and y positions
        # For curved transducers, this uses the lateral positions
        delays = (
            self.element_positions[:, 0] * np.sin(theta_x) +
            self.element_positions[:, 1] * np.sin(theta_y)
        ) / c
        
        # Normalize to positive delays
        delays -= delays.min()
        return delays

    def apodization_hanning(self):
        """Return Hanning apodization weights across elements."""
        # Prefer SciPy's recommended Hann window implementation when available,
        # but fall back to NumPy's deprecated np.hanning for backward compatibility.
        if HAS_SCIPY_WINDOWS:
            return signal_windows.hann(self.n_elements)
        else:
            return np.hanning(self.n_elements)

    def map_to_grid(self, grid, z0=None):
        """Map element centers to grid indices (ix, iy, iz) using Grid object.
        
        For curved transducers, uses the actual z-coordinate of each element.
        For flat transducers, uses z0 if provided, otherwise element z-coordinate.
        
        Args:
            grid: Grid object with world_to_index method
            z0: Optional z-position override. If None, uses element z-coordinates.
            
        Returns:
            List of (ix, iy, iz) tuples for each element
        """
        idx = []
        for i in range(self.n_elements):
            x = self.element_positions[i, 0]
            y = self.element_positions[i, 1]
            z = z0 if z0 is not None else self.element_positions[i, 2]
            ix, iy, iz = grid.world_to_index(x, y, z)
            idx.append((ix, iy, iz))
        return idx

    def generate_element_surface_points(self, element_idx, n_points_x, n_points_y):
        """
        Generate uniformly distributed points on rectangular element surface.
        
        Points are distributed on a regular orthogonal grid within element boundaries.
        For curved transducers, points are generated relative to the element's
        3D position.
        
        Args:
            element_idx: Index of the element (0 to n_elements-1)
            n_points_x: Number of sample points along element width (lateral)
            n_points_y: Number of sample points along element height (elevation)
            
        Returns:
            points: Array of shape (n_points_x * n_points_y, 3) with (x, y, z) coordinates
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center position (3D)
        center_x = self.element_positions[element_idx, 0]
        center_y = self.element_positions[element_idx, 1]
        center_z = self.element_positions[element_idx, 2]
        
        # Get element dimensions (support for variable height)
        if self.element_heights is not None:
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
        y_coords = center_y + yy.flatten()
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

    def band_limited_interpolation_weights(self, grid, points, z0=None, kernel_radius=3,
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
            z0: Deprecated - kept for backward compatibility. 
                Uses points[:, 2] for z-coordinates.
            kernel_radius: Sinc kernel radius in grid cells (e.g., 3 means -3 to +3)
            staggered_component: None for pressure (centered), 'x', 'y', or 'z' for velocity
            tolerance: Weight threshold for selecting BLI star points (default: 1e-3)
            use_gpu: Use CuPy for GPU acceleration if available
            
        Returns:
            indices: (N, 3) array of grid indices (i, j, k)
            weights: (N,) array of corresponding weights
        """
        # Issue deprecation warning for z0 parameter
        if z0 is not None:
            warnings.warn(
                "The 'z0' parameter is deprecated and will be removed in a future version. "
                "Pass z-coordinates via the 'points' array instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
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
        
        # Pre-compute reciprocals for better performance
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
        grid_origin_x = grid_x_vec[0] + offset_x
        grid_origin_y = grid_y_vec[0] + offset_y
        grid_origin_z = grid_z_vec[0] + offset_z
        
        n_points = points.shape[0]
        
        # Use actual point coordinates (points[:, 2]), not z0 scalar
        px = (points[:, 0] - grid_origin_x) * inv_dx
        py = (points[:, 1] - grid_origin_y) * inv_dy
        pz = (points[:, 2] - grid_origin_z) * inv_dz
        
        # ix = floor(px)
        ix = xp.floor(px).astype(int)
        iy = xp.floor(py).astype(int)
        iz = xp.floor(pz).astype(int)
        
        # rx = px - ix (fractional part)
        rx = px - ix
        ry = py - iy
        rz = pz - iz
        
        # Compute sinc for all BLI star points
        sinc_x_all = xp.sinc(rx[:, None] + bli_star_x[None, :])
        sinc_y_all = xp.sinc(ry[:, None] + bli_star_y[None, :])
        sinc_z_all = xp.sinc(rz[:, None] + bli_star_z[None, :])
        
        # Compute 3D weights
        sinc_all = sinc_x_all * sinc_y_all * sinc_z_all
        
        # Generate indices for all points and BLI star offsets
        ix_all = ix[:, None] + bli_star_x[None, :]
        iy_all = iy[:, None] + bli_star_y[None, :]
        iz_all = iz[:, None] + bli_star_z[None, :]
        
        # Flatten all arrays
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
        
        if len(ix_valid) == 0:
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
            weight_grid_local[:] = 0
        else:
            weight_grid_local = xp.zeros((subgrid_nx, subgrid_ny, subgrid_nz), dtype=xp.float32)
            self._weight_grid_cache[cache_key] = weight_grid_local
        
        # Map global indices to local subgrid coordinates
        ix_local = ix_valid - ix_min
        iy_local = iy_valid - iy_min
        iz_local = iz_valid - iz_min
        
        # Accumulate weights in local 3D grid
        if use_gpu and HAS_CUPY:
            linear_indices = ix_local + iy_local * subgrid_nx + iz_local * subgrid_nx * subgrid_ny
            cp.scatter_add(weight_grid_local.flatten(), linear_indices, sinc_valid)
        else:
            np.add.at(weight_grid_local, (ix_local, iy_local, iz_local), sinc_valid)
        
        # Find non-zero cells in local grid
        nonzero_mask = weight_grid_local != 0
        if use_gpu and HAS_CUPY:
            indices_i_local, indices_j_local, indices_k_local = cp.where(nonzero_mask)
            weights = weight_grid_local[nonzero_mask]
            
            indices_i = cp.asnumpy(indices_i_local) + ix_min
            indices_j = cp.asnumpy(indices_j_local) + iy_min
            indices_k = cp.asnumpy(indices_k_local) + iz_min
            weights = cp.asnumpy(weights)
        else:
            indices_i_local, indices_j_local, indices_k_local = np.where(nonzero_mask)
            weights = weight_grid_local[nonzero_mask]
            
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

    def create_element_mask(self, grid, element_idx, n_points_x, n_points_y, z0=None,
                           kernel_radius=3, tolerance=1e-3, staggered_component=None, use_gpu=False):
        """
        Create BLI mask for a single element.
        
        Args:
            grid: Grid object
            element_idx: Element index
            n_points_x: Number of sample points along width
            n_points_y: Number of sample points along height
            z0: Deprecated - kept for backward compatibility. 
                Uses element's 3D position from element_positions.
            kernel_radius: Sinc kernel radius (grid cells)
            tolerance: Weight threshold for BLI star point selection
            staggered_component: None, 'x', 'y', or 'z' for staggered grids
            use_gpu: Use GPU acceleration
            
        Returns:
            indices: (N, 3) array of grid indices
            weights: (N,) array of weights
        """
        # Issue deprecation warning for z0 parameter
        if z0 is not None:
            warnings.warn(
                "The 'z0' parameter is deprecated and will be removed in a future version. "
                "Use the element's z-coordinate in element_positions instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Generate surface points (uses element's 3D position)
        points = self.generate_element_surface_points(element_idx, n_points_x, n_points_y)
        
        # Compute BLI weights
        indices, weights = self.band_limited_interpolation_weights(
            grid, points, z0, kernel_radius, staggered_component, tolerance, use_gpu
        )
        
        return indices, weights

    def create_element_masks_staggered(self, grid, element_idx, n_points_x, n_points_y,
                                      z0=None, kernel_radius=3, tolerance=1e-3, use_gpu=False):
        """
        Create staggered grid masks for velocity components.
        
        Returns 3 masks for velocity components Vx, Vy, Vz where each component
        is staggered along its corresponding axis.
        
        Args:
            grid: Grid object
            element_idx: Element index
            n_points_x: Number of sample points along width
            n_points_y: Number of sample points along height
            z0: Deprecated - kept for backward compatibility. 
                Uses element's 3D position from element_positions.
            kernel_radius: Sinc kernel radius
            tolerance: Weight threshold for BLI star point selection
            use_gpu: Use GPU acceleration
            
        Returns:
            dict: {'vx': (indices, weights), 'vy': (indices, weights), 'vz': (indices, weights)}
        """
        # Issue deprecation warning for z0 parameter
        if z0 is not None:
            warnings.warn(
                "The 'z0' parameter is deprecated and will be removed in a future version. "
                "Use the element's z-coordinate in element_positions instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
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

    def create_all_element_masks(self, grid, n_points_x, n_points_y, z0=None,
                                 kernel_radius=3, tolerance=1e-3, staggered=False, use_gpu=False):
        """
        Create BLI masks for all elements.
        
        Args:
            grid: Grid object
            n_points_x: Number of sample points along width per element
            n_points_y: Number of sample points along height per element
            z0: Deprecated - kept for backward compatibility. 
                Uses element's 3D position from element_positions.
            kernel_radius: Sinc kernel radius
            tolerance: Weight threshold for BLI star point selection
            staggered: If True, return staggered masks for velocity components
            use_gpu: Use GPU acceleration
            
        Returns:
            If staggered=False: list of (indices, weights) tuples
            If staggered=True: list of dicts with 'vx', 'vy', 'vz' keys
        """
        # Issue deprecation warning for z0 parameter
        if z0 is not None:
            warnings.warn(
                "The 'z0' parameter is deprecated and will be removed in a future version. "
                "Use the element's z-coordinate in element_positions instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
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


# Backward compatibility aliases
Transducer1p5D = Transducer
MatrixTransducer = Transducer


# Test the implementation
if __name__ == '__main__':
    from grid import Grid
    
    print("Testing unified Transducer class...")
    
    # Create test setup
    grid = Grid(nx=64, ny=64, nz=64, dx=1e-4)
    
    # Test 1: Basic 1D linear transducer
    print("\n=== Test 1: 1D Linear Transducer ===")
    tx1d = Transducer(n_elements=4, element_width=0.0003, element_height=0.002)
    print(f"Created 1D transducer with {tx1d.n_elements} elements")
    print(f"Element positions shape: {tx1d.element_positions.shape}")
    print(f"First element position: {tx1d.element_positions[0]}")
    
    # Test 2: 2D matrix transducer
    print("\n=== Test 2: 2D Matrix Transducer ===")
    tx2d = Transducer(n_elements_x=8, n_elements_y=8, pitch=0.0003)
    print(f"Created 2D matrix with {tx2d.n_elements} elements ({tx2d.n_elements_x}x{tx2d.n_elements_y})")
    print(f"Element positions shape: {tx2d.element_positions.shape}")
    
    # Test 3: Custom curved transducer
    print("\n=== Test 3: Curved/Concave Transducer ===")
    # Create simple curved array (arc)
    n_elem = 16
    angles = np.linspace(-np.pi/4, np.pi/4, n_elem)
    radius = 0.05  # 5cm radius
    curved_positions = np.stack([
        radius * np.sin(angles),  # x
        np.zeros(n_elem),          # y
        radius * (1 - np.cos(angles))  # z (concave towards positive z)
    ], axis=1)
    tx_curved = Transducer(element_positions=curved_positions, element_width=0.0003)
    print(f"Created curved transducer with {tx_curved.n_elements} elements")
    print(f"Z-range: [{tx_curved.element_positions[:, 2].min():.4f}, {tx_curved.element_positions[:, 2].max():.4f}]m")
    
    # Test 4: Generate points and BLI mask
    print("\n=== Test 4: BLI Mask Generation ===")
    points = tx1d.generate_element_surface_points(0, n_points_x=5, n_points_y=5)
    print(f"Generated {len(points)} points")
    print(f"Point range X: [{points[:, 0].min()*1e3:.3f}, {points[:, 0].max()*1e3:.3f}]mm")
    print(f"Point range Z: [{points[:, 2].min()*1e3:.3f}, {points[:, 2].max()*1e3:.3f}]mm")
    
    indices, weights = tx1d.create_element_mask(grid, 0, n_points_x=5, n_points_y=5)
    print(f"Sparse entries: {len(weights)}")
    print(f"Weight sum: {weights.sum():.6f} (should be ~1.0)")
    
    # Test 5: 3D focusing
    print("\n=== Test 5: 3D Focusing ===")
    delays = tx2d.delays_for_focus((0.0, 0.0, 0.03))
    print(f"Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}]µs")
    
    print("\n✓ All tests completed")
