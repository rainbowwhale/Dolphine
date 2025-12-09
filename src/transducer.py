import numpy as np


class Transducer:
    """Transducer element geometry and beamforming utilities.

    Default parameters set to typical medical linear probe values.
    
    Reference for band-limited interpolation:
    https://doi.org/10.1121/1.5116132
    """
    
    # Default sinc kernel radius for band-limited interpolation (in grid cells)
    DEFAULT_KERNEL_RADIUS = 3

    def __init__(self, n_elements=64, pitch=0.0003, element_width=0.00028, kerf=0.00002, 
                 center_freq=5e6, c=1540.0, element_height=0.010):
        self.n_elements = n_elements
        self.pitch = pitch
        self.element_width = element_width
        self.kerf = kerf
        self.center_freq = center_freq
        self.c = c
        self.element_height = element_height  # Height of rectangular element (elevation direction)
        # Generate element center positions along x-axis centered at zero
        x_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch
        self.element_positions = np.stack([x_positions, np.zeros_like(x_positions)], axis=1)  # (x, z=0)

    def delays_for_focus(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for each element to focus at `focus_point` (x,z) in meters.

        Returns delays in seconds (non-negative, relative to minimum).
        """
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

    def generate_element_surface_points(self, element_idx, n_points_x=None, n_points_y=None, 
                                       grid=None, grid_aligned=True,
                                       grid_center_only_x=False, grid_center_only_y=False,
                                       grid_center_only_z=False):
        """Generate a set of points on the surface of a rectangular element.
        
        Args:
            element_idx: Index of the element (0 to n_elements-1)
            n_points_x: Number of sample points along element width (lateral direction).
                       If None and grid is provided, calculated from grid spacing.
            n_points_y: Number of sample points along element height (elevation direction).
                       If None and grid is provided, calculated from grid spacing.
            grid: Grid object for automatic point spacing calculation
            grid_aligned: If True and grid provided, align points with grid spacing
            grid_center_only_x: If True, constrain x coordinates to grid centers only
            grid_center_only_y: If True, constrain y coordinates to grid centers only
            grid_center_only_z: If True, constrain z coordinates to grid centers only
            
        Returns:
            points: Array of shape (n_points, 3) with (x, y, z) coordinates in meters
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center position
        center_x = self.element_positions[element_idx, 0]
        center_z = self.element_positions[element_idx, 1]
        
        # Calculate number of points based on grid spacing if not provided
        if grid is not None and grid_aligned:
            # Use minimum grid spacing for point spacing (more conservative sampling)
            min_grid_spacing = self._get_min_grid_spacing(grid)
            
            if n_points_x is None:
                # Use minimum grid spacing to determine number of points
                n_points_x = max(3, int(np.ceil(self.element_width / min_grid_spacing)) + 1)
            if n_points_y is None:
                n_points_y = max(3, int(np.ceil(self.element_height / min_grid_spacing)) + 1)
            
            # Generate grid-aligned points using minimum spacing
            x_samples = np.arange(n_points_x) * min_grid_spacing
            x_samples = x_samples - x_samples.mean()  # Center around zero
            x_samples = np.clip(x_samples, -self.element_width/2, self.element_width/2)
            
            y_samples = np.arange(n_points_y) * min_grid_spacing
            y_samples = y_samples - y_samples.mean()  # Center around zero
            y_samples = np.clip(y_samples, -self.element_height/2, self.element_height/2)
        else:
            # Use uniform spacing (legacy behavior)
            if n_points_x is None:
                n_points_x = 5
            if n_points_y is None:
                n_points_y = 5
            x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
            y_samples = np.linspace(-self.element_height/2, self.element_height/2, n_points_y)
        
        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_samples, y_samples)
        x_coords = center_x + xx.flatten()
        y_coords = yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        # Apply grid-center-only constraints if requested
        if grid is not None:
            grid_center_x, grid_center_y, grid_center_z = self._compute_grid_centers(grid)
            
            if grid_center_only_x:
                x_coords = self._snap_to_grid_center(x_coords, grid.dx, grid_center_x)
            
            if grid_center_only_y:
                y_coords = self._snap_to_grid_center(y_coords, grid.dy, grid_center_y)
            
            if grid_center_only_z:
                z_coords = self._snap_to_grid_center(z_coords, grid.dz, grid_center_z)
        
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        return points

    def generate_all_elements_surface_points(self, n_points_x=None, n_points_y=None, 
                                            grid=None, grid_aligned=True,
                                            grid_center_only_x=False, grid_center_only_y=False,
                                            grid_center_only_z=False):
        """Generate surface points for all transducer elements.
        
        Args:
            n_points_x: Number of sample points along element width (lateral direction)
            n_points_y: Number of sample points along element height (elevation direction)
            grid: Grid object for automatic point spacing calculation
            grid_aligned: If True and grid provided, align points with grid spacing
            grid_center_only_x: If True, constrain x coordinates to grid centers only
            grid_center_only_y: If True, constrain y coordinates to grid centers only
            grid_center_only_z: If True, constrain z coordinates to grid centers only
            
        Returns:
            points_list: List of arrays, each containing points for one element
            element_indices: List of element indices corresponding to each point set
        """
        points_list = []
        element_indices = []
        
        for elem_idx in range(self.n_elements):
            points = self.generate_element_surface_points(
                elem_idx, n_points_x, n_points_y, grid, grid_aligned,
                grid_center_only_x, grid_center_only_y, grid_center_only_z
            )
            points_list.append(points)
            element_indices.append(elem_idx)
        
        return points_list, element_indices

    def calculate_bli_points_for_error(self, error_tolerance=0.01, grid=None):
        """Calculate the number of BLI sampling points needed for a given error tolerance.
        
        Can work with or without grid spacing information. When grid is provided,
        uses grid spacing for more accurate calculation. When grid is None, uses
        a conservative estimate based on element dimensions alone.
        
        Args:
            error_tolerance: Desired relative error tolerance (e.g., 0.01 for 1% error)
                           Smaller values require more sampling points
            grid: Grid object (optional). If None, calculates based on element size only.
            
        Returns:
            tuple: (n_points_x, n_points_y) recommended number of sampling points
        """
        # Error tolerance to oversampling factor mapping
        # Lower error requires higher oversampling
        if error_tolerance <= 0.001:  # 0.1% error
            factor = 4.0
        elif error_tolerance <= 0.01:  # 1% error
            factor = 3.0
        elif error_tolerance <= 0.05:  # 5% error
            factor = 2.0
        else:  # > 5% error
            factor = 1.5
        
        if grid is not None:
            # Calculate based on element size and minimum grid spacing
            # Use minimum grid spacing for conservative sampling
            min_grid_spacing = self._get_min_grid_spacing(grid)
            n_points_x = max(3, int(np.ceil(self.element_width / min_grid_spacing * factor)))
            n_points_y = max(3, int(np.ceil(self.element_height / min_grid_spacing * factor)))
        else:
            # Grid-independent calculation based on element dimensions
            # Use a heuristic: assume typical wavelength-based sampling
            # For ultrasound at typical frequencies (1-20 MHz) with c=1540 m/s,
            # wavelengths range from ~0.08mm to 1.5mm
            # Conservative estimate: 50µm (0.05mm) is ~1/2 of smallest typical wavelength (80µm at 20MHz)
            # This ensures adequate sampling even for high-frequency applications
            # May be overkill for low-frequency (1-5 MHz) but guarantees convergence
            assumed_min_spacing = 5e-5  # 50 micrometers
            
            n_points_x = max(3, int(np.ceil(self.element_width / assumed_min_spacing * factor)))
            n_points_y = max(3, int(np.ceil(self.element_height / assumed_min_spacing * factor)))
        
        return n_points_x, n_points_y

    def _get_min_grid_spacing(self, grid):
        """Get minimum grid spacing across all dimensions.
        
        Args:
            grid: Grid object
            
        Returns:
            float: Minimum of dx, dy, dz
        """
        return min(grid.dx, grid.dy, grid.dz)

    def _snap_to_grid_center(self, coords, grid_spacing, grid_center):
        """Snap coordinates to nearest grid center.
        
        Args:
            coords: Array of coordinates to snap
            grid_spacing: Grid spacing for this dimension
            grid_center: Grid center offset for this dimension
            
        Returns:
            Array of snapped coordinates
        """
        indices = np.round((coords + grid_center) / grid_spacing).astype(int)
        return indices * grid_spacing - grid_center

    def _compute_grid_centers(self, grid):
        """Compute grid center offsets for centered coordinate system.
        
        The Grid class uses world coordinates starting at (0,0,0), but transducer
        elements are positioned relative to a centered coordinate system. This method
        computes the offsets needed to convert between these systems.
        
        Args:
            grid: Grid object
            
        Returns:
            tuple: (grid_center_x, grid_center_y, grid_center_z) in meters
        """
        grid_center_x = (grid.nx - 1) * grid.dx / 2.0
        grid_center_y = (grid.ny - 1) * grid.dy / 2.0
        grid_center_z = (grid.nz - 1) * grid.dz / 2.0
        return grid_center_x, grid_center_y, grid_center_z

    def _world_to_centered_grid_index(self, x, y, z, grid):
        """Convert world coordinates to grid indices using centered coordinate system.
        
        Args:
            x, y, z: World coordinates in meters
            grid: Grid object
            
        Returns:
            tuple: (ix, iy, iz) grid indices
        """
        grid_center_x, grid_center_y, grid_center_z = self._compute_grid_centers(grid)
        ix = int(round((x + grid_center_x) / grid.dx))
        iy = int(round((y + grid_center_y) / grid.dy))
        iz = int(round((z + grid_center_z) / grid.dz))
        return ix, iy, iz

    def _centered_grid_index_to_world(self, ix, iy, iz, grid, offset_x=0.0, offset_y=0.0, offset_z=0.0):
        """Convert grid indices to world coordinates using centered coordinate system.
        
        Args:
            ix, iy, iz: Grid indices
            grid: Grid object
            offset_x, offset_y, offset_z: Additional offsets (e.g., for staggered grids)
            
        Returns:
            tuple: (x, y, z) world coordinates in meters
        """
        grid_center_x, grid_center_y, grid_center_z = self._compute_grid_centers(grid)
        x = ix * grid.dx - grid_center_x + offset_x
        y = iy * grid.dy - grid_center_y + offset_y
        z = iz * grid.dz - grid_center_z + offset_z
        return x, y, z
    def band_limited_interpolation_mask(self, grid, element_idx, n_points_x=None, n_points_y=None, 
                                       z0=0.0, staggered=False, kernel_radius=None, 
                                       error_tolerance=None, grid_aligned=True,
                                       grid_center_only_x=False, grid_center_only_y=False,
                                       grid_center_only_z=False):
        """Create a mask using band-limited interpolation for a single element.
        
        Based on band-limited interpolation method from https://doi.org/10.1121/1.5116132
        Uses sinc interpolation to distribute element surface points onto the grid.
        
        Optimized for large arrays using vectorized operations.
        
        Args:
            grid: Grid object defining the computational domain
            element_idx: Index of the element to create mask for
            n_points_x: Number of sample points along element width (auto-calculated if None)
            n_points_y: Number of sample points along element height (auto-calculated if None)
            z0: Z-position of the transducer surface in meters
            staggered: If True, use staggered grid offsets (half-grid spacing)
            kernel_radius: Sinc kernel radius in grid cells (default: 3)
                          Larger values increase accuracy but also computation cost
            error_tolerance: If provided, auto-calculate n_points for this error level
            grid_aligned: If True, align sampling points with minimum grid spacing
            grid_center_only_x: If True, constrain x coordinates to grid centers only
            grid_center_only_y: If True, constrain y coordinates to grid centers only
            grid_center_only_z: If True, constrain z coordinates to grid centers only
            
        Returns:
            mask: Array of shape matching grid dimensions with interpolated weights
        """
        if kernel_radius is None:
            kernel_radius = self.DEFAULT_KERNEL_RADIUS
        
        # Auto-calculate number of points based on error tolerance
        if error_tolerance is not None:
            n_points_x, n_points_y = self.calculate_bli_points_for_error(error_tolerance, grid)
            
        # Generate surface points for this element
        points = self.generate_element_surface_points(
            element_idx, n_points_x, n_points_y, grid, grid_aligned,
            grid_center_only_x, grid_center_only_y, grid_center_only_z
        )
        
        # Adjust points z-coordinate
        points[:, 2] = z0
        
        # Create mask array
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float32)
        
        # Grid offset for staggered grid (half-cell shift)
        offset_x = grid.dx / 2.0 if staggered else 0.0
        offset_y = grid.dy / 2.0 if staggered else 0.0
        offset_z = grid.dz / 2.0 if staggered else 0.0
        
        # Weight per point for normalization
        weight_per_point = 1.0 / len(points)
        
        # Vectorized approach for better performance with many points
        # Process all points at once for each grid cell
        grid_center_x, grid_center_y, grid_center_z = self._compute_grid_centers(grid)
        
        for point in points:
            px, py, pz = point
            
            # Convert world coordinates to grid indices (centered coordinate system)
            ix_center, iy_center, iz_center = self._world_to_centered_grid_index(px, py, pz, grid)
            
            # Define bounds for this point's kernel
            ix_min = max(0, ix_center - kernel_radius)
            ix_max = min(grid.nx, ix_center + kernel_radius + 1)
            iy_min = max(0, iy_center - kernel_radius)
            iy_max = min(grid.ny, iy_center + kernel_radius + 1)
            iz_min = max(0, iz_center - kernel_radius)
            iz_max = min(grid.nz, iz_center + kernel_radius + 1)
            
            # Create grid index arrays for vectorized computation
            ix_range = np.arange(ix_min, ix_max)
            iy_range = np.arange(iy_min, iy_max)
            iz_range = np.arange(iz_min, iz_max)
            
            # Compute grid positions (vectorized)
            gx = ix_range * grid.dx - grid_center_x + offset_x
            gy = iy_range * grid.dy - grid_center_y + offset_y
            gz = iz_range * grid.dz - grid_center_z + offset_z
            
            # Compute distances in grid units (vectorized)
            dist_x = (px - gx) / grid.dx
            dist_y = (py - gy) / grid.dy
            dist_z = (pz - gz) / grid.dz
            
            # Compute sinc values (vectorized)
            sinc_x = np.sinc(dist_x)
            sinc_y = np.sinc(dist_y)
            sinc_z = np.sinc(dist_z)
            
            # Create 3D weight grid using einsum (efficient and clear)
            # einsum('i,j,k->ijk') computes outer product: sinc_x[:, None, None] * sinc_y[None, :, None] * sinc_z[None, None, :]
            weights_3d = np.einsum('i,j,k->ijk', sinc_x, sinc_y, sinc_z) * weight_per_point
            
            # Add to mask
            mask[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += weights_3d
        
        return mask

    def create_element_masks(self, grid, z0=0.0, n_points_x=None, n_points_y=None, 
                            staggered=False, kernel_radius=None, error_tolerance=None,
                            grid_aligned=True, grid_center_only_x=False, 
                            grid_center_only_y=False, grid_center_only_z=False):
        """Create masks for all transducer elements.
        
        Note: For large grids with many elements, this method can consume significant memory
        (each mask is nx × ny × nz × 4 bytes). Consider:
        - Computing masks on-demand if memory is limited
        - Using sparse matrix representations for storage
        - Reducing grid size or number of active elements
        
        Performance optimizations for large arrays (e.g., 2D matrix probes with 10K elements):
        - Uses vectorized operations to avoid nested loops
        - Grid-aligned sampling using minimum grid spacing for better interpolation accuracy
        - Adaptive point calculation based on error tolerance
        
        Args:
            grid: Grid object defining the computational domain
            z0: Z-position of the transducer surface in meters
            n_points_x: Number of sample points along element width (auto if None)
            n_points_y: Number of sample points along element height (auto if None)
            staggered: If True, use staggered grid offsets
            kernel_radius: Sinc kernel radius in grid cells (default: 3)
            error_tolerance: If provided, auto-calculate n_points (e.g., 0.01 for 1% error)
            grid_aligned: If True, align sampling points with minimum grid spacing
            grid_center_only_x: If True, constrain x coordinates to grid centers only
            grid_center_only_y: If True, constrain y coordinates to grid centers only
            grid_center_only_z: If True, constrain z coordinates to grid centers only
            
        Returns:
            masks: List of mask arrays, one for each element
        """
        masks = []
        for elem_idx in range(self.n_elements):
            mask = self.band_limited_interpolation_mask(
                grid, elem_idx, n_points_x, n_points_y, z0, staggered, 
                kernel_radius, error_tolerance, grid_aligned,
                grid_center_only_x, grid_center_only_y, grid_center_only_z
            )
            masks.append(mask)
        return masks
