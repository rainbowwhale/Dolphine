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

    def generate_element_surface_points(self, element_idx, n_points_x=5, n_points_y=5):
        """Generate a set of points on the surface of a rectangular element.
        
        Args:
            element_idx: Index of the element (0 to n_elements-1)
            n_points_x: Number of sample points along element width (lateral direction)
            n_points_y: Number of sample points along element height (elevation direction)
            
        Returns:
            points: Array of shape (n_points, 3) with (x, y, z) coordinates in meters
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center position
        center_x = self.element_positions[element_idx, 0]
        center_z = self.element_positions[element_idx, 1]
        
        # Generate uniformly distributed points on element surface
        x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
        y_samples = np.linspace(-self.element_height/2, self.element_height/2, n_points_y)
        
        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_samples, y_samples)
        x_coords = center_x + xx.flatten()
        y_coords = yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        return points

    def generate_all_elements_surface_points(self, n_points_x=5, n_points_y=5):
        """Generate surface points for all transducer elements.
        
        Args:
            n_points_x: Number of sample points along element width (lateral direction)
            n_points_y: Number of sample points along element height (elevation direction)
            
        Returns:
            points_list: List of arrays, each containing points for one element
            element_indices: List of element indices corresponding to each point set
        """
        points_list = []
        element_indices = []
        
        for elem_idx in range(self.n_elements):
            points = self.generate_element_surface_points(elem_idx, n_points_x, n_points_y)
            points_list.append(points)
            element_indices.append(elem_idx)
        
        return points_list, element_indices

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
    def band_limited_interpolation_mask(self, grid, element_idx, n_points_x=5, n_points_y=5, 
                                       z0=0.0, staggered=False, kernel_radius=None):
        """Create a mask using band-limited interpolation for a single element.
        
        Based on band-limited interpolation method from https://doi.org/10.1121/1.5116132
        Uses sinc interpolation to distribute element surface points onto the grid.
        
        Args:
            grid: Grid object defining the computational domain
            element_idx: Index of the element to create mask for
            n_points_x: Number of sample points along element width
            n_points_y: Number of sample points along element height
            z0: Z-position of the transducer surface in meters
            staggered: If True, use staggered grid offsets (half-grid spacing)
            kernel_radius: Sinc kernel radius in grid cells (default: 3)
                          Larger values increase accuracy but also computation cost
            
        Returns:
            mask: Array of shape matching grid dimensions with interpolated weights
        """
        if kernel_radius is None:
            kernel_radius = self.DEFAULT_KERNEL_RADIUS
            
        # Generate surface points for this element
        points = self.generate_element_surface_points(element_idx, n_points_x, n_points_y)
        
        # Adjust points z-coordinate
        points[:, 2] = z0
        
        # Create mask array
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float32)
        
        # Grid offset for staggered grid (half-cell shift)
        offset_x = grid.dx / 2.0 if staggered else 0.0
        offset_y = grid.dy / 2.0 if staggered else 0.0
        offset_z = grid.dz / 2.0 if staggered else 0.0
        
        # For each surface point, apply band-limited interpolation using sinc function
        # Sinc interpolation spreads point contribution to nearby grid cells
        weight_per_point = 1.0 / len(points)
        
        for point in points:
            px, py, pz = point
            
            # Convert world coordinates to grid indices (centered coordinate system)
            ix_center, iy_center, iz_center = self._world_to_centered_grid_index(px, py, pz, grid)
            
            # Loop over neighborhood
            for ix in range(max(0, ix_center - kernel_radius), 
                          min(grid.nx, ix_center + kernel_radius + 1)):
                for iy in range(max(0, iy_center - kernel_radius), 
                              min(grid.ny, iy_center + kernel_radius + 1)):
                    for iz in range(max(0, iz_center - kernel_radius), 
                                  min(grid.nz, iz_center + kernel_radius + 1)):
                        # Grid cell center position (centered coordinate system)
                        gx, gy, gz = self._centered_grid_index_to_world(
                            ix, iy, iz, grid, offset_x, offset_y, offset_z
                        )
                        
                        # Distance in grid units
                        dist_x = (px - gx) / grid.dx
                        dist_y = (py - gy) / grid.dy
                        dist_z = (pz - gz) / grid.dz
                        
                        # Band-limited sinc interpolation kernel
                        # Using numpy's optimized sinc: sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
                        sinc_x = np.sinc(dist_x)
                        sinc_y = np.sinc(dist_y)
                        sinc_z = np.sinc(dist_z)
                        
                        # Combined weight
                        weight = sinc_x * sinc_y * sinc_z * weight_per_point
                        mask[ix, iy, iz] += weight
        
        return mask

    def create_element_masks(self, grid, z0=0.0, n_points_x=5, n_points_y=5, 
                            staggered=False, kernel_radius=None):
        """Create masks for all transducer elements.
        
        Note: For large grids with many elements, this method can consume significant memory
        (each mask is nx × ny × nz × 4 bytes). Consider:
        - Computing masks on-demand if memory is limited
        - Using sparse matrix representations for storage
        - Reducing grid size or number of active elements
        
        Args:
            grid: Grid object defining the computational domain
            z0: Z-position of the transducer surface in meters
            n_points_x: Number of sample points along element width
            n_points_y: Number of sample points along element height
            staggered: If True, use staggered grid offsets
            kernel_radius: Sinc kernel radius in grid cells (default: 3)
            
        Returns:
            masks: List of mask arrays, one for each element
        """
        masks = []
        for elem_idx in range(self.n_elements):
            mask = self.band_limited_interpolation_mask(
                grid, elem_idx, n_points_x, n_points_y, z0, staggered, kernel_radius
            )
            masks.append(mask)
        return masks
