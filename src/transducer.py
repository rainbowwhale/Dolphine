import numpy as np


class Transducer:
    """Transducer element geometry and beamforming utilities.

    Default parameters set to typical medical linear probe values.
    
    Supports:
    - Single-row linear arrays (n_rows=1)
    - Multi-row arrays (n_rows > 1) with uniform or per-row element heights
    - 2D matrix arrays (configured via element_positions)
    - Convex/curved arrays (radius != None, single-row only)
    
    Args:
        n_elements: Total number of elements
        pitch: Element spacing in azimuth direction (x-axis)
        element_width: Physical width of each element
        element_height: Default height for all elements (can be overridden by row_heights)
        kerf: Gap between elements
        center_freq: Center frequency in Hz
        c: Speed of sound in m/s
        n_rows: Number of rows for multi-row arrays (must be 1 if radius is specified)
        elevation_pitch: Element spacing in elevation direction (y-axis), defaults to pitch
        row_heights: Optional per-row heights. Can be:
            - None: all rows use element_height (default)
            - Single value: all rows use this height
            - List/array of n_rows values: each row has specific height (e.g., [0.001, 0.003, 0.005, 0.003, 0.001])
        radius: Radius of curvature for convex arrays (in meters, must be > 0). If provided, creates a convex array.
        angle_span: Angular span for convex arrays in degrees (must be > 0 and <= 180, default 60). Only used if radius is specified.
    """

    def __init__(self, n_elements=64, pitch=0.0003, element_width=0.00028, element_height=0.00028, 
                 kerf=0.00002, center_freq=5e6, c=1540.0, n_rows=1, elevation_pitch=None, row_heights=None,
                 radius=None, angle_span=60.0):
        self.n_elements = n_elements
        self.pitch = pitch
        self.element_width = element_width
        self.kerf = kerf
        self.center_freq = center_freq
        self.c = c
        self.n_rows = n_rows
        self.radius = radius
        self.angle_span = angle_span
        
        # For multi-row arrays, use elevation_pitch if provided, otherwise use pitch
        self.elevation_pitch = elevation_pitch if elevation_pitch is not None else pitch
        
        # Validate n_rows
        if n_rows < 1:
            raise ValueError("n_rows must be >= 1")
        
        # Validate convex array parameters
        if radius is not None:
            if radius <= 0:
                raise ValueError(f"radius must be positive, got {radius}")
            if angle_span <= 0 or angle_span > 180:
                raise ValueError(f"angle_span must be positive and <= 180 degrees, got {angle_span}")
            if n_rows > 1:
                raise ValueError("Convex arrays (radius != None) do not support multi-row configuration (n_rows > 1)")
        
        # Handle row_heights: can be a single value or an array of heights per row
        if row_heights is not None:
            if isinstance(row_heights, (list, tuple, np.ndarray)):
                # Array of heights per row
                if len(row_heights) != n_rows:
                    raise ValueError(f"row_heights length ({len(row_heights)}) must match n_rows ({n_rows})")
                self.row_heights = np.array(row_heights)
                # For backward compatibility, set element_height to the first row's height or mean
                self.element_height = self.row_heights[0] if n_rows == 1 else np.mean(self.row_heights)
            else:
                # Single value for all rows
                self.row_heights = np.full(n_rows, row_heights)
                self.element_height = row_heights
        else:
            # Use element_height for all rows (backward compatible)
            self.row_heights = np.full(n_rows, element_height)
            self.element_height = element_height
        
        # Generate element center positions
        if radius is not None:
            # Convex array: arrange elements along an arc in x-z plane
            # Elements are distributed with uniform angular spacing across the arc
            # Arc center is at origin (0, 0, 0) with elements positioned at positive z
            theta_span = np.deg2rad(angle_span)
            thetas = np.linspace(-theta_span/2, theta_span/2, n_elements)
            x_positions = radius * np.sin(thetas)
            y_positions = np.zeros_like(thetas)
            z_positions = radius * (1 - np.cos(thetas))  # Positions arc forward (positive z direction)
        elif n_rows == 1:
            # Single-row linear array: positions along x-axis centered at zero
            x_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch
            y_positions = np.zeros_like(x_positions)
            z_positions = np.zeros_like(x_positions)
        else:
            # Multi-row array: distribute elements in x-y grid
            # Check if n_elements is evenly divisible by n_rows
            if n_elements % n_rows != 0:
                raise ValueError(f"n_elements ({n_elements}) must be evenly divisible by n_rows ({n_rows})")
            
            elements_per_row = n_elements // n_rows
            x_positions = []
            y_positions = []
            z_positions = []
            for row_idx in range(n_rows):
                row_x = (np.arange(elements_per_row) - (elements_per_row - 1) / 2.0) * pitch
                row_y = np.full_like(row_x, (row_idx - (n_rows - 1) / 2.0) * self.elevation_pitch)
                row_z = np.zeros_like(row_x)
                x_positions.extend(row_x)
                y_positions.extend(row_y)
                z_positions.extend(row_z)
            x_positions = np.array(x_positions)
            y_positions = np.array(y_positions)
            z_positions = np.array(z_positions)
        
        self.element_positions = np.stack([x_positions, y_positions, z_positions], axis=1)  # (x, y, z)

    def delays_for_focus(self, focus_point, speed_of_sound=None):
        """Compute transmission delays for each element to focus at `focus_point` in meters.

        Args:
            focus_point: tuple of (x, z) or (x, y, z) coordinates in meters
            speed_of_sound: optional override for speed of sound
            
        Returns delays in seconds (non-negative, relative to minimum).
        """
        if speed_of_sound is None:
            c = self.c
        else:
            c = speed_of_sound
        pos = self.element_positions
        
        # Validate focus_point
        if len(focus_point) not in (2, 3):
            raise ValueError(f"focus_point must have 2 or 3 elements, got {len(focus_point)}")
        
        # Handle both 2D (x, z) and 3D (x, y, z) focus points
        if len(focus_point) == 2:
            # 2D focus point (x, z)
            dx = pos[:, 0] - focus_point[0]
            dy = pos[:, 1]
            dz = focus_point[1]
        else:
            # 3D focus point (x, y, z)
            dx = pos[:, 0] - focus_point[0]
            dy = pos[:, 1] - focus_point[1]
            dz = focus_point[2]
            
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        delays = distances / c
        delays -= delays.min()
        return delays

    def apodization_hanning(self):
        """Return Hanning apodization weights across elements."""
        return np.hanning(self.n_elements)

    def map_to_grid(self, grid, z0=0.0):
        """Map element centers to grid indices (ix, iy, iz) using `Grid` object.
        
        Args:
            grid: Grid object with world_to_index method
            z0: z-coordinate for element centers (default 0.0)
            
        Returns:
            List of (ix, iy, iz) tuples for each element
        """
        idx = []
        for pos in self.element_positions:
            x, y, z_elem = pos
            # Use element's z-coordinate if it's non-zero, otherwise use z0
            z = z_elem if z_elem != 0.0 else z0
            ix, iy, iz = grid.world_to_index(x, y, z)
            idx.append((ix, iy, iz))
        return idx
