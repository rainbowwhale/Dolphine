import numpy as np


class Transducer:
    """Transducer element geometry and beamforming utilities.

    Default parameters set to typical medical linear probe values.
    
    Supports:
    - Single-row linear arrays (n_rows=1)
    - Multi-row arrays (n_rows > 1)
    - 2D matrix arrays (configured via element_positions)
    """

    def __init__(self, n_elements=64, pitch=0.0003, element_width=0.00028, element_height=0.00028, 
                 kerf=0.00002, center_freq=5e6, c=1540.0, n_rows=1):
        self.n_elements = n_elements
        self.pitch = pitch
        self.element_width = element_width
        self.element_height = element_height
        self.kerf = kerf
        self.center_freq = center_freq
        self.c = c
        self.n_rows = n_rows
        
        # Generate element center positions
        if n_rows == 1:
            # Single-row linear array: positions along x-axis centered at zero
            x_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch
            y_positions = np.zeros_like(x_positions)
            z_positions = np.zeros_like(x_positions)
        else:
            # Multi-row array: distribute elements in x-y grid
            # Assume n_elements is total number, distribute evenly across rows
            elements_per_row = n_elements // n_rows
            x_positions = []
            y_positions = []
            z_positions = []
            for row_idx in range(n_rows):
                row_x = (np.arange(elements_per_row) - (elements_per_row - 1) / 2.0) * pitch
                row_y = np.full_like(row_x, (row_idx - (n_rows - 1) / 2.0) * pitch)
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
        
        # Handle both 2D (x, z) and 3D (x, y, z) focus points
        if len(focus_point) == 2:
            # 2D focus point (x, z)
            dx = pos[:, 0] - focus_point[0]
            dy = pos[:, 1] - 0.0
            dz = focus_point[1] - 0.0
        else:
            # 3D focus point (x, y, z)
            dx = pos[:, 0] - focus_point[0]
            dy = pos[:, 1] - focus_point[1]
            dz = focus_point[2] - 0.0
            
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
