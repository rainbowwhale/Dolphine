"""
Transducer Module - Clean Implementation

Ultrasound transducer array modeling with acoustic lens support.

This is a clean implementation (not refactoring) with the following features:
- N x M array structure (n_cols x n_rows)
- Element properties: position (3D), angle, size (width x height)
- Array ROC (radius of curvature)
- Multi-layer acoustic lens support with elevational ROC
- Array dimension calculations
- Plotting functions for element array and lens shape
"""
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


class LensLayer:
    """Single layer of an acoustic lens.
    
    Attributes:
        elevational_roc: Radius of curvature in elevation direction (m).
                        Positive = convex, Negative = concave, 0 or inf = flat.
        max_thickness: Maximum thickness of the lens layer (m).
        speed_of_sound: Speed of sound in the lens material (m/s).
        density: Density of the lens material (kg/m³).
        name: Optional name/identifier for the layer.
    """
    
    def __init__(self, elevational_roc: float, max_thickness: float,
                 speed_of_sound: float = 1000.0, density: float = 1100.0,
                 name: str = ""):
        """Initialize a lens layer.
        
        Args:
            elevational_roc: Radius of curvature in elevation (m).
                           Positive for convex, negative for concave, 0/inf for flat.
            max_thickness: Maximum thickness of the layer (m).
            speed_of_sound: Speed of sound in lens material (m/s). Default 1000 m/s.
            density: Density of lens material (kg/m³). Default 1100 kg/m³.
            name: Optional name for the layer.
        """
        self.elevational_roc = elevational_roc
        self.max_thickness = max_thickness
        self.speed_of_sound = speed_of_sound
        self.density = density
        self.name = name
    
    @property
    def is_convex(self) -> bool:
        """Check if the lens layer is convex."""
        return self.elevational_roc > 0 and np.isfinite(self.elevational_roc)
    
    @property
    def is_concave(self) -> bool:
        """Check if the lens layer is concave."""
        return self.elevational_roc < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if the lens layer is flat."""
        return self.elevational_roc == 0 or not np.isfinite(self.elevational_roc)
    
    def get_thickness_profile(self, y_positions: np.ndarray) -> np.ndarray:
        """Calculate thickness profile along elevation direction.
        
        Args:
            y_positions: Array of y-positions (elevation) in meters.
            
        Returns:
            thickness: Array of thickness values at each y-position.
        """
        if self.is_flat:
            return np.full_like(y_positions, self.max_thickness, dtype=float)
        
        roc = abs(self.elevational_roc)
        # Calculate lens surface profile using arc equation
        # For convex: thicker in center, thinner at edges
        # For concave: thinner in center, thicker at edges
        
        # Clamp y_positions to valid range for arc calculation
        y_clipped = np.clip(y_positions, -roc, roc)
        
        # Height of arc at each y position
        arc_height = roc - np.sqrt(np.maximum(0, roc**2 - y_clipped**2))
        
        if self.is_convex:
            # Convex: max thickness at center (y=0), decreasing toward edges
            thickness = self.max_thickness - arc_height
        else:
            # Concave: min thickness at center, increasing toward edges
            thickness = arc_height + (self.max_thickness - arc_height.max())
        
        # Ensure non-negative thickness
        return np.maximum(0, thickness)


class AcousticLens:
    """Multi-layer acoustic lens for ultrasound transducers.
    
    Supports multiple lens layers with different properties for
    complex lens designs used in medical ultrasound transducers.
    
    Attributes:
        layers: List of LensLayer objects.
    """
    
    def __init__(self, layers: list = None):
        """Initialize an acoustic lens.
        
        Args:
            layers: List of LensLayer objects. If None, creates an empty lens.
        """
        self.layers = layers if layers is not None else []
    
    def add_layer(self, layer: LensLayer):
        """Add a layer to the lens.
        
        Args:
            layer: LensLayer object to add.
        """
        self.layers.append(layer)
    
    @property
    def n_layers(self) -> int:
        """Number of layers in the lens."""
        return len(self.layers)
    
    @property
    def total_max_thickness(self) -> float:
        """Total maximum thickness of all layers."""
        if not self.layers:
            return 0.0
        return sum(layer.max_thickness for layer in self.layers)
    
    def get_total_thickness_profile(self, y_positions: np.ndarray) -> np.ndarray:
        """Calculate total thickness profile of all layers.
        
        Args:
            y_positions: Array of y-positions (elevation) in meters.
            
        Returns:
            total_thickness: Array of total thickness values at each y-position.
        """
        if not self.layers:
            return np.zeros_like(y_positions, dtype=float)
        
        total = np.zeros_like(y_positions, dtype=float)
        for layer in self.layers:
            total += layer.get_thickness_profile(y_positions)
        return total
    
    def get_layer_boundaries(self, y_positions: np.ndarray) -> list:
        """Calculate z-positions of layer boundaries.
        
        Args:
            y_positions: Array of y-positions (elevation) in meters.
            
        Returns:
            boundaries: List of arrays, each containing z-positions
                       of the boundary between layers.
        """
        if not self.layers:
            return []
        
        boundaries = []
        cumulative_z = np.zeros_like(y_positions, dtype=float)
        
        for layer in self.layers:
            thickness = layer.get_thickness_profile(y_positions)
            cumulative_z = cumulative_z + thickness
            boundaries.append(cumulative_z.copy())
        
        return boundaries


class Transducer:
    """Ultrasound transducer array with acoustic lens support.
    
    Models an N x M array of transducer elements where:
    - N = n_cols (number of columns, lateral direction)
    - M = n_rows (number of rows, elevation direction)
    
    Each element has:
    - 3D position (x, y, z)
    - Normal angle (for curved arrays)
    - Size (width x height)
    
    The array can have:
    - Radius of curvature (ROC) for curved arrays
    - Multi-layer acoustic lens with elevational focusing
    
    Attributes:
        n_cols: Number of columns (lateral direction).
        n_rows: Number of rows (elevation direction).
        n_elements: Total number of elements (n_cols * n_rows).
        pitch: Lateral pitch (element spacing in x-direction) in meters.
        row_pitch: Elevation pitch (element spacing in y-direction) in meters.
        element_width: Width of each element (lateral) in meters.
        element_height: Height of each element (elevation) in meters.
        kerf: Gap between elements in meters.
        roc: Radius of curvature of the array (0 for flat).
        center_freq: Center frequency in Hz.
        speed_of_sound: Speed of sound in medium (m/s).
        element_positions: (N, 3) array of element center positions.
        element_angles: (N, 2) array of element normal angles (theta_x, theta_y).
        lens: AcousticLens object for lens modeling.
    """
    
    def __init__(self, n_cols: int, n_rows: int,
                 pitch: float = 0.0003, row_pitch: float = None,
                 element_width: float = None, element_height: float = None,
                 kerf: float = 0.00002,
                 roc: float = 0.0,
                 center_freq: float = 5e6, speed_of_sound: float = 1540.0,
                 lens: AcousticLens = None):
        """Initialize the transducer array.
        
        Args:
            n_cols: Number of columns (lateral/x direction).
            n_rows: Number of rows (elevation/y direction).
            pitch: Center-to-center spacing between columns (m). Default 0.3mm.
            row_pitch: Center-to-center spacing between rows (m). Default equals pitch.
            element_width: Width of each element (m). Default pitch - kerf.
            element_height: Height of each element (m). Default row_pitch - kerf.
            kerf: Gap between elements (m). Default 0.02mm.
            roc: Radius of curvature (m). 0 for flat array, >0 for convex.
            center_freq: Center frequency (Hz). Default 5 MHz.
            speed_of_sound: Speed of sound in medium (m/s). Default 1540 m/s.
            lens: AcousticLens object. Default None (no lens).
        """
        # Array dimensions
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_elements = n_cols * n_rows
        
        # Element spacing
        self.pitch = pitch
        self.row_pitch = row_pitch if row_pitch is not None else pitch
        self.kerf = kerf
        
        # Element dimensions
        self.element_width = element_width if element_width is not None else (pitch - kerf)
        self.element_height = element_height if element_height is not None else (self.row_pitch - kerf)
        
        # Array curvature
        self.roc = roc
        
        # Acoustic properties
        self.center_freq = center_freq
        self.speed_of_sound = speed_of_sound
        
        # Acoustic lens
        self.lens = lens if lens is not None else AcousticLens()
        
        # Generate element positions and angles
        self._generate_element_geometry()
    
    def _generate_element_geometry(self):
        """Generate element positions and normal angles based on array geometry."""
        # Calculate element positions
        if self.roc > 0:
            # Curved array (convex)
            self.element_positions, self.element_angles = self._generate_curved_geometry()
        else:
            # Flat array
            self.element_positions, self.element_angles = self._generate_flat_geometry()
    
    def _generate_flat_geometry(self) -> tuple:
        """Generate positions and angles for a flat array.
        
        Returns:
            positions: (N, 3) array of element positions.
            angles: (N, 2) array of element normal angles.
        """
        # X positions (lateral) - centered at origin
        x_positions = (np.arange(self.n_cols) - (self.n_cols - 1) / 2.0) * self.pitch
        
        # Y positions (elevation) - centered at origin
        y_positions = (np.arange(self.n_rows) - (self.n_rows - 1) / 2.0) * self.row_pitch
        
        # Create meshgrid (row-major order: iterate over rows first)
        xx, yy = np.meshgrid(x_positions, y_positions, indexing='xy')
        
        # All elements at z=0 for flat array
        zz = np.zeros_like(xx)
        
        # Stack into (N, 3) array
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        # All elements have normal pointing in +z direction (angle = 0, 0)
        angles = np.zeros((self.n_elements, 2))
        
        return positions, angles
    
    def _generate_curved_geometry(self) -> tuple:
        """Generate positions and angles for a curved (convex) array.
        
        The curvature is applied in the lateral (x-z) plane.
        All rows follow the same curved arc.
        
        Returns:
            positions: (N, 3) array of element positions.
            angles: (N, 2) array of element normal angles (theta_x, theta_y).
        """
        # Calculate angular positions for lateral curvature
        arc_length = (self.n_cols - 1) * self.pitch
        theta_span = arc_length / self.roc  # Total angular span
        thetas = np.linspace(-theta_span / 2, theta_span / 2, self.n_cols)
        
        # X and Z positions on the curved arc
        x_positions = self.roc * np.sin(thetas)
        z_positions = self.roc * (1 - np.cos(thetas))
        
        # Y positions (elevation) - centered at origin
        y_positions = (np.arange(self.n_rows) - (self.n_rows - 1) / 2.0) * self.row_pitch
        
        # Create full arrays for all elements (row-major order)
        xx = np.tile(x_positions, self.n_rows)
        yy = np.repeat(y_positions, self.n_cols)
        zz = np.tile(z_positions, self.n_rows)
        
        positions = np.stack([xx, yy, zz], axis=1)
        
        # Element angles: theta_x varies with lateral position, theta_y = 0
        theta_x = np.tile(thetas, self.n_rows)
        theta_y = np.zeros(self.n_elements)
        
        angles = np.stack([theta_x, theta_y], axis=1)
        
        return positions, angles
    
    # ===== Array Properties =====
    
    @property
    def array_width(self) -> float:
        """Width of the array (lateral dimension) in meters."""
        return (self.n_cols - 1) * self.pitch + self.element_width
    
    @property
    def array_height(self) -> float:
        """Height of the array (elevation dimension) in meters."""
        return (self.n_rows - 1) * self.row_pitch + self.element_height
    
    @property
    def array_size(self) -> tuple:
        """Size of the array as (width, height) in meters."""
        return (self.array_width, self.array_height)
    
    @property
    def min_dimension(self) -> float:
        """Minimum dimension of the array in meters."""
        return min(self.array_width, self.array_height)
    
    @property
    def max_dimension(self) -> float:
        """Maximum dimension of the array in meters."""
        return max(self.array_width, self.array_height)
    
    @property
    def wavelength(self) -> float:
        """Wavelength at center frequency in meters."""
        return self.speed_of_sound / self.center_freq
    
    @property
    def x_positions(self) -> np.ndarray:
        """X-coordinates of all element centers."""
        return self.element_positions[:, 0]
    
    @property
    def y_positions(self) -> np.ndarray:
        """Y-coordinates of all element centers."""
        return self.element_positions[:, 1]
    
    @property
    def z_positions(self) -> np.ndarray:
        """Z-coordinates of all element centers."""
        return self.element_positions[:, 2]
    
    # ===== Element Access Methods =====
    
    def get_element_position(self, element_idx: int) -> np.ndarray:
        """Get the 3D position of a specific element.
        
        Args:
            element_idx: Element index (0 to n_elements-1).
            
        Returns:
            position: (3,) array with (x, y, z) coordinates.
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        return self.element_positions[element_idx]
    
    def get_element_angle(self, element_idx: int) -> np.ndarray:
        """Get the normal angles of a specific element.
        
        Args:
            element_idx: Element index (0 to n_elements-1).
            
        Returns:
            angles: (2,) array with (theta_x, theta_y) angles in radians.
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        return self.element_angles[element_idx]
    
    def get_element_size(self, element_idx: int = None) -> tuple:
        """Get the size of an element.
        
        Args:
            element_idx: Element index (optional, all elements have same size).
            
        Returns:
            size: (width, height) tuple in meters.
        """
        return (self.element_width, self.element_height)
    
    def get_element_row_col(self, element_idx: int) -> tuple:
        """Get the row and column indices for an element.
        
        Args:
            element_idx: Linear element index (0 to n_elements-1).
            
        Returns:
            (row_idx, col_idx): Row and column indices.
        """
        row_idx = element_idx // self.n_cols
        col_idx = element_idx % self.n_cols
        return row_idx, col_idx
    
    def get_element_index(self, row: int, col: int) -> int:
        """Get the linear element index from row and column.
        
        Args:
            row: Row index (0 to n_rows-1).
            col: Column index (0 to n_cols-1).
            
        Returns:
            element_idx: Linear element index.
        """
        if row < 0 or row >= self.n_rows:
            raise ValueError(f"Row {row} out of range [0, {self.n_rows})")
        if col < 0 or col >= self.n_cols:
            raise ValueError(f"Column {col} out of range [0, {self.n_cols})")
        return row * self.n_cols + col
    
    # ===== Beamforming Methods =====
    
    def delays_for_focus(self, focus_point: tuple, speed_of_sound: float = None) -> np.ndarray:
        """Compute transmission delays for focusing at a point.
        
        Args:
            focus_point: (x, y, z) or (x, z) focus point in meters.
            speed_of_sound: Speed of sound (m/s). Uses self.speed_of_sound if None.
            
        Returns:
            delays: Array of delays for each element (seconds).
        """
        c = speed_of_sound if speed_of_sound is not None else self.speed_of_sound
        
        # Handle 2D or 3D focus point
        if len(focus_point) == 2:
            focus_x, focus_z = focus_point
            focus_y = 0.0
        else:
            focus_x, focus_y, focus_z = focus_point
        
        # Compute distances from each element to focus point
        dx = self.element_positions[:, 0] - focus_x
        dy = self.element_positions[:, 1] - focus_y
        dz = self.element_positions[:, 2] - focus_z
        
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Convert to delays and normalize
        delays = distances / c
        delays -= delays.min()
        
        return delays
    
    def delays_for_steering(self, steering_angles: tuple, speed_of_sound: float = None) -> np.ndarray:
        """Compute transmission delays for beam steering.
        
        Args:
            steering_angles: (theta_x, theta_y) steering angles in radians.
            speed_of_sound: Speed of sound (m/s). Uses self.speed_of_sound if None.
            
        Returns:
            delays: Array of delays for each element (seconds).
        """
        c = speed_of_sound if speed_of_sound is not None else self.speed_of_sound
        
        theta_x, theta_y = steering_angles
        
        # Compute delays based on plane wave steering
        delays = (
            self.element_positions[:, 0] * np.sin(theta_x) +
            self.element_positions[:, 1] * np.sin(theta_y)
        ) / c
        
        # Normalize to positive delays
        delays -= delays.min()
        
        return delays
    
    def apodization_hanning(self) -> np.ndarray:
        """Generate Hanning apodization weights.
        
        For 2D arrays, returns a 2D Hanning window flattened to 1D.
        
        Returns:
            weights: Apodization weights for each element.
        """
        if self.n_rows == 1:
            return np.hanning(self.n_cols)
        else:
            # 2D Hanning window
            hann_x = np.hanning(self.n_cols)
            hann_y = np.hanning(self.n_rows)
            hann_2d = np.outer(hann_y, hann_x)
            return hann_2d.flatten()
    
    # ===== Plotting Methods =====
    
    def plot_array(self, ax=None, show_elements: bool = True,
                   show_normals: bool = False, normal_length: float = 0.001,
                   figsize: tuple = (10, 8)):
        """Plot the transducer element array.
        
        Args:
            ax: Matplotlib axes (3D). Creates new figure if None.
            show_elements: Show element rectangles.
            show_normals: Show element normal vectors.
            normal_length: Length of normal vectors in meters.
            figsize: Figure size if creating new figure.
            
        Returns:
            ax: The matplotlib axes object.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot element centers
        ax.scatter(
            self.x_positions * 1e3,
            self.y_positions * 1e3,
            self.z_positions * 1e3,
            c='blue', s=20, label='Element centers'
        )
        
        # Plot element rectangles (simplified as points for now)
        if show_elements:
            for i in range(self.n_elements):
                x, y, z = self.element_positions[i] * 1e3
                w = self.element_width * 1e3 / 2
                h = self.element_height * 1e3 / 2
                
                # Draw element outline (rectangle in x-y plane for flat array)
                if self.roc == 0:
                    rect_x = [x-w, x+w, x+w, x-w, x-w]
                    rect_y = [y-h, y-h, y+h, y+h, y-h]
                    rect_z = [z, z, z, z, z]
                    ax.plot(rect_x, rect_y, rect_z, 'b-', alpha=0.5, linewidth=0.5)
        
        # Plot normal vectors
        if show_normals:
            for i in range(self.n_elements):
                x, y, z = self.element_positions[i]
                theta_x, theta_y = self.element_angles[i]
                
                # Normal direction
                nx = np.sin(theta_x)
                ny = np.sin(theta_y)
                nz = np.cos(theta_x) * np.cos(theta_y)
                
                # Scale and plot
                ax.quiver(
                    x * 1e3, y * 1e3, z * 1e3,
                    nx * normal_length * 1e3,
                    ny * normal_length * 1e3,
                    nz * normal_length * 1e3,
                    color='red', alpha=0.5
                )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Transducer Array ({self.n_cols}×{self.n_rows} elements)')
        
        # Equal aspect ratio
        max_range = max(self.array_width, self.array_height) * 1e3 / 2
        ax.set_xlim(-max_range * 1.1, max_range * 1.1)
        ax.set_ylim(-max_range * 1.1, max_range * 1.1)
        
        return ax
    
    def plot_lens(self, ax=None, n_points: int = 100, figsize: tuple = (10, 6)):
        """Plot the acoustic lens cross-section.
        
        Shows the lens thickness profile along the elevation direction.
        
        Args:
            ax: Matplotlib axes. Creates new figure if None.
            n_points: Number of points for lens profile.
            figsize: Figure size if creating new figure.
            
        Returns:
            ax: The matplotlib axes object.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if not self.lens.layers:
            print("No lens layers defined. Nothing to plot.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Y positions spanning the array elevation
        y_range = self.array_height / 2 * 1.2
        y_positions = np.linspace(-y_range, y_range, n_points)
        
        # Plot each layer
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, self.lens.n_layers))
        
        cumulative_z = np.zeros(n_points)
        
        for i, layer in enumerate(self.lens.layers):
            thickness = layer.get_thickness_profile(y_positions)
            
            # Plot filled region for this layer
            ax.fill_between(
                y_positions * 1e3,
                cumulative_z * 1e3,
                (cumulative_z + thickness) * 1e3,
                alpha=0.5,
                color=colors[i],
                label=f'Layer {i+1}: {layer.name}' if layer.name else f'Layer {i+1}'
            )
            
            # Plot boundary line
            ax.plot(y_positions * 1e3, (cumulative_z + thickness) * 1e3, 
                   color=colors[i], linewidth=1.5)
            
            cumulative_z = cumulative_z + thickness
        
        # Plot transducer surface line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Transducer surface')
        
        # Mark array extent
        ax.axvline(x=-self.array_height/2 * 1e3, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=self.array_height/2 * 1e3, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Elevation (mm)')
        ax.set_ylabel('Thickness (mm)')
        ax.set_title('Acoustic Lens Cross-Section')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_array_2d(self, ax=None, figsize: tuple = (10, 8)):
        """Plot the transducer element array in 2D (top-down view).
        
        Args:
            ax: Matplotlib axes. Creates new figure if None.
            figsize: Figure size if creating new figure.
            
        Returns:
            ax: The matplotlib axes object.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Draw each element as a rectangle
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        
        patches = []
        for i in range(self.n_elements):
            x, y, z = self.element_positions[i]
            w = self.element_width
            h = self.element_height
            
            # Rectangle centered at element position
            rect = Rectangle(
                (x - w/2, y - h/2),
                w, h
            )
            patches.append(rect)
        
        # Color by z-position for curved arrays
        collection = PatchCollection(patches, cmap='viridis', alpha=0.8, edgecolor='black', linewidth=0.5)
        collection.set_array(self.z_positions)
        ax.add_collection(collection)
        
        # Add colorbar for curved arrays
        if self.roc > 0:
            plt.colorbar(collection, ax=ax, label='Z position (m)')
        
        ax.set_xlim(-self.array_width/2 * 1.1, self.array_width/2 * 1.1)
        ax.set_ylim(-self.array_height/2 * 1.1, self.array_height/2 * 1.1)
        ax.set_xlabel('X - Lateral (m)')
        ax.set_ylabel('Y - Elevation (m)')
        ax.set_title(f'Transducer Array ({self.n_cols}×{self.n_rows} elements)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    # ===== BLI (Band-Limited Interpolation) Methods =====
    
    def generate_element_surface_points(self, element_idx: int,
                                        n_points_x: int, n_points_y: int) -> np.ndarray:
        """Generate uniformly distributed points on element surface.
        
        Args:
            element_idx: Index of the element.
            n_points_x: Number of points along element width.
            n_points_y: Number of points along element height.
            
        Returns:
            points: (n_points_x * n_points_y, 3) array of (x, y, z) coordinates.
        """
        if element_idx < 0 or element_idx >= self.n_elements:
            raise ValueError(f"Element index {element_idx} out of range [0, {self.n_elements})")
        
        # Get element center
        center_x, center_y, center_z = self.element_positions[element_idx]
        
        # Generate uniform grid on element surface
        x_samples = np.linspace(-self.element_width/2, self.element_width/2, n_points_x)
        y_samples = np.linspace(-self.element_height/2, self.element_height/2, n_points_y)
        
        xx, yy = np.meshgrid(x_samples, y_samples, indexing='xy')
        
        # Offset by element center
        x_coords = center_x + xx.flatten()
        y_coords = center_y + yy.flatten()
        z_coords = np.full_like(x_coords, center_z)
        
        return np.stack([x_coords, y_coords, z_coords], axis=1)
    
    def create_element_mask(self, grid, element_idx: int,
                           n_points_x: int = 5, n_points_y: int = 5,
                           kernel_radius: int = 3, tolerance: float = 1e-3,
                           staggered_component: str = None,
                           use_gpu: bool = False) -> tuple:
        """Create BLI mask for a single element.
        
        Args:
            grid: Grid object with axis vectors.
            element_idx: Element index.
            n_points_x: Sample points along width.
            n_points_y: Sample points along height.
            kernel_radius: Sinc kernel radius in grid cells.
            tolerance: Weight threshold for BLI star selection.
            staggered_component: None, 'x', 'y', or 'z' for staggered grids.
            use_gpu: Use GPU acceleration.
            
        Returns:
            indices: (N, 3) array of grid indices.
            weights: (N,) array of weights.
        """
        # Generate surface points
        points = self.generate_element_surface_points(element_idx, n_points_x, n_points_y)
        
        # Compute BLI weights
        return self._band_limited_interpolation(
            grid, points, kernel_radius, tolerance, staggered_component, use_gpu
        )
    
    def create_all_element_masks(self, grid, n_points_x: int = 5, n_points_y: int = 5,
                                 kernel_radius: int = 3, tolerance: float = 1e-3,
                                 staggered: bool = False, use_gpu: bool = False) -> list:
        """Create BLI masks for all elements.
        
        Args:
            grid: Grid object.
            n_points_x: Sample points per element width.
            n_points_y: Sample points per element height.
            kernel_radius: Sinc kernel radius.
            tolerance: Weight threshold.
            staggered: If True, return staggered masks for velocity components.
            use_gpu: Use GPU acceleration.
            
        Returns:
            If staggered=False: List of (indices, weights) tuples.
            If staggered=True: List of dicts with 'vx', 'vy', 'vz' keys.
        """
        masks = []
        
        for elem_idx in range(self.n_elements):
            if staggered:
                mask = {
                    'vx': self.create_element_mask(grid, elem_idx, n_points_x, n_points_y,
                                                   kernel_radius, tolerance, 'x', use_gpu),
                    'vy': self.create_element_mask(grid, elem_idx, n_points_x, n_points_y,
                                                   kernel_radius, tolerance, 'y', use_gpu),
                    'vz': self.create_element_mask(grid, elem_idx, n_points_x, n_points_y,
                                                   kernel_radius, tolerance, 'z', use_gpu),
                }
            else:
                mask = self.create_element_mask(
                    grid, elem_idx, n_points_x, n_points_y,
                    kernel_radius, tolerance, None, use_gpu
                )
            masks.append(mask)
        
        return masks
    
    def _band_limited_interpolation(self, grid, points: np.ndarray,
                                    kernel_radius: int, tolerance: float,
                                    staggered_component: str, use_gpu: bool) -> tuple:
        """Compute BLI weights for source points.
        
        Uses sinc interpolation: weight = sinc((px-gx)/dx) * sinc((py-gy)/dy) * sinc((pz-gz)/dz)
        
        Args:
            grid: Grid object with axis vectors.
            points: (N, 3) array of point coordinates.
            kernel_radius: Sinc kernel radius in grid cells.
            tolerance: Weight threshold.
            staggered_component: None, 'x', 'y', or 'z'.
            use_gpu: Use GPU acceleration.
            
        Returns:
            indices: (M, 3) array of grid indices.
            weights: (M,) array of weights.
        """
        xp = cp if use_gpu and HAS_CUPY else np
        
        if use_gpu and HAS_CUPY:
            points = cp.asarray(points)
            grid_x = cp.asarray(grid.x_vec)
            grid_y = cp.asarray(grid.y_vec)
            grid_z = cp.asarray(grid.z_vec)
        else:
            grid_x = grid.x_vec
            grid_y = grid.y_vec
            grid_z = grid.z_vec
        
        # Grid spacing
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        
        # Staggered grid offsets
        offset_x = dx / 2 if staggered_component == 'x' else 0
        offset_y = dy / 2 if staggered_component == 'y' else 0
        offset_z = dz / 2 if staggered_component == 'z' else 0
        
        # Grid origins with offset
        origin_x = grid_x[0] + offset_x
        origin_y = grid_y[0] + offset_y
        origin_z = grid_z[0] + offset_z
        
        # Normalized point positions
        px = (points[:, 0] - origin_x) / dx
        py = (points[:, 1] - origin_y) / dy
        pz = (points[:, 2] - origin_z) / dz
        
        # Base indices
        ix = xp.floor(px).astype(int)
        iy = xp.floor(py).astype(int)
        iz = xp.floor(pz).astype(int)
        
        # Fractional parts
        rx = px - ix
        ry = py - iy
        rz = pz - iz
        
        # BLI star offsets
        bli_range = np.arange(-kernel_radius, kernel_radius + 1)
        bli_x, bli_y, bli_z = np.meshgrid(bli_range, bli_range, bli_range, indexing='ij')
        bli_x = bli_x.flatten()
        bli_y = bli_y.flatten()
        bli_z = bli_z.flatten()
        
        # Filter by tolerance (1/(|x|*|y|*|z|) >= tolerance)
        with np.errstate(divide='ignore', invalid='ignore'):
            level_x = np.where(bli_x != 0, 1.0 / np.abs(bli_x), 1.0)
            level_y = np.where(bli_y != 0, 1.0 / np.abs(bli_y), 1.0)
            level_z = np.where(bli_z != 0, 1.0 / np.abs(bli_z), 1.0)
            bli_level = level_x * level_y * level_z
        
        selected = bli_level >= tolerance
        bli_x = bli_x[selected]
        bli_y = bli_y[selected]
        bli_z = bli_z[selected]
        
        if use_gpu and HAS_CUPY:
            bli_x = cp.asarray(bli_x)
            bli_y = cp.asarray(bli_y)
            bli_z = cp.asarray(bli_z)
        
        n_points = points.shape[0]
        n_bli = len(bli_x)
        
        # Compute sinc weights
        sinc_x = xp.sinc(rx[:, None] + bli_x[None, :])
        sinc_y = xp.sinc(ry[:, None] + bli_y[None, :])
        sinc_z = xp.sinc(rz[:, None] + bli_z[None, :])
        
        weights_all = sinc_x * sinc_y * sinc_z
        
        # Generate indices
        ix_all = ix[:, None] + bli_x[None, :]
        iy_all = iy[:, None] + bli_y[None, :]
        iz_all = iz[:, None] + bli_z[None, :]
        
        # Flatten
        ix_flat = ix_all.flatten()
        iy_flat = iy_all.flatten()
        iz_flat = iz_all.flatten()
        weights_flat = weights_all.flatten()
        
        # Filter valid indices
        valid = (
            (ix_flat >= 0) & (ix_flat < grid.nx) &
            (iy_flat >= 0) & (iy_flat < grid.ny) &
            (iz_flat >= 0) & (iz_flat < grid.nz)
        )
        
        ix_valid = ix_flat[valid]
        iy_valid = iy_flat[valid]
        iz_valid = iz_flat[valid]
        weights_valid = weights_flat[valid]
        
        if len(ix_valid) == 0:
            return np.array([], dtype=np.int32).reshape(0, 3), np.array([], dtype=np.float32)
        
        # Aggregate weights at same indices
        # Create local subgrid
        ix_min, ix_max = int(ix_valid.min()), int(ix_valid.max())
        iy_min, iy_max = int(iy_valid.min()), int(iy_valid.max())
        iz_min, iz_max = int(iz_valid.min()), int(iz_valid.max())
        
        subgrid_shape = (ix_max - ix_min + 1, iy_max - iy_min + 1, iz_max - iz_min + 1)
        weight_grid = xp.zeros(subgrid_shape, dtype=xp.float32)
        
        # Map to local indices
        ix_local = ix_valid - ix_min
        iy_local = iy_valid - iy_min
        iz_local = iz_valid - iz_min
        
        if use_gpu and HAS_CUPY:
            linear_idx = ix_local + iy_local * subgrid_shape[0] + iz_local * subgrid_shape[0] * subgrid_shape[1]
            cp.scatter_add(weight_grid.flatten(), linear_idx, weights_valid)
        else:
            np.add.at(weight_grid, (ix_local, iy_local, iz_local), weights_valid)
        
        # Extract non-zero entries
        nonzero = weight_grid != 0
        if use_gpu and HAS_CUPY:
            i_local, j_local, k_local = cp.where(nonzero)
            weights_out = weight_grid[nonzero]
            
            indices_i = cp.asnumpy(i_local) + ix_min
            indices_j = cp.asnumpy(j_local) + iy_min
            indices_k = cp.asnumpy(k_local) + iz_min
            weights_out = cp.asnumpy(weights_out)
        else:
            i_local, j_local, k_local = np.where(nonzero)
            weights_out = weight_grid[nonzero]
            
            indices_i = i_local + ix_min
            indices_j = j_local + iy_min
            indices_k = k_local + iz_min
        
        indices = np.stack([indices_i, indices_j, indices_k], axis=1).astype(np.int32)
        weights_out = weights_out.astype(np.float32)
        
        # Normalize
        weight_sum = weights_out.sum()
        if weight_sum > 1e-10:
            weights_out = weights_out / weight_sum
        
        return indices, weights_out
    
    # ===== String Representation =====
    
    def __repr__(self) -> str:
        return (f"Transducer(n_cols={self.n_cols}, n_rows={self.n_rows}, "
                f"pitch={self.pitch*1e3:.3f}mm, roc={self.roc*1e3:.1f}mm, "
                f"lens_layers={self.lens.n_layers})")
    
    def __str__(self) -> str:
        lines = [
            f"Transducer Array ({self.n_cols} × {self.n_rows} = {self.n_elements} elements)",
            f"  Pitch: {self.pitch*1e3:.3f}mm (lateral), {self.row_pitch*1e3:.3f}mm (elevation)",
            f"  Element size: {self.element_width*1e3:.3f}mm × {self.element_height*1e3:.3f}mm",
            f"  Array size: {self.array_width*1e3:.2f}mm × {self.array_height*1e3:.2f}mm",
            f"  ROC: {self.roc*1e3:.1f}mm {'(curved)' if self.roc > 0 else '(flat)'}",
            f"  Center frequency: {self.center_freq/1e6:.1f}MHz",
            f"  Wavelength: {self.wavelength*1e3:.3f}mm",
            f"  Lens layers: {self.lens.n_layers}",
        ]
        if self.lens.n_layers > 0:
            lines.append(f"  Lens max thickness: {self.lens.total_max_thickness*1e3:.3f}mm")
        return "\n".join(lines)


# Test the implementation
if __name__ == '__main__':
    print("=" * 70)
    print("TRANSDUCER MODULE TEST")
    print("=" * 70)
    
    # Test 1: Basic flat array
    print("\n=== Test 1: Basic Flat Array ===")
    tx1 = Transducer(n_cols=32, n_rows=8, pitch=0.0003, roc=0)
    print(tx1)
    print(f"\nElement 0 position: {tx1.get_element_position(0) * 1e3} mm")
    print(f"Element 0 angle: {tx1.get_element_angle(0)} rad")
    print(f"Element 0 size: {tx1.get_element_size()} m")
    
    # Test 2: Curved array
    print("\n=== Test 2: Curved Array ===")
    tx2 = Transducer(n_cols=64, n_rows=1, pitch=0.0003, roc=0.05)
    print(tx2)
    print(f"\nZ-range: [{tx2.z_positions.min()*1e3:.4f}, {tx2.z_positions.max()*1e3:.4f}] mm")
    
    # Test 3: Array with acoustic lens
    print("\n=== Test 3: Array with Acoustic Lens ===")
    lens = AcousticLens()
    lens.add_layer(LensLayer(elevational_roc=0.020, max_thickness=0.001, name="Focus layer"))
    lens.add_layer(LensLayer(elevational_roc=0, max_thickness=0.0005, name="Matching layer"))
    
    tx3 = Transducer(n_cols=64, n_rows=5, pitch=0.0003, row_pitch=0.0004, lens=lens)
    print(tx3)
    
    # Test lens profile
    y = np.linspace(-0.002, 0.002, 21)
    thickness = lens.get_total_thickness_profile(y)
    print(f"\nLens thickness range: [{thickness.min()*1e3:.3f}, {thickness.max()*1e3:.3f}] mm")
    
    # Test 4: Focusing delays
    print("\n=== Test 4: Focusing Delays ===")
    focus = (0.0, 0.0, 0.03)
    delays = tx1.delays_for_focus(focus)
    print(f"Focus at {focus[2]*1e3:.0f}mm depth")
    print(f"Delay range: [{delays.min()*1e6:.3f}, {delays.max()*1e6:.3f}] µs")
    
    # Test 5: Element indexing
    print("\n=== Test 5: Element Indexing ===")
    for idx in [0, 31, 32, 255]:
        if idx < tx1.n_elements:
            row, col = tx1.get_element_row_col(idx)
            print(f"Element {idx}: row={row}, col={col}")
    
    # Test 6: Array properties
    print("\n=== Test 6: Array Properties ===")
    print(f"Array size: {tx1.array_size[0]*1e3:.2f}mm × {tx1.array_size[1]*1e3:.2f}mm")
    print(f"Min dimension: {tx1.min_dimension*1e3:.2f}mm")
    print(f"Max dimension: {tx1.max_dimension*1e3:.2f}mm")
    
    # Test 7: Plotting (if matplotlib available)
    if HAS_MATPLOTLIB:
        print("\n=== Test 7: Plotting ===")
        print("Creating plots...")
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: 3D array view
        ax1 = fig.add_subplot(131, projection='3d')
        tx1.plot_array(ax=ax1, show_normals=False)
        
        # Plot 2: 2D array view
        ax2 = fig.add_subplot(132)
        tx1.plot_array_2d(ax=ax2)
        
        # Plot 3: Lens cross-section
        ax3 = fig.add_subplot(133)
        tx3.plot_lens(ax=ax3)
        
        plt.tight_layout()
        plt.savefig('/tmp/transducer_test_plots.png', dpi=150)
        print("Plots saved to /tmp/transducer_test_plots.png")
    else:
        print("\n=== Test 7: Plotting (skipped - matplotlib not available) ===")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)
