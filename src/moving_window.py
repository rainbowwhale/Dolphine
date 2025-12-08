class MovingWindow:
    """Manage a moving window over the full scan domain.

    Keeps track of the current window origin (in grid indices) and extracts
    leading-edge slices from a `Medium` instance for the solver.
    """

    def __init__(self, full_shape, window_shape, origin=(0, 0, 0)):
        self.full_shape = tuple(full_shape)
        self.window_shape = tuple(window_shape)
        self.origin = list(origin)

    def get_slices(self):
        sx = slice(self.origin[0], self.origin[0] + self.window_shape[0])
        sy = slice(self.origin[1], self.origin[1] + self.window_shape[1])
        sz = slice(self.origin[2], self.origin[2] + self.window_shape[2])
        return (sx, sy, sz)

    def step(self, dx=0, dy=0, dz=0):
        self.origin[0] = max(0, min(self.full_shape[0] - self.window_shape[0], self.origin[0] + dx))
        self.origin[1] = max(0, min(self.full_shape[1] - self.window_shape[1], self.origin[1] + dy))
        self.origin[2] = max(0, min(self.full_shape[2] - self.window_shape[2], self.origin[2] + dz))

    def leading_edge_slice(self, direction='z', width=1):
        """Return slices corresponding to the leading edge in a given direction.

        direction: 'x'|'y'|'z'
        width: number of grid points to include
        """
        if direction == 'z':
            start = self.origin[2] + self.window_shape[2] - width
            sx = slice(self.origin[0], self.origin[0] + self.window_shape[0])
            sy = slice(self.origin[1], self.origin[1] + self.window_shape[1])
            sz = slice(start, start + width)
            return (sx, sy, sz)
        raise NotImplementedError("leading_edge_slice supports only 'z' currently")
