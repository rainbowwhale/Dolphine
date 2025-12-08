import numpy as np


class Grid:
    """Simulation grid and time-step manager.

    Attributes:
        dx, dy, dz: spatial steps in meters
        nx, ny, nz: number of grid points
        dt: time step computed from CFL condition
    """

    def __init__(self, nx, ny, nz, dx=1e-4, dy=None, dz=None, c_max=1540.0, safety=0.5):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.dz = dz if dz is not None else dx
        # CFL-limited dt (scalar acoustic approx): dt <= safety * min(dx,dy,dz) / (c_max*sqrt(3))
        self.c_max = c_max
        self.dt = safety * min(self.dx, self.dy, self.dz) / (self.c_max * np.sqrt(3.0))

    def index_to_world(self, ix, iy, iz):
        """Convert integer grid indices to world coordinates (meters)."""
        x = ix * self.dx
        y = iy * self.dy
        z = iz * self.dz
        return x, y, z

    def world_to_index(self, x, y, z):
        """Convert world coords to nearest grid index (integers)."""
        ix = int(round(x / self.dx))
        iy = int(round(y / self.dy))
        iz = int(round(z / self.dz))
        return ix, iy, iz
