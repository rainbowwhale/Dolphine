import numpy as np


class Medium:
    """Host-resident medium property maps (memory-optimized view slicing).

    Holds `rho`, `c`, `alpha` as numpy arrays in host RAM. dtype can be specified by user
    (e.g., np.float16 for reduced memory).
    """

    def __init__(self, shape, dtype=np.float32, default_c=1540.0, default_rho=1000.0, default_alpha=0.0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.rho = np.full(self.shape, default_rho, dtype=dtype)
        self.c = np.full(self.shape, default_c, dtype=dtype)
        self.alpha = np.full(self.shape, default_alpha, dtype=dtype)

    def set_region(self, slices, c=None, rho=None, alpha=None):
        """Set region values using numpy slices. Example: slices=(slice(10,20), slice(None), slice(30,40))"""
        if c is not None:
            self.c[slices] = c
        if rho is not None:
            self.rho[slices] = rho
        if alpha is not None:
            self.alpha[slices] = alpha

    def slice_roi(self, slices, copy=False):
        """Return views (or copies) of medium maps for a region of interest.

        This method avoids duplicating the entire domain for solver windows.
        """
        if copy:
            return self.rho[slices].copy(), self.c[slices].copy(), self.alpha[slices].copy()
        return self.rho[slices], self.c[slices], self.alpha[slices]
