import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def envelope(signal):
    """Return signal envelope using analytic signal (Hilbert transform)."""
    analytic = hilbert(signal)
    env = np.abs(analytic)
    return env


def log_compress(img, dynamic_range_db=60.0, eps=1e-12):
    maxv = img.max()
    db = 20.0 * np.log10(img / (maxv + eps) + eps)
    return np.clip(db, -dynamic_range_db, 0.0)


def save_mip(intensity, out_path, extent=None, cmap='gray'):
    plt.figure(figsize=(6, 8))
    plt.imshow(intensity, cmap=cmap, origin='upper', extent=extent)
    plt.colorbar(label='dB')
    plt.xlabel('Lateral (mm)')
    plt.ylabel('Depth (mm)')
    plt.title('Pressure Intensity (MIP)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_line_profile(profile, out_path):
    plt.figure(figsize=(6, 3))
    plt.plot(profile)
    plt.xlabel('Index')
    plt.ylabel('Amplitude (a.u.)')
    plt.title('Beam Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
