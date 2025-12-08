import numpy as np


def tone_burst(center_freq, sampling_rate, n_cycles=3, amplitude=1.0):
    t_cycle = 1.0 / center_freq
    t = np.arange(0, n_cycles * t_cycle, 1.0 / sampling_rate)
    window = np.hanning(len(t))
    signal = amplitude * window * np.sin(2 * np.pi * center_freq * t)
    return signal.astype(np.float32), sampling_rate


def gaussian_pulse(center_freq, sampling_rate, bandwidth=0.6, amplitude=1.0):
    # Gaussian envelope in frequency domain converted to time domain approx
    t_max = 2.0 / center_freq
    t = np.arange(-t_max, t_max, 1.0 / sampling_rate)
    sigma = (1.0 / (center_freq * bandwidth))
    pulse = amplitude * np.exp(-t ** 2 / (2.0 * sigma ** 2)) * np.cos(2 * np.pi * center_freq * t)
    return pulse.astype(np.float32), sampling_rate
