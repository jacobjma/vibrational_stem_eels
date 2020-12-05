import numpy as np
from ase import units


def cosine_squared_window(n_points):
    points = np.arange(n_points)
    window = np.cos(np.pi * points / (n_points - 1) / 2) ** 2
    return window


def _single_fft_autocorrelation(data):
    data = (data - np.mean(data)) / np.std(data)
    n_points = data.shape[0]
    fft_forward = np.fft.fft(data, n=2 * n_points)
    fft_autocorr = fft_forward * np.conjugate(fft_forward)
    fft_backward = np.fft.ifft(fft_autocorr)[:n_points] / n_points
    return np.real(fft_backward)


def fft_autocorrelation(data):
    orig_shape = data.shape
    reshaped_data = data.reshape((orig_shape[0], -1))
    autocorrelations = np.zeros((reshaped_data.shape[1], data.shape[0]))
    for i in range(reshaped_data.shape[1]):
        autocorrelations[i, :] = _single_fft_autocorrelation(reshaped_data[:, i])
    return autocorrelations.reshape((*orig_shape[1:], -1))


def velocity_autocorrelation(velocities):
    autocorrelation = fft_autocorrelation(velocities)
    autocorrelation = np.sum(autocorrelation, axis=1)
    autocorrelation = np.mean(autocorrelation, axis=0)
    return autocorrelation


def compute_spectra(data, timestep, resolution=None, frequency_units='THz'):
    if resolution:
        data = data[: resolution]

    orig_shape = data.shape[0]
    data *= cosine_squared_window(orig_shape)

    data_padded = np.zeros(4 * orig_shape)
    data_padded[:orig_shape] = data

    data_mirrored = np.hstack((np.flipud(data_padded), data_padded))

    n_fourier = 8 * orig_shape
    intensities = np.abs(timestep * np.fft.fft(data_mirrored, n=n_fourier)[: n_fourier // 2])
    frequencies = np.arange(n_fourier // 2) / (n_fourier * timestep)

    if frequency_units:
        if frequency_units.lower() == 'thz':
            frequencies *= 1e3 * units.fs
        else:
            raise ValueError()

    return frequencies, intensities


def pdos_spectrum(velocities, timestep, resolution=None):
    data = velocity_autocorrelation(velocities)
    frequencies, intensities = compute_spectra(data, timestep=timestep, resolution=resolution)
    return frequencies, intensities
