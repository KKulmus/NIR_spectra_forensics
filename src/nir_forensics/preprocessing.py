from scipy.signal import savgol_filter
from nir_forensics.dataView import ReadData
import pandas as pd
import numpy as np
from numpy import interp

def snv(spectrum: pd.Series) -> pd.Series:
    """Apply Standart Normal Variarte normalization to a single spectrum

        SNV centers the spectrum to mean=0 and scales to std=1,
        removing absolute absorbance differences between measurements.
        Useful for comparing spectra from different instruments
    """
    spectrum = spectrum.astype(float)
    mu = spectrum.mean()
    sigma = spectrum.std()
    if isinstance(spectrum, pd.DataFrame):
        raise TypeError(f"snv erwartet eine Series (ein Spektrum), bekam DataFrame mit "
                        f"{len(spectrum)} Replikaten. Erst mitteln: .mean(axis=0)")
    if sigma == 0:
        raise ValueError("Cannot normalize spectrum with zero standart variance.")
    return (spectrum-mu)/sigma

def savgol(spectrum: pd.Series, window: int = 11, polyorder: int = 2, derivative: int = 1, apply_snv=False):
    """Apply Savitzky-Golay filter for smoothing and differentiation.
    
    Args:
        spectrum (pd.Series): Input spectrum (already SNV-normalized if desired).
        window (int): Window length (must be odd).
        polyorder (int): Order of polynomial fit.
        derivative (int): Derivative order (0 = smooth only).
        apply_snv (bool): Do SNV before savgol. Defaults to false
    
    Returns:
        pd.Series: Processed spectrum.
    """
    if window%2 == 0:
        raise ValueError("Window length must be odd.")
    if window >= len(spectrum):
        raise ValueError(f"Window {window} must be < spectrum length ({len(spectrum)}).")
    if apply_snv:
        spectrum = snv(spectrum)
    return pd.Series(savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=derivative), index=spectrum.index)

def mean_centering(X: pd.DataFrame) -> pd.Series:
    """Calculates the mean of each column in the Dataframe
        (average spectrum across all samples)

        Args:
            X (pd.DataFrame): Datframe with the NIR values
        Returns:
            pd.Series with the means of each column (aka. wavelength)

    """
    return X.mean(axis=0)

def interpolate_to_grid(spectrum: pd.Series, target_wavelength: np.ndarray) -> pd.Series:
    """Interpolates a given spectrum onto a different target grid, with a different wavelength resolution.

        Args:
            spectrum (pd.Series): spectrum to be interpolated
            target_wavelength (np.array):grid with values for which the data of the spectrum should be interpolated

        Returns:
            pd.Series with the interpolated NIR-Data on the the desired wavelengths
    """
    #np.interp(x_target, x_source, y_source)
    spectrum.index = spectrum.index.astype(float)
    spectrum = spectrum.sort_index()
    result = np.interp(target_wavelength, spectrum.index, spectrum.values)
    return pd.Series(result, index = target_wavelength)
    
    
