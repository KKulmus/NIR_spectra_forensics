#PlotSpectra

import matplotlib.pyplot as plt
from nir_forensics.dataView import ReadData
import pandas as pd
from nir_forensics.preprocessing import snv, savgol
from pathlib import Path

class PlotSpectra():
    """Visualizes raw NIR sprectra from forensic drug analysis datasets.

        Attributes:
            filename (str): Name of the Excel-File containg spectral data.
            sample_id (str): Unique identifier for the probe (e.g., "K11", "N5"), defaults to None
            rd (ReadData): instance of the ReadData class for file loading

        Example:
            >>> plotter = PlotSpectra("K_ASD.xlsx", "K11")
            >>> axes = plotter.plot_replicates()
            >>> plt.show()
    """

    def __init__(self, filename: str, sample_id: str = None):
        """Initialize PlotSpectra with file and sample identifier.

            Args:
                filename (str): Excel file name (e.g., "K_ASD.xlsx")
                sample_id (str. optional): Sample code to load (e.g., "K11"). Defaults to None
        
        """
        
        self.filename = filename
        self.sample_id = sample_id
        self.rd = ReadData(self.filename,self.sample_id)

    def plot_replicates(self) -> list:
        """Plot all replicates of a given sample.
        
        Creates subplots for each replicate of the sample, displaying
        absorbance vs. wavelength. Handles variable numbers of replicates
        (single to multiple) and adjusts layout accordingly.
        
        Returns:
            list: List of matplotlib.axes.Axes objects. Length equals number
                  of replicates. Always returns a list, even for single
                  replicate (for consistent API).
        
        Raises:
            ValueError: If no samples found for the given sample_id.
        """
        df = self.rd.get_sample()
        n_reps = len(df)
        if n_reps == 0:
            raise ValueError(f"No samples found for {self.sample_id}")
        instrument = self.rd.instrument
        if n_reps == 1:
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(df.wl.astype(float),df.spectral_cols)
            ax.set_title(f"{instrument}, sample {df.iloc[0,0]} Rep 1")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Absorbance")
            ax.grid(True, alpha=0.3)
            
            return [ax]
        if n_reps <= 4:
            fig, axes = plt.subplots(1,n_reps, figsize=(n_reps*4,4))
        else:
            fig, axes = plt.subplots(n_reps,1, figsize=(6,n_reps*2))
        for i in range(n_reps):
            axes[i].plot(df.wl.astype(float),df.spectral_cols)
            axes[i].set_title(f"{instrument}, sample {df.sprectral_cols} Rep {i+1}")
            axes[i].grid(True, alpha=0.3)
        fig.supxlabel("Wavelength (nm)", x=0.5, fontsize=12)
        fig.supylabel("Absorbance", y=0.5, fontsize=12)
        plt.tight_layout()
        return axes


def compare_devices(filenames: list, sample_id: str, apply_snv = False,
                    apply_deriv: int = 0,  # 0 = none, 1 = first derivative, 2 = second
                    window: int = 11) -> plt.Axes:
    """ Takes a sample from different files and compares spectra from instruments graphically
        Therefore for each sample a new instance of the class (PlotData) is needed
        
        Args:
            filenames (list(str)): Names of the Excel-files with the spactral data
            sample_id (str): Sample code to load (e.g., "N11" )
            apply_snv (bool): If True: performs a SNV to normalize the spectra. Defaults to False
            apply_deriv (int): whether the preprocessing should be used with 1 or 2 derivative (defaults to 0)
            window (int): window-length for the savgol filter, dafults to 11
        Returns:
            matplotlib.axes.Axes object with the spectra of one sample from different
            instruments in one plot 
    """
    fig, ax = plt.subplots(figsize = (10,6))
    for name in filenames:
        rd = ReadData(name, sample_id)
        df = rd.get_sample()
        wavelengths = rd.wl.astype(float)
        spectrum = df.sprectral_cols.astype(float)
        if apply_snv:
            spectrum = snv(spectrum)
            
        if apply_deriv != 0:
            spectrum = savgol(spectrum = spectrum, window = window, derivative = apply_deriv)
            
        ax.plot(wavelengths, spectrum, label = rd.instrument)
    ax.set_xlabel("Wavelength (nm)")
    if apply_deriv == 0:
        ax.set_ylabel("Absorbance")
    elif apply_deriv == 1:
        ax.set_ylabel("1st Derivative (Absorbance/nm)")
    else:
        ax.set_ylabel(f"{apply_deriv}nd Derivative (Absorbance/nm²)")
    ax.legend()
    ax.grid(True, alpha = 0.3)
    return ax


#def snv(spectrum: pd.Series) -> pd.Series:
#    """Apply Standart Normal Variarte normalization to a single spectrum

#        SNV centers the spectrum to mean=0 and scales to std=1,
#        removing absolute absorbance differences between measurements.
#        Useful for comparing spectra from 
#    """
#    spectrum = spectrum.astype(float)
#    mu = spectrum.mean()
#    sigma = spectrum.std()
#    if sigma == 0:
#        raise ValueError("Cannot normalize spectrum with zero standart variance.")
#    return (spectrum-mu)/sigma
    

def compare_samples(metadata_filename: str,  path_metadata: str, samples_by_file: dict, apply_snv = False,
                    apply_deriv: int = 0,  # 0 = none, 1 = first derivative, 2 = second
                    window: int = 11) -> plt.Axes:
    """ Takes a sample from different files and compares spectra from different sample_id graphically
        #Therefore for each sample a new instance of the class (PlotData) is needed
        
        Args:
            metadata_filename (str): Name of the metadata-file
            path_metadata (str):
            samples_by_file (dict): Mapping file -> list of sample ids,
            e.g. {"NCD_ASD.xlsx": ["D12", "D13"], "PAM_ASD.xlsx": ["PAM184"]}
            apply_snv (bool): If True: performs a SNV to normalize the spectra. Defaults to False
            apply_deriv (int): whether the preprocessing should be used with 1 or 2 derivative (defaults to 0)
            window (int): window-length for the savgol filter, defults to 11
        Returns:
            matplotlib.axes.Axes object with the spectra of each sample in given list in one plot 
    """
    df_metadata = get_metadata(metadata_filename, path_metadata)
    fig, ax = plt.subplots(figsize = (10,6))
    for filename, ids in samples_by_file.items():
        for sample in ids:
            rd = ReadData.load(filename)
            #df = rd.get_sample(sample)
            wavelengths = rd.wl.astype(float)
            spectrum = rd.get_avg_spectrum(sample)#df[rd.spectral_cols].astype(float)
            if apply_snv:
                spectrum = snv(spectrum)
                
            if apply_deriv != 0:
                spectrum = savgol(spectrum = spectrum, window = window, derivative = apply_deriv)

            matches = df_metadata[df_metadata["code"] == sample]
            label = matches["component"].iloc[0] if len(matches) > 0 else sample    
            ax.plot(wavelengths, spectrum, label = label)
    ax.set_xlabel("Wavelength (nm)")
    if apply_deriv == 0:
        ax.set_ylabel("Absorbance")
    elif apply_deriv == 1:
        ax.set_ylabel("1st Derivative (Absorbance/nm)")
    else:
        ax.set_ylabel(f"{apply_deriv}nd Derivative (Absorbance/nm²)")
    ax.legend()
    ax.grid(True, alpha = 0.3)
    return ax


def get_metadata(metadata_filename: str, path :str):
    """ loads metadata-file as df
        Args:
            metadata_file (str): name of the file with Meta-data
            path (str): path to the file

        Returns:
            pd-DataFrame with metadata
       
    """
    return pd.read_excel(Path(path)/metadata_filename)








        
        
