import pandas as pd
from pathlib import Path
import re
import numpy as np

class ReadData():
    """Loads one spectral-file once; provides samples and inventory.
    """
    _CACHE: dict = {} # filename -> ReadData
    def __init__(self, filename: str, path = "dataset/datafiles_raw/"):
        self.filename = filename
        self.instrument = filename.split("_")[1].split(".")[0].lower()
        self.probe = filename.split("_")[0] #Type of the samples, e.g. PAM (Police Amsterdam powdered casework samples)
        self.path = path
        self.df = pd.read_excel(Path(self.path)/self.filename) 
        self.wl = self.df.columns[1:].astype(float).values # catch grid
        # SCiO exports %Reflected light; all other instruments export absorbance units.
        # cf. Kranenburg et al., Forensic Chemistry 30 (2022) 100437, Materials & Methods.
        self.is_reflectance = self.instrument ==  "scio"
        self._prepare_codes()
        

    @classmethod
    def load(cls, filename, path="dataset/datafiles_raw/"):
        if filename not in cls._CACHE:
            cls._CACHE[filename] = cls(filename, path)
        return cls._CACHE[filename]
        
        
    def read_excel_as_df(self):
        return self.df


    def get_sample(self, code: str) -> pd.DataFrame:
        sub = self.df[self.df.iloc[:, 0] == code].copy()
        if len(sub) == 0:
            raise ValueError(f"Sample {code} not in {self.filename}")
        vals = sub[self.spectral_cols].astype(float)
        if (vals <= 0).any().any():
            raise ValueError(f"Non-positive reflectance in {self.filename}, sample {code}: "
                     f"-np.log would fail. Check raw data for this scan.")
        sub[self.spectral_cols] = -np.log(vals) if self.is_reflectance else vals
        return sub

 
    def get_samples_by_prefix(self, prefix):
        prefix_dataframe = self.df[self.df.iloc[:,0].str.startswith(prefix)]
        if len(prefix_dataframe)==0:
            raise ValueError(f'No sample of type {prefix} in chosen file.')
        return prefix_dataframe
            
    def get_available_samples(self) -> list:
        """Returns all sample codes present in this file."""
        return self.df.iloc[:, 0].dropna().unique().tolist()

    
    def _prepare_codes(self):
        """Vectorized: normalize all codes once, store as column 'code_norm'."""
        self.spectral_cols = self.df.columns[1:].tolist()   # INVARIANT position
        codes = self.df.iloc[:, 0].astype(str).str.strip()
        letters = codes.str.extract(r"([A-Za-z]+)")[0]
        digits  = codes.str.extract(r"(\d+)")[0]
        digits  = digits.str.lstrip("0").replace("", "0")
        self.df["code_norm"] = letters + digits
            
    def get_avg_spectrum(self, code: str) -> pd.Series:
        """Replicate-averaged spectrum (Paper: 'Avg_'-Files). Log-transform bereits enthalten."""
        sub = self.get_sample(code)[self.spectral_cols].astype(float)
        return sub.mean(axis=0)
