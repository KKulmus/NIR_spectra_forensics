#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:09:52 2026

@author: kathrin
"""

import pandas as pd
from nir_forensics.paths import PROJECT_ROOT
import warnings


AVG_DIR = PROJECT_ROOT / "dataset" / "datafiles_avg"

EXPECTED_ROWS = {"MicroNIR": 429, "ASD" : 389 , "NeoSpectra":429  ,"Scio":417 , "NIRONE":341 }

def build_modeling_dataset(instrument,
                           metadata_path = None,
                           avg_dir = None,
                           expected_rows = None) -> pd.DataFrame:
    """
    Builds a Dataframe for one instrument, where all sample-families, e.g., P, T, PAM... are 
    merged. Columns are: code, wavelengths, type; Index is range-index
    type is taken from metadata.xlsx using merge on df-instrument-sample-family
    ARGS: 
        instrument (string): instrument of interest, e.g., Scio
        metadata_path (path): path to *.xlsx-file with metadata
        avg_dir (path): path to folder with *avg.parquet-files
        expected_rows (dict): dictionary {string(instrument) : int(number of samples)
        
    RETURNS:
        pd.Dataframe
    
    Notes:
        in merge: validate = "1:1": each sample-code is unique in metadata and in 
        the instrument-data (we use the averaged data)
        Empty/NaN types are tolerated by design; verified for all instruments 
        via class distribution check
    """
    if metadata_path is None:
        metadata_path = PROJECT_ROOT / "dataset/metadata.xlsx"
    if avg_dir is None:
        avg_dir = AVG_DIR
    if expected_rows is None:
        expected_rows = EXPECTED_ROWS
    instrument = instrument.upper()
    mapping_dict = {"MICRONIR" : "MicroNIR", "ASD":"ASD", "NEOSPECTRA": "NeoSpectra", "SCIO": "Scio", "NIRONE":"NIRONE"}
    if instrument not in list(mapping_dict.keys()):
        raise ValueError(f"Unknown {instrument} Perhaps typo?")
    else:
        instrument = mapping_dict[instrument]
    df_meta = pd.read_excel(metadata_path)
    files = sorted(avg_dir.glob(f"*_{instrument}_avg.parquet"))
    if len(files) == 0:
        raise ValueError("Parquet files are not cached. Use build_all(instrument) first.")
    dataframes = []
    for f in files:
        df = pd.read_parquet(f)
        dataframes.extend([df])
    dataframes = pd.concat(dataframes)
    joined = pd.merge(df_meta,dataframes,  right_index = True,left_on="code", how="inner", validate = "1:1")
    results = dataframes.index.to_list()
    missing = set(results) - set(joined["code"])
    if missing:
        warnings.warn(f"Warning: {len(missing)} spectra without metadata: {missing}")
    if len(joined) !=expected_rows[instrument]:
        raise ValueError(f"Merging failed: for {instrument}, {expected_rows[instrument]} are expected, joined has {len(joined)} rows")
    df = pd.DataFrame(joined)
    return df