#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:18:09 2026

@author: kathrin

investigation of type-distribution in the dataset
"""

from nir_forensics.modeling.dataset import build_modeling_dataset
from nir_forensics.paths import PROJECT_ROOT
import pandas as pd


def get_type_distribution(instrument : str):
    """Returns the absolute number of substances of distinct 
    types in a instrument
    
    Args:
        instrument (string) : Name of the instrument in question
    Returns:
        table with type and accoding number in the dataset
    """
    instrument_data = build_modeling_dataset(instrument)
    return instrument_data["type"].value_counts()

instrument_list = ["Scio", "Neospectra", "NirOne", "Micronir"]
#for instrument in instrument_list:
#    print(instrument)
#    print(get_type_distribution(instrument))

def get_group_distribution(instrument : str):
    """returns the number of unique groups within one type 
    w.r.t. a single instrument
    
    Args:
        instrument (string) : Name of the instrument in question
    Returns:
        table with type an according number of unique groups
    """
    instrument_data = build_modeling_dataset(instrument)
    return instrument_data[["type", "group"]].groupby("type").nunique()

#for instrument in instrument_list:
#    print(instrument)
#    print(get_group_distribution(instrument))

def get_t_p_type_distribution():
    """
    Returns:
        type distribution within the T/P-set gained from metadata
        interessting because T and P build one group in the test-train split
        """
    
    metadata_path = PROJECT_ROOT / "dataset/metadata.xlsx"    
    df_meta = pd.read_excel(metadata_path)

    df_t_p = df_meta.loc[df_meta["set"].isin(["T", "P"])]
    return df_t_p[["nr","type"]].groupby("type").nunique()


