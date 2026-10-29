#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:01:28 2026

@author: kathrin
"""

import pandas as pd
import warnings

def get_codes(df_joined: pd.DataFrame) -> pd.DataFrame:
    """Checks a dataframe with columns "code", "set" and "nr",
        checks codes, excludes invalid rows and calls make_groups()
    Rules: 
        if code == set+nr and calls make_groups 
        if code != set+nr exclude invalid rows and issue a warning
    Args:
        df_joined (pd.DataFrame): Dataframe with columns "code", "set" and "nr"
                                  build e.g., from dataset.build_modeling_dataset
    Returns:
        pd.DataFrame with new column "group"
    """
    
    code = df_joined["code"]
    nr = df_joined["nr"].astype(str)
    sset = df_joined["set"]
    mask = code == sset+nr
         
    if (~mask).any():
        invalid_codes = df_joined[~mask]["code"].to_list()
        warnings.warn(f"Code != set + nr for {invalid_codes}. "
                      "These spectra will be excluded. Manual inspection is required "
                      "before using them.", UserWarning)
    df = df_joined[mask].copy()
        
    return make_groups(df)
        

def make_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns group labels based on set and nr,
        makes new column "group"
    Rules:
        if "set" == "P" or "set" == "T", "group" = "E" + nr, 
        else: "group" = "code"
        Example: T7 -> E7, P7 -> E7, K13 -> K13
    Args:
        df (pd.DataFrame): Dataframe with columns "code", "set" and "nr"
                                  build e.g., from get_codes
    Returns:
        pd.DataFrame with new column "group" 
    """
   
    mask = df["set"].isin(["P","T"])
    df["group"] = ("E"+df["nr"].astype(str)).where(mask, df["code"]) 
    return  df
   
        
    
    
    
    
    
    
  