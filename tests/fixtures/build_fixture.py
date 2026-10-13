#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:48:54 2026

@author: kathrin

# --- Fixture code design (dataview)-----------------------------------------------------
# Each code in this fixture exists to cover a specific test case in
# test_data_view.py (TestLoaderFixture). The fixture is NOT a subset of the
r# real dataset -- it is test coverage design:
#
#   Code             Rows   Test purpose
#   T0, T10, T150    1 ea.  plain codes survive normalization unchanged
#                          (increasing digit counts)
#   T101             2      replicate collapse (get_available_samples) and
#                          replicate averaging (get_avg_spectrum)
#   PAM001           1      leading zeros are stripped (PAM001 -> PAM1)
#   PAM000           1      zero-only digits must not vanish (PAM000 -> PAM0)
#
# A PAM code in a "T_" file is intentional: _prepare_codes() ignores the
# file-prefix convention, so both normalization edge cases ride along in a
# single fixture. Values are synthetic (seed 42), 0.65 < R < 0.75, safe for
# the -log reflectance transform.
# -----------------------------------------------------------------------------


"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
wl = np.linspace(740, 760, 21)
spectral_data = 0.05*np.sin(np.random.randn(7,21)) + 0.7

#T0,T10, T101,T101,T150,PAM001, PAM000

wl = [str(int(i)) for i in wl]
df = pd.DataFrame(spectral_data, columns = wl)
df.insert(0, 'axis: wavelength (nm) / data: raw spectral data (-)', ['T0','T10','T101','T101','T150', 'PAM001', 'PAM000'])
print(df.head())


save_path = Path(__file__).parent / "T_Scio_fixture.xlsx"
df.to_excel(save_path, index=False)
print(f"Fixture written to {save_path}")

# --- Fixture code design (dataset.py) -----------------------------------------------------
# Each code in this fixture exists to cover a specific test case in
# test_dataset.py (TestLoaderFixture). The fixture is NOT a subset of the
# real dataset -- it is test coverage design:
#
#Parquet:
#   Code                Test purpose
#   T1, T2, P1, PAM7    plain codes: joining on metadata is fine 
#   X12                 code not in metadata: missing-path
#   
#Excel:
#   metadata.xlsx : 
#                       Test Purpose                                
#   T1, T2, PAM7        joining works as it should, expected rows-check
#   with type-values    
#   P1                  test validate 1:1   
#   without type
#
# -----------------------------------------------------------------------------




wl5 = np.linspace(740, 750, 5)
spectral_data5 = 0.05*np.sin(np.random.randn(5,5)) + 0.7

#T1, T2, P1, PAM7, X12

wl5 = [float(i) for i in wl5]
df = pd.DataFrame(spectral_data5, columns = wl5)
df.insert(0, 'code', ['T1', 'T2', 'P1', 'PAM7', 'X12'])
df = df.set_index("code")
print(df.head())


save_path_parquet = Path(__file__).parent / "X_MicroNIR_avg.parquet"
df.to_parquet(save_path_parquet)
print(f"Fixture written to {save_path_parquet}")

metadata_dict = {'code' : ['T1', 'T2', 'P1', 'PAM7'],        
                 'type' : ['other', 'cocaine', '','cocaine']}

df_meta = pd.DataFrame(metadata_dict)
save_path_metadata = Path(__file__).parent / "metadata.xlsx"
df_meta.to_excel(save_path_metadata, index=False)
print(f"Meta-fixture written to {save_path_metadata}")



