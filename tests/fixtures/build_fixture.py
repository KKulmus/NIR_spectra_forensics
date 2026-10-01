#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:48:54 2026

@author: kathrin
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

df_excel =  pd.read_excel(save_path)

