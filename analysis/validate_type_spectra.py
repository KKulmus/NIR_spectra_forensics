#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:43:00 2026

@author: kathrin

Plots for ASD laboratory spectrometer the type-grouped (target variable) spectra
for each sample with the type-class median shown in the plots.
Outputs a 2x4 subplots figure. 

"""

from nir_forensics.modeling.dataset import build_modeling_dataset
from nir_forensics.paths import PROJECT_ROOT
from nir_forensics.plots import plot_class_spectra
import matplotlib.pyplot as plt

dataset = build_modeling_dataset("asd")
fig = plot_class_spectra(dataset , "ASD", apply_snv = True)
fig.savefig(PROJECT_ROOT / "figs" /"ASD_type_comparison.png", dpi=300,
    bbox_inches="tight" )
plt.close()

