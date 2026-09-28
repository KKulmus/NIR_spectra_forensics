#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:41:44 2026

@author: kathrin

Is SCiO T150 background-spectum or tablet-spectrum? 
T150 is not present in metadata, we need to check, what kind of spectrum it is.
Expectation:
    If background: no/less not very distinct features, low variance
    If tablet: distinct features (peaks), high variance, no flat spektrum with
               few features and/or very low peaks      
"""


from nir_forensics.dataView import ReadData
from nir_forensics.preprocessing import savgol, interpolate_to_grid
from nir_forensics.plots import compare_samples, get_metadata
from nir_forensics.paths import PROJECT_ROOT
import pandas as pd
from pathlib import Path


rd_t = ReadData.load("T_Scio.xlsx")
rd_p = ReadData.load("PAM_Scio.xlsx")
rd_asd  = ReadData.load("T_ASD.xlsx")

scio_wl = rd_t.wl

# 1. Check whether T150 is present in any other T_... file
for inst in ["ASD", "MicroNIR", "NeoSpectra", "NIRONE"]:
    fname = f"T_{inst}.xlsx"
    if (Path("dataset/datafiles_raw") / fname).exists():
        avail = ReadData.load(fname).get_available_samples()
        print(f"{inst}: 'T150' present? {'T150' in avail}")
    else:
        print(f"T150 not present within {inst}")

# 1. Test T150 against ASD-Tablet-spectra
def prep_scio(code):
    spec = rd_t.get_avg_spectrum(code)    
    return savgol(spec, window=11, derivative=1, apply_snv = True)

def prep_asd(code):
    spec = rd_asd.get_avg_spectrum(code)   
    on_scio = interpolate_to_grid(spec, scio_wl)
    return savgol(on_scio, window=11, derivative=1, apply_snv = True)

#Baseline: all tablets which are present in both files
common = sorted(set(rd_t.get_available_samples()) & set(rd_asd.get_available_samples()))

#Expectation: r = 1 for identical tablets
baseline = {c: prep_scio(c).corr(prep_asd(c)) for c in common}
baseline_series = pd.Series(baseline)
print(baseline_series.describe())

# rank T150 against all ASD-tablets ranken, not only against common, 
#     since T150 exists only within SCiO!
t150 = prep_scio("T150")
candidates = {c: t150.corr(prep_asd(c)) for c in rd_asd.get_available_samples()}
ranking = pd.Series(candidates).sort_values(ascending=False)
print(ranking.head(10))
print(f"\nT120: r = {candidates['T120']:.4f}, Rank {list(ranking.index).index('T120')+1} of {len(ranking)}")

## 2. Calculate variance, test against background
variances = {}
for code in rd_t.df["code_norm"].unique():
    spec = rd_t.get_avg_spectrum(code)      # Mean of the replicetae
    savgol_spec = savgol(spec, window = 11, derivative=1, apply_snv=True)
    variances[code] = savgol_spec.var()         # a number

# 2a. PAM000 for comparison/controll: is background! 
pam000_spec = rd_p.get_avg_spectrum("PAM000")
pam000_savgol = savgol(pam000_spec,window = 11, derivative=1, apply_snv=True)     
pam000_var = pam000_savgol.var()

# 2b.. sort t150 variances, return lowest values
sorted_vars = sorted(variances.items(), key=lambda x: x[1])  # low to high
print("Lowest variance (bottom 5):", sorted_vars[:5])
print(f"\nT150 Variance: {variances['T150']}")
print(f"PAM000 Variance: {pam000_var}")

"""
Return: 
Lowest variance (bottom 5): [('T30', 6.097514958440014e-05), 
                             ('T33', 6.912469518904614e-05), 
                             ('T39', 7.138727502451966e-05), 
                             ('T150', 7.28170233297252e-05), 
                             ('T29', 8.80782976335948e-05)]

T150 Variance: 7.28170233297252e-05
PAM000 Variance: 1.588477715123846e-05

We decide to exclude the T150-data: it is a real measure, WITHOUT a metadata-entry,
                                    we cannot reconstruct the origin, but we don't
                                    delete it silencely'
"""

# 3. Graphical comparison T150 vs background PAM000

ax = compare_samples(metadata_filename = "metadata.xlsx",  
                            path_metadata = PROJECT_ROOT / "dataset" , 
                            samples_by_file = {"T_Scio.xlsx": ["T150"],  "PAM_Scio.xlsx": ["PAM000"]}, 
                            apply_snv = True,
                            apply_deriv = 1,  # 0 = none, 1 = first derivative, 2 = second
                            window = 11)

ax.set_title("T150 (unknown) shows genuine spectral structure unlike background reference PAM000 (SCiO, 1st derivative after SNV)")   
#ax.figure.savefig(PROJECT_ROOT / "figs" / "t150_diagnosis.png", dpi=300, bbox_inches="tight") 

# 4. additional: we see structures in PAM000 between 850-900nm. The paper dosn't state whether the 
#    background is subtracted. Hence we check PAM000 against the 5 Scio_T spectra with the lowest variance:
#    do they show the same structure within in wavelength range?

ax_background_check = compare_samples(metadata_filename = "metadata.xlsx",  
                            path_metadata = PROJECT_ROOT / "dataset" , 
                            samples_by_file = {"T_Scio.xlsx": ["T29","T150","T39","T33","T30"],  "PAM_Scio.xlsx": ["PAM000"]}, 
                            apply_snv = True,
                            apply_deriv = 1,  # 0 = none, 1 = first derivative, 2 = second
                            window = 11)   
ax_background_check.set_title("Comparison of background (PAM000) with real tablet-spectra: check for common artefacts")
    
ax_background_check.figure.savefig(PROJECT_ROOT / "figs" / "background_diagnosis.png", dpi=300, bbox_inches="tight")     

# correlation of average of all first derivatives of T-samples and PAM000
ax_background_check_raw = compare_samples(metadata_filename = "metadata.xlsx",  
                            path_metadata = PROJECT_ROOT / "dataset" , 
                            samples_by_file = {"T_Scio.xlsx": ["T29","T150","T39","T33","T30"],  "PAM_Scio.xlsx": ["PAM000"]}, 
                            apply_snv = False,
                            apply_deriv = 0,  # 0 = none, 1 = first derivative, 2 = second
                            window = 11)   
ax_background_check_raw.set_title("Comparison of background (PAM000) with real tablet-spectra: check for common artefacts - RAW data")

sigma = []
for code in ["T119", "T118", "T34", "T29","T133", "T129"]:
    spec = rd_t.get_avg_spectrum(code)      # Mean of the replicetae
    sigma.append(spec.std())

sigma.append(pam000_spec.std())    

print(f"Sigma: 2-CB, MDMA, Amphetamin-Type (each twice) and PAM000 {sigma}")


rd_pt = ReadData.load("P_Scio.xlsx")
rd_ncd = ReadData.load("NCD_Scio.xlsx")
rd_k = ReadData.load("K_Scio.xlsx")

meta = get_metadata("metadata.xlsx", PROJECT_ROOT / "dataset")     # du hast die Funktion bereits in plots.py!
meta_indexed = meta.set_index(meta["code"].str.strip().str.upper())  # Achtung: Code-Format!
class_map = meta_indexed["type"]
pam_proc = savgol(rd_p.get_avg_spectrum("PAM000"), window=11, derivative=1, apply_snv=True) 
t150_proc = savgol(rd_t.get_avg_spectrum("T150"), window=11, derivative=1, apply_snv=True) 

rd_list = [rd_t, rd_pt, rd_p, rd_ncd, rd_k]
for rd in rd_list:
    avg_rows = {}
    if rd == rd_p:
        for code in rd.df.iloc[:,0].unique():
            avg_rows[code] = rd.get_avg_spectrum(code)
            avg_rows[code] = savgol(avg_rows[code], window=11, derivative=1, apply_snv=True )
    else:
        for code in rd.df["code_norm"].unique():
            avg_rows[code] = rd.get_avg_spectrum(code)
            avg_rows[code] = savgol(avg_rows[code], window=11, derivative=1, apply_snv=True )
    avg_df = pd.DataFrame(avg_rows).T    # Zeilen = Codes, Spalten = Wellenlängen


    merged = avg_df.join(class_map)
    centroid = merged.groupby("type").mean()      # Zeilen = Klassen, Spalten = Wellenlängen
    
    # Konsistenz: die centroid-Zeilen müssen mit PAM000 dieselbe Verarbeitung durchlaufen haben —
    # sieh Baustein 2: prozessiere die avg-Spektren VOR dem groupby, nicht danach!
    scores_pam = centroid.apply(lambda row: row.corr(pam_proc), axis=1).sort_values(ascending=False)
    scores_t150 = centroid.apply(lambda row: row.corr(t150_proc), axis=1).sort_values(ascending=False)
    
    print(f"Scores PAM000: {scores_pam}")
    print(f"Scores T150: {scores_t150}")

















