#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:45:23 2026

@author: kathrin
"""

from nir_forensics.dataView import ReadData
from nir_forensics.paths import PROJECT_ROOT
from nir_forensics.dataCache import load_avg
import pandas as pd

# check negative values wich occur within the micronir
# details in README
RAW_DIR = PROJECT_ROOT / "dataset" / "datafiles_raw"
microNIRList = ['K_MicroNIR.xlsx', 'NCD_MicroNIR.xlsx', 'PAM_MicroNIR.xlsx', 'P_MicroNIR.xlsx', 'T_MicroNIR.xlsx']


liste_code =[]
range_wl = []
range_val = []

for file in microNIRList:
    rd = ReadData.load(file)
    df = load_avg(rd, "Micronir")
    for code in df.index:
        mask = df.loc[code] < 0
        neg = df.loc[code][mask]
        if len(neg)>0:
            liste_code.append(code)
            range_wl.append([neg.index.min(), neg.index.max()])
            range_val.append([round(neg.min(),7), round(neg.max(),7)])
            
dic_out = {"codes": liste_code, "val_range": range_val, "wl_range": range_wl}

df_out = pd.DataFrame(dic_out).T.to_markdown()
print(pd.DataFrame(dic_out))
        
rd = ReadData.load("K_MicroNIR.xlsx")
df_k14 = rd.df.loc[rd.df["code_norm"] == "K14", rd.spectral_cols]  # Zeile mit nur Spektrum

nn = [(wl, val) for wl in df_k14.columns for val in df_k14[wl] if val < 0]


#Table with available samples per instrument (for README)

AVG_DIR = PROJECT_ROOT / "dataset" / "datafiles_avg"

df_meta = pd.read_excel(PROJECT_ROOT / "dataset/metadata.xlsx")
dic = {"code": df_meta["code"], "type":df_meta["type"]}
df_meta = pd.DataFrame(dic)


files = sorted(AVG_DIR.glob("*.parquet"))

instruments = {}
for f in files:
    instrument = f.stem.split("_")[1]
    instruments.setdefault(instrument, []).append(f)

results = {}
dropped = {}
for i in list(instruments.keys()):
    all_codes = []
    for file in instruments[i]:
        df = pd.read_parquet(file)
        all_codes.extend(df.index.tolist())
    df_codes = pd.DataFrame({"code": all_codes})
    joined = pd.merge(df_codes, df_meta, on="code", how="left")  # left oder inner?
    #out = joined.groupby('type').size()
    
    #results[i] = out
    results[i] = joined["type"].value_counts().to_dict()
    dropped[i] = joined["type"].isna().sum()
        
final_df = pd.DataFrame(results).fillna(0).astype(int)  # Fehlt etwas -> 0

final_df["Meta Data"] = df_meta.groupby('type').size()
final_df = final_df.sort_values("Meta Data", ascending = False)  # Types alphabetisch sortieren
final_df.loc["Total"] = final_df.sum()
dropped_df = dropped

print(final_df)
print(dropped_df)        
# liste = []
# for f in files:
#      rd = ReadData.load(f.stem + ".xlsx")
#      df = load_avg(rd, f.stem.split('_')[1])
#      for code in df.index:
#         liste.append(code)


# df2 = pd.read_excel(PROJECT_ROOT / "dataset/metadata.xlsx")
# dic = {"code": df2["code"], "type":df2["type"]}
# df2 = pd.DataFrame(dic)

# dic2 = {"code": liste}
# df = pd.DataFrame(dic2)

# joined = pd.merge(df, df2, on="code", how="left")
# print(joined)

# out = joined.groupby('type').size()

# out = out.sort_values(ascending = False
#                       )
# print(out)


