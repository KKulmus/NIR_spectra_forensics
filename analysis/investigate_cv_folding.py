#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:04:23 2026

@author: kathrin
"""

from nir_forensics.modeling.split import create_cv_splits
from nir_forensics.modeling.dataset import build_modeling_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
INSTRUMENTS = ["scio", "nirone", "neospectra", "micronir"]


def fold_overview(instrument):
    df = build_modeling_dataset(instrument)
    train_indices, test_indices = create_cv_splits(df)
   
    train = []
    test = []
    for i in range(len(train_indices)):
        train.append(df.iloc[train_indices[i]]["type"].value_counts(normalize = True))
        test.append(df.iloc[test_indices[i]]["type"].value_counts(normalize = True))
    class_names = ['cocaine', 'MDMA', 'no_controlled_substance', 'amphetamine_type',
           '2C-B_phenethylamine-psychedelics', 'cathinone', 'other', 'ketamine']
    train = pd.DataFrame(train).reindex(columns=class_names, fill_value=0)
    test = pd.DataFrame(test).reindex(columns=class_names, fill_value=0)
    train["Fold"] = range(len(train_indices))
    train["Set"] = "Train"
    test["Fold"] = range(len(train_indices))
    test["Set"] = "Test"
    columns = ['Fold', 'Set', 'cocaine', 'MDMA', 'no_controlled_substance', 'amphetamine_type',
           '2C-B_phenethylamine-psychedelics', 'cathinone', 'other', 'ketamine']
    train = train.reindex(columns = columns, fill_value = 0).fillna(0)
    test = test.reindex(columns = columns, fill_value = 0).fillna(0)
    out = pd.concat([train,test], ignore_index = True)
    out["Set"] = pd.Categorical(out["Set"], categories=["Train", "Test"], ordered=True)
    out = out.sort_values(by=["Fold", "Set"])
    return out

def difference_test_total_dataset(instrument : str):
    df = fold_overview(instrument)
    df_total = build_modeling_dataset(instrument)
    df_total = df_total["type"].value_counts(normalize=True)
    diff = df[df["Set"] == "Test"].drop(["Set", "Fold"], axis = 1)-df_total
    return round(diff*100,1)


differences = []

for instrument in INSTRUMENTS:
    diff = difference_test_total_dataset(instrument)
    diff["Instrument"] = instrument
    diff["Fold"] = range(len(diff))
    differences.append(diff)

differences = pd.concat(differences, ignore_index=True)
differences = differences.set_index(["Instrument", "Fold"])
plt.figure(figsize = (10,8))

ax = sns.heatmap(differences, center=0
            , vmin = -7, vmax = 7
            , cmap = 'coolwarm'
            , square = True
            , annot = True)
ax.hlines([5,10,15], 0, 8, colors = "black")
plt.show()

        
    
