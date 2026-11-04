#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:09:16 2026

@author: kathrin
"""

from nir_forensics.modeling.ml_models import dummy_classified
from nir_forensics.modeling.dataset import build_modeling_dataset
from nir_forensics.modeling.metrics import calculate_metric
import pandas as pd

dataset = build_modeling_dataset("micronir")
results = dummy_classified(dataset)
f1_list = []
for i in range(len(results)):
    macro = calculate_metric(results[i]["true_labels"], results[i]["predictions"], "macro")
    weighted = calculate_metric(results[i]["true_labels"], results[i]["predictions"], "weighted")
    f1_list.append({"fold": i,
                    "Macro-F1": macro,
                    "Weighted-F1": weighted})
for i, result in enumerate(results):
    print(i, result["predictions"][0])

df_f1 = pd.DataFrame(f1_list).set_index("fold")

mean_std = df_f1.agg(["mean", "std"])