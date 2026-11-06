#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:09:16 2026

@author: kathrin
"""

from nir_forensics.modeling.ml_models import fit_dummy_classifier
from nir_forensics.modeling.split import create_cv_splits, split_features_target
from nir_forensics.modeling.dataset import build_modeling_dataset
from nir_forensics.modeling.metrics import calculate_metric
import pandas as pd

dataset = build_modeling_dataset("micronir")
train_indices, test_indices = create_cv_splits(dataset)
results = []
for i, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
    trainset = dataset.iloc[train_idx]
    testset = dataset.iloc[test_idx]
    train_features, train_target, test_features, test_target = split_features_target(trainset, testset)
    dummy_cls = fit_dummy_classifier(train_features, train_target)
    predictions = dummy_cls.predict(test_features)
    results.append({"fold": i,
                    "predictions": predictions,
                    "true_labels": test_target})
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