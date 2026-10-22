#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:56:33 2026

@author: kathrin
"""


from nir_forensics.modeling.split import split_dataset, split_features_target
from sklearn.dummy import DummyClassifier
import pandas as pd

def dummy_classified(dataset: pd.DataFrame, target = "type"):
    """Fits a DummyClassifier model using a prepared dataset.

    Args:
        dataset (pd.DataFrame): Dataset for one instrument, preprocessed
            using build_modeling_dataset().
        target (str) : name of the target column in dataset

    Returns:
        tuple: Training features, training target, test features,
            test target, and the trained classifier.
    """
    dataset = dataset.copy()
    dataset.columns = dataset.columns.astype(str)
    train, test = split_dataset(dataset)
    train_features, train_target, test_features, test_target = split_features_target(train, test, target)
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(train_features, train_target)
    return train_features, train_target, test_features, test_target, dummy_clf
    

