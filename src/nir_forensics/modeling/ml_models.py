#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:56:33 2026

@author: kathrin
"""


from nir_forensics.modeling.split import create_cv_splits, split_features_target
from sklearn.dummy import DummyClassifier
import pandas as pd

def dummy_classified(dataset: pd.DataFrame, target = "type"):
    """Fits DummyClassifier model using a prepared dataset.

    Args:
        dataset (pd.DataFrame): Dataset for one instrument, preprocessed
            using build_modeling_dataset().
        target (str) : name of the target column in dataset

    Returns:
        results: list of dictionaries with predictions and true_labels for each fold
    """
    dataset = dataset.copy()
    train_indices, test_indices = create_cv_splits(dataset)
    results = []
    
    for i, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
    
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]
    
        train_features, train_target, test_features, test_target = (
            split_features_target(train, test, target)
        )
    
        dummy_clf = DummyClassifier(strategy="prior")
        dummy_clf.fit(train_features, train_target)
    
        predictions = dummy_clf.predict(test_features)
        results.append({"fold": i,
                        "predictions" : predictions, 
                        "true_labels" : test_target})
     
    return results
    

#create_cv_splits(dataset: pd.DataFrame