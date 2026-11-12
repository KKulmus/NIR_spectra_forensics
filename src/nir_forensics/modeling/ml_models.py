#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:56:33 2026

@author: kathrin
"""


from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def fit_dummy_classifier(train_features: pd.DataFrame,
                         train_target: pd.Series):
    """Fits a DummyClassifier using the provided training data.

    Args:
        train_features (pd.DataFrame): Training features.
        train_target (pd.Series): Training target labels.

    Returns:
        DummyClassifier: Fitted dummy classifier.
    """
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(train_features, train_target)

    return dummy_clf

def fit_random_forest_classifier(train_features: pd.DataFrame,
                         train_target: pd.Series,
                         n_estimators = 100, random_state = 42, class_weight = None):
    """Fits a RandomForestClassifier using the provided training data.

    Args:
        train_features (pd.DataFrame): Training features.
        train_target (pd.Series): Training target labels.

    Returns:
        RandomForestClassifier: Fitted random forest classifier.
    """
    rfc_clf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, class_weight = class_weight)
    rfc_clf.fit(train_features, train_target)

    return rfc_clf

