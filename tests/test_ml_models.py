#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:14:51 2026

@author: kathrin
"""

import unittest
from nir_forensics.modeling.ml_models import fit_dummy_classifier
from sklearn.dummy import DummyClassifier

from nir_forensics.paths import FIXTURE_DIR
from nir_forensics.modeling.dataset import build_modeling_dataset
from sklearn.utils.validation import check_is_fitted
from nir_forensics.modeling.split import split_dataset

import pandas as pd
import numpy as np



dataset = build_modeling_dataset(instrument = "MicroNIR",
                            metadata_path = FIXTURE_DIR / "metadata.xlsx",
                            avg_dir = FIXTURE_DIR,
                            expected_rows = {"MicroNIR":4})


class TestMlModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load data-files once (using the cache)"""
        cls.df = pd.read_excel(FIXTURE_DIR / "cv_folding_fixture.xlsx") 
    
    def test_dummy_classifier_fit_returns_expected_object(self):
        """dummy_classified returns traines DummyClassifier"""
        features = self.df.loc[:, pd.to_numeric(self.df.columns, errors="coerce").notna()]
        target = self.df["type"]
        returns = fit_dummy_classifier(features, target)
        self.assertTrue(isinstance(returns, DummyClassifier))
        check_is_fitted(returns)
        
    def test_dummy_classifier_predict(self):
        """dummy classifier returns expected number of predictions 
        and exactly one predicted type (prior)"""
        train, test = split_dataset(self.df)
        train_features = train.loc[:, pd.to_numeric(train.columns, errors="coerce").notna()]
        test_features = test.loc[:, pd.to_numeric(test.columns, errors="coerce").notna()]
        train_target = train["type"]
        dummy_cls = fit_dummy_classifier(train_features, train_target)
        prediction = dummy_cls.predict(test_features)
        self.assertEqual(len(test_features), len(prediction))
        self.assertEqual(len(np.unique(prediction)), 1)
               