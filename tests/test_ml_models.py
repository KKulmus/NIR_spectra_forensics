#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:14:51 2026

@author: kathrin
"""

import unittest
from nir_forensics.modeling.ml_models import dummy_classified
from sklearn.dummy import DummyClassifier

from nir_forensics.paths import FIXTURE_DIR
from nir_forensics.modeling.dataset import build_modeling_dataset
import pandas as pd
import numpy as np



dataset = build_modeling_dataset(instrument = "MicroNIR",
                            metadata_path = FIXTURE_DIR / "metadata.xlsx",
                            avg_dir = FIXTURE_DIR,
                            expected_rows = {"MicroNIR":4})


class TestMlModels(unittest.TestCase):
    
    def test_dummy_classified_returns_expected_objects(self):
        """dummy_classified returns expected objects"""
        returns = dummy_classified(dataset)
        self.assertEqual(len(returns), 5)
        self.assertTrue(isinstance(returns[0], pd.DataFrame))
        self.assertTrue(isinstance(returns[1], pd.Series))
        self.assertTrue(isinstance(returns[2], pd.DataFrame))
        self.assertTrue(isinstance(returns[3], pd.Series))
        self.assertTrue(isinstance(returns[4], DummyClassifier))
        
    def test_dummy_classified_predicts(self):
        """len(test_fatures) = len(pred), learned_class = pred, only one type is learned"""
        train_features, train_target, test_features, test_target, model = dummy_classified(dataset)
        pred = model.predict(test_features)
        test_rows = test_features.shape[0]
        self.assertEqual(len(pred),test_rows)
        learned_class = model.classes_[0]
        self.assertEqual(learned_class, np.unique(pred)[0])
        self.assertEqual(len(np.unique(pred)), 1)
        
    def test_dummy_classified_features(self):
        """type not in features"""
        train_features, train_target, test_features, test_target, model = dummy_classified(dataset)
        self.assertNotIn("type", train_features.columns.tolist())
        self.assertNotIn("type", test_features.columns.tolist())
        self.assertNotIn("group", train_features.columns.tolist())
        self.assertNotIn("group", test_features.columns.tolist())
        
    def test_dummy_classified_target(self):
        """Target == type"""
        train_features, train_target, test_features, test_target, model = dummy_classified(dataset)
        self.assertEqual(train_target.name, "type")
        self.assertEqual(test_target.name, "type")
        
    def test_dummy_classified_train_test_split(self):
        """T1 and P1 on same site"""
        train_features, train_target, test_features, test_target, model = dummy_classified(dataset)
        train_codes = set(train_features["code"])
        test_codes = set(test_features["code"])
        self.assertTrue(
            ("T1" in train_codes and "P1" in train_codes and "T1" not in test_codes and "P1" not in test_codes)
            or
            ("T1" in test_codes and "P1" in test_codes and "T1" not in train_codes and "P1" not in train_codes))
        
    
    
      