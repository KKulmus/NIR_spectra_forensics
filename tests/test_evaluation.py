#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:56:39 2026

@author: kathrin
"""

import unittest

from nir_forensics.paths import FIXTURE_DIR, PROJECT_ROOT
from nir_forensics.modeling.evaluation import pooling_folds 
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.models import prepare_dataset, make_predictions
import pandas as pd
import numpy as np


class TestEvaluation(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):
        """Load  and prepare data-files once (using the fixture)"""
        df = pd.read_excel(FIXTURE_DIR / "cv_folding_fixture.xlsx")
        cls.dataset, cls.train_indices, cls.test_indices = prepare_dataset(dataset = df)
        cls.results = make_predictions(cls.dataset, cls.train_indices, cls.test_indices, "dummy")
        
    def test_pooling_results_concatenated_list(self):
        """Test whether true_labels and predictions are one-dimensional np.arrays
            and lenghts are as expected
        """
        true_labels, predictions = pooling_folds(self.results)
        self.assertIsInstance(true_labels, np.ndarray)
        self.assertEqual(true_labels.ndim, 1)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.ndim, 1)
        
        self.assertEqual(len(predictions), len(self.dataset))
        self.assertEqual(len(true_labels), len(self.dataset))
        
    def test_pooling_keeps_labels_and_predictions(self):
        """Test whether the labels are not changed through pooling"""
        true_fold = [["foo", "bar", "foo"], ["bar", "bar", "bar"]]
        predictions = [["foo", "foo", "foo"],["bar", "foo", "bar"]]
        results = [{"fold": i , 
                    "true_labels": true_fold[i],
                    "predictions" : predictions[i]} for i in range(len(true_fold))]
        true_labels, predictions = pooling_folds(results)
        np.testing.assert_array_equal(true_labels, ["foo", "bar", "foo", "bar", "bar", "bar"])
        np.testing.assert_array_equal(predictions, ["foo", "foo", "foo","bar", "foo", "bar"] )
        
        
        
        