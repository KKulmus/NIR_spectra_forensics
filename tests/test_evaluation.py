#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:56:39 2026

@author: kathrin
"""

import unittest

import pandas as pd
import numpy as np

from nir_forensics.paths import FIXTURE_DIR, PROJECT_ROOT
from nir_forensics.modeling.folds_utils import pooling_folds
from nir_forensics.modeling.ml_models import fit_dummy_classifier
from nir_forensics.modeling.evaluation import aggregate_fold_metrics, create_model_comparison




import sys
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.models import prepare_dataset, make_predictions



class TestEvaluation(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):
        """Load  and prepare data-files once (using the fixture)"""
        df = pd.read_excel(FIXTURE_DIR / "cv_folding_fixture.xlsx")
        cls.dataset, cls.train_indices, cls.test_indices = prepare_dataset(dataset = df)
        cls.results = make_predictions(cls.dataset, cls.train_indices, cls.test_indices, "dummy", fit_dummy_classifier)
        
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
        
    
 
    def setUp(self):
        """Create dummy results with two folds."""

        self.instrument = "test_instrument"

        self.aggregate_results = [
            {
                "fold": 0,
                "true_labels": [
                    "MDMA", "MDMA", "cocaine", "cocaine"
                ],
                "predictions": [
                    "MDMA", "cocaine", "cocaine", "cocaine"
                ]
            },
            {
                "fold": 1,
                "true_labels": [
                    "MDMA", "cocaine", "cocaine", "no_controlled_substance"
                ],
                "predictions": [
                    "MDMA", "cocaine", "no_controlled_substance",
                    "no_controlled_substance"
                ]
            }
        ]

        self.df_cm_folds, self.df_cm_stats, \
            self.df_fms_stats, self.df_fms_folds = (
                aggregate_fold_metrics(
                    self.aggregate_results, self.instrument
                )
            )

    def test_return_format(self):
        """Test whether the function returns a tuple of four DataFrames."""

        output = aggregate_fold_metrics(
            self.aggregate_results, self.instrument
        )

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)

        for dataframe in output:
            self.assertIsInstance(dataframe, pd.DataFrame)

    def test_columns(self):
        """Test whether all expected columns are present."""

        expected_columns = [
            ["instrument", "fold", "type", "recall", "support"],
            ["type",
                ("recall", "mean"),
                ("recall", "std"),
                ("support", "mean"),
                ("support", "std")],
            ["Macro-F1", "Weighted-F1", "instrument"],
            ["fold", "Macro-F1", "Weighted-F1", "instrument"]
        ]

        dataframes = [
            self.df_cm_folds,
            self.df_cm_stats,
            self.df_fms_stats,
            self.df_fms_folds
        ]

        for dataframe, expected in zip(dataframes, expected_columns):
            for column in expected:
                self.assertIn(column, dataframe.columns)

    def test_instrument(self):
        """Test whether the instrument name is correctly assigned."""

        for dataframe in [
            self.df_cm_folds,
            self.df_cm_stats,
            self.df_fms_stats,
            self.df_fms_folds
        ]:
            self.assertTrue(
                (dataframe["instrument"] == self.instrument).all()
            )

    def test_recall_and_support(self):
        """Test class-wise Recall and Support for each fold."""

        mdma_fold0 = self.df_cm_folds[
            (self.df_cm_folds["fold"] == 0) &
            (self.df_cm_folds["type"] == "MDMA")
        ].iloc[0]

        self.assertEqual(mdma_fold0["support"], 2)
        self.assertEqual(mdma_fold0["recall"], 0.5)

        cocaine_fold0 = self.df_cm_folds[
            (self.df_cm_folds["fold"] == 0) &
            (self.df_cm_folds["type"] == "cocaine")
        ].iloc[0]

        self.assertEqual(cocaine_fold0["support"], 2)
        self.assertEqual(cocaine_fold0["recall"], 1.0)

        no_controlled_fold1 = self.df_cm_folds[
            (self.df_cm_folds["fold"] == 1) &
            (self.df_cm_folds["type"] == "no_controlled_substance")
        ].iloc[0]

        self.assertEqual(no_controlled_fold1["support"], 1)
        self.assertEqual(no_controlled_fold1["recall"], 1.0)

    def test_classwise_aggregation(self):
        """Test mean and standard deviation of class-wise metrics."""

        mdma_stats = self.df_cm_stats[
            self.df_cm_stats["type"] == "MDMA"
        ].iloc[0]

        self.assertAlmostEqual(mdma_stats[("recall", "mean")], 0.75)
        self.assertAlmostEqual(
            mdma_stats[("recall", "std")],
            np.std([0.5, 1.0], ddof=1)
        )

        self.assertAlmostEqual(mdma_stats[("support", "mean")], 1.5)
        self.assertAlmostEqual(
            mdma_stats[("support", "std")],
            np.std([2, 1], ddof=1)
        )

    def test_f1_per_fold(self):
        """Test Macro-F1 and Weighted-F1 for each fold."""

        fold0 = self.df_fms_folds[
            self.df_fms_folds["fold"] == 0
        ].iloc[0]

        self.assertAlmostEqual(fold0["Macro-F1"], 0.1833, places = 4)
        self.assertAlmostEqual(fold0["Weighted-F1"], 0.7333, places = 4)

        fold1 = self.df_fms_folds[
            self.df_fms_folds["fold"] == 1
        ].iloc[0]

        self.assertAlmostEqual(fold1["Macro-F1"], 0.29167, places = 4)
        self.assertAlmostEqual(fold1["Weighted-F1"], 0.7500, places = 4)

    def test_f1_aggregation(self):
        """Test mean and standard deviation of F1 scores."""

        self.assertAlmostEqual(
            self.df_fms_stats.loc[0, "Macro-F1"],
            0.2375
        )

        self.assertAlmostEqual(
            self.df_fms_stats.loc[1, "Macro-F1"],
            np.std([0.1833333333, 0.2916666667], ddof=1)
        )

        self.assertAlmostEqual(
            self.df_fms_stats.loc[0, "Weighted-F1"],
            0.74167, places = 4
        )

        self.assertAlmostEqual(
            self.df_fms_stats.loc[1, "Weighted-F1"],
            np.std([0.7333333, 0.75], ddof=1)
        )
        
    def test_create_model_comparison(self):
        """Test whether fold-based and pooled evaluation results
        are correctly combined into a structured dictionary.
        """

        model_name = "dummy"

        comparison = create_model_comparison(
            self.aggregate_results,
            self.instrument,
            model_name
        )

        # Check overall structure
        self.assertIsInstance(comparison, dict)
        self.assertEqual(comparison["instrument"], self.instrument)
        self.assertEqual(comparison["model"], model_name)

        # Check fold evaluation
        self.assertIn("fold_evaluation", comparison)

        expected_fold_keys = [
            "class_metrics",
            "class_statistics",
            "f1_scores",
            "f1_statistics"
        ]

        fold_evaluation = comparison["fold_evaluation"]

        for key in expected_fold_keys:
            self.assertIn(key, fold_evaluation)
            self.assertIsInstance(
                fold_evaluation[key], pd.DataFrame
            )

        # Check pooled metrics
        self.assertIn("pooled_metrics", comparison)

        pooled_metrics = comparison["pooled_metrics"]

        self.assertIsInstance(pooled_metrics, dict)
        self.assertEqual(pooled_metrics["instrument"], self.instrument)
        self.assertEqual(pooled_metrics["model_name"], model_name)

        self.assertIn("Macro-F1", pooled_metrics)
        self.assertIn("Weighted-F1", pooled_metrics)


        