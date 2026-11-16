#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:55:19 2026

@author: kathrin
"""
import unittest

from nir_forensics.paths import FIXTURE_DIR, PROJECT_ROOT
 
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.models import prepare_dataset, make_predictions, run_model_evaluation
from nir_forensics.modeling.ml_models import fit_dummy_classifier, fit_random_forest_classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TestModels(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):
        """Load  and prepare data-files once (using the fixture)"""
        df = pd.read_excel(FIXTURE_DIR / "cv_folding_fixture.xlsx")
        cls.dataset, cls.train_indices, cls.test_indices = prepare_dataset(dataset = df)
        cls.results = make_predictions(cls.dataset, 
                                       cls.train_indices, 
                                       cls.test_indices, 
                                       "dummy", 
                                       fit_dummy_classifier)
        cls.model_functions = {
                                "dummy": {
                                    "func": fit_dummy_classifier,
                                    "params": {}
                                },
                                "rfc": {
                                    "func": fit_random_forest_classifier,
                                    "params": {}
                                }
                            }
        cls.out = run_model_evaluation(
            instruments=["MicroNIR"],
            model_functions=cls.model_functions
        )
    @classmethod
    def tearDownClass(cls):
        for result in cls.out.values():
            plt.close(result["figures"]["confusion_matrix"])
    
    def test_prepare_dataset_invalid_arguments(self):
        """Test raising ValueError in prepare_dataset"""
        with self.assertRaises(ValueError):
            prepare_dataset()
        with self.assertRaises(ValueError):
            prepare_dataset("micronir", self.dataset)

    def test_make_predictions_structure(self):
        """Test structure and length of make_predictions results."""
        
        # Check return type and number of folds
        self.assertIsInstance(self.results, list)
        self.assertEqual(len(self.results), 5)
    
        # Check dictionary structure
        i = 0
        for result in self.results:
            self.assertIsInstance(result, dict)
            self.assertEqual(
                set(result.keys()),
                {"fold", "predictions", "true_labels", "feature_importances", "wavelengths"}
            )
    
            # Check fold number
            self.assertIn(result["fold"], range(5))
            self.assertEqual(result["fold"], i)
            i+=1
    
            # Check prediction and target lengths
            self.assertEqual(
                len(result["predictions"]),
                len(result["true_labels"])
            )
    
    
    def test_make_predictions_valid_labels(self):
        """Test whether predictions and true labels contain valid classes."""
    
          
        labels = [
            "2C-B_phenethylamine-psychedelics",
            "MDMA",
            "amphetamine_type",
            "cathinone",
            "cocaine",
            "ketamine",
            "no_controlled_substance",
            "other"
        ]
    
        for result in self.results:
            for label in result["predictions"]:
                self.assertIn(label, labels)
    
            for label in result["true_labels"]:
                self.assertIn(label, labels)
    
    
 
    def test_make_predictions_rfs_feature_importances(self):
        """Random Forest: feature_importance != None and same lengths as wavelengths"""
        out = make_predictions(self.dataset, 
                               self.train_indices, 
                               self.test_indices, 
                               "rfc",
                               fit_random_forest_classifier)
        for i in range(len(out)):
            self.assertIn("feature_importances", out[i])
            self.assertIsNotNone(out[i]["feature_importances"])
            self.assertEqual(len(out[i]["feature_importances"]), len(out[i]["wavelengths"]))
        
    def test_make_predictions_dummy_feature_importances(self):
         """Dummy: feature_importance == None and wavelengths not empty"""
         out = make_predictions(self.dataset, 
                                self.train_indices, 
                                self.test_indices, 
                                "dummy",
                                fit_dummy_classifier)
         for i in range(len(out)):
             self.assertIn("feature_importances", out[i])
             self.assertIsNone(out[i]["feature_importances"])
             self.assertIn("wavelengths", out[i])
             self.assertGreater(len(out[i]["wavelengths"]),  0)
             
    def test_run_model_evaluation_structure(self):
        """Test return type and structure of run_model_evaluation."""
    
        self.assertIsInstance(self.out, dict)
        self.assertEqual(len(self.out), 2)
    
        for key, result in self.out.items():
    
            # Check instrument-model keys
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
    
            instrument, model_name = key
    
            self.assertEqual(instrument, "MicroNIR")
            self.assertIn(model_name, ["dummy", "rfc"])
    
            # Check result dictionary
            self.assertEqual(
                set(result.keys()),
                {"metrics", "predictions", "confusion_matrix", "figures"}
            )


    def test_run_model_evaluation_predictions(self):
        """Test predictions stored in evaluation results."""
    
        
    
        for result in self.out.values():
    
            self.assertIsInstance(result["predictions"], list)
            self.assertEqual(len(result["predictions"]), 5)
    
            for fold_result in result["predictions"]:
                self.assertIn("predictions", fold_result)
                self.assertIn("true_labels", fold_result)
    
    
    def test_run_model_evaluation_metrics(self):
        """Test metrics stored in evaluation results."""
    
       
        for result in self.out.values():
    
            self.assertIsInstance(result["metrics"], dict)
    
            self.assertIn("instrument", result["metrics"])
            self.assertIn("model", result["metrics"])
            self.assertIn("fold_evaluation", result["metrics"])
            self.assertIn("pooled_metrics", result["metrics"])
    
    
    def test_run_model_evaluation_confusion_matrix(self):
        """Test confusion matrix stored in evaluation results."""
    
       
        for result in self.out.values():
    
            self.assertIsInstance(result["confusion_matrix"], np.ndarray)
            self.assertEqual(result["confusion_matrix"].shape, (8, 8))
            self.assertGreater(result["confusion_matrix"].sum(), 0)
    
    
    def test_run_model_evaluation_figures(self):
        """Test figures stored in evaluation results."""
    
     
        for result in self.out.values():
    
            self.assertIn("confusion_matrix", result["figures"])
            self.assertIsNotNone(result["figures"]["confusion_matrix"])