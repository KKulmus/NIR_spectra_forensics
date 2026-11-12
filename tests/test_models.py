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
from analysis.models import prepare_dataset, make_predictions
import pandas as pd


class TestModels(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):
        """Load  and prepare data-files once (using the fixture)"""
        df = pd.read_excel(FIXTURE_DIR / "cv_folding_fixture.xlsx")
        cls.dataset, cls.train_indices, cls.test_indices = prepare_dataset(dataset = df)
        cls.results = make_predictions(cls.dataset, cls.train_indices, cls.test_indices, "dummy")
    
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
                {"fold", "predictions", "true_labels"}
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
    
    
    def test_make_predictions_invalid_model(self):
        """Test whether invalid model names raise a ValueError."""
    
           
        with self.assertRaises(ValueError):
            make_predictions(self.dataset, self.train_indices, self.test_indices, "invalid_model")
