#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:47:12 2026

@author: kathrin
"""

import unittest

from nir_forensics.modeling.metrics import calculate_metric, calculate_confusion_matrix, calculate_classification_metrics
from nir_forensics.modeling.metrics import calculate_fold_confusion_matrices, calculate_fold_metrics

y_true = ["A", "A", "A","B"]
y_pred = ["A", "A", "B","B"]
#class A: F1 = 0.8
#class B: F1 ≈ 0.667
#Macro-F1 ≈ 0.733
#Weighted ~ 0.766

all_labels = [
      "2C-B_phenethylamine-psychedelics",
      "MDMA",
      "amphetamine_type",
      "cathinone",
      "cocaine",
      "ketamine",
      "no_controlled_substance",
      "other"
  ]
true_fold = [["2C-B_phenethylamine-psychedelics",
              "2C-B_phenethylamine-psychedelics",
              "MDMA",
              "amphetamine_type", 
              "cathinone", 
              "cocaine", 
              "ketamine", 
              "no_controlled_substance", 
              "other"], ["2C-B_phenethylamine-psychedelics",
              "MDMA",
              "amphetamine_type", 
              "cathinone", 
              "cocaine", 
              "ketamine", 
              "no_controlled_substance", 
              "MDMA",
              "MDMA"]]
predictions = [["cocaine",
              "2C-B_phenethylamine-psychedelics",
              "MDMA",
              "amphetamine_type", 
              "cathinone", 
              "cocaine", 
              "ketamine", 
              "no_controlled_substance", 
              "other"], ["2C-B_phenethylamine-psychedelics",
              "MDMA",
              "amphetamine_type", 
              "MDMA", 
              "cocaine", 
              "ketamine", 
              "no_controlled_substance", 
              "MDMA",
              "MDMA"]]
results = [{"fold": i , 
            "true_labels": true_fold[i],
            "predictions" : predictions[i]} for i in range(len(true_fold))]

class TestMetrics(unittest.TestCase):
    
    def test_calculate_f1_macro(self):
        """Macro F1 is calculated correctly"""
        f1_macro = calculate_metric(y_true, y_pred, 'macro')
        self.assertAlmostEqual(f1_macro, 0.733, places=3)
        
    def test_calculate_f1_weighted(self):
        """Weighted F1 is calculated correctly"""
        f1_weighted = calculate_metric(y_true, y_pred, 'weighted')
        self.assertAlmostEqual(f1_weighted, 0.7666, places=3)
        
    def test_confusion_matrix(self):
        """Test whether the pooled and row-normalized confusion matrix
        contains all eight classes and correctly represents classification
        results and misclassifications.
        """
              
        cm =  calculate_confusion_matrix(results)
        self.assertEqual(cm.shape, (8, 8))
        
        idx_2cb = all_labels.index("2C-B_phenethylamine-psychedelics")
        idx_cocaine = all_labels.index("cocaine")
        
        self.assertAlmostEqual(cm[idx_2cb, idx_2cb], 2/3)
        self.assertAlmostEqual(cm[idx_2cb, idx_cocaine], 1/3)
        self.assertAlmostEqual(cm[idx_2cb].sum(), 1.0)
        
    def test_calculate_classification_metrics(self):
        """Test whether the classification report contains all eight classes
            and correctly calculates precision, recall, F1-score and support
            for selected classes.
        """
        
        true_labels = ["2C-B_phenethylamine-psychedelics",
                      "2C-B_phenethylamine-psychedelics",
                      "MDMA",
                      "amphetamine_type", 
                      "cathinone", 
                      "cocaine", 
                      "ketamine", 
                      "no_controlled_substance", 
                      "other"]
        predictions = ["cocaine",
                      "2C-B_phenethylamine-psychedelics",
                      "MDMA",
                      "amphetamine_type", 
                      "cathinone", 
                      "cocaine", 
                      "ketamine", 
                      "other", 
                      "other"]
        
        report = calculate_classification_metrics(true_labels, predictions)
       
        self.assertIsInstance(report, dict)
        for label in all_labels:
            self.assertIn(label, report)
        self.assertAlmostEqual(report["cocaine"]["precision"], 0.5)
        self.assertAlmostEqual(report["cocaine"]["recall"], 1)
        self.assertAlmostEqual(report["cocaine"]["f1-score"], 2/3)
        self.assertAlmostEqual(report["cocaine"]["support"], 1)
        self.assertAlmostEqual(report["no_controlled_substance"]["precision"], 0)
        
    def test_calculate_fold_confusion_matrices(self):
        """Test whether calculate_fold_confusion_matrices returns a list
            of dictionaries containing the correct fold number and an 8x8
            confusion matrix with the expected absolute class counts.
        """
        cm = calculate_fold_confusion_matrices(results)
        
        self.assertIsInstance(cm, list)
        self.assertEqual(len(cm), 2)
        for i in range(2):
            self.assertIsInstance(cm[i], dict)
            self.assertEqual(cm[i]["fold"], i)
            self.assertIn("confusion_matrix", cm[i])
            self.assertEqual(cm[i]["confusion_matrix"].shape, (8,8))
        idx_2cb = all_labels.index("2C-B_phenethylamine-psychedelics")
        idx_cocaine = all_labels.index("cocaine")
        
        self.assertAlmostEqual(cm[0]["confusion_matrix"][idx_2cb, idx_2cb], 1)
        self.assertAlmostEqual(cm[0]["confusion_matrix"][idx_2cb, idx_cocaine], 1)
        self.assertAlmostEqual(cm[0]["confusion_matrix"][idx_2cb].sum(), 2)
        
    def test_calculate_fold_metrics(self):
         f1 = calculate_fold_metrics(results)
         self.assertIsInstance(f1, list)
         self.assertEqual(len(f1), 2)
         for i in range(2):
             self.assertIsInstance(f1[i], dict)
             self.assertEqual(f1[i]["fold"], i)
             self.assertIn("Macro-F1", f1[i])
             self.assertIn("Weighted-F1", f1[i])
         
         self.assertAlmostEqual(f1[0]["Macro-F1"], 11/12)
         self.assertAlmostEqual(f1[0]["Weighted-F1"], 8/9)
        
             
        