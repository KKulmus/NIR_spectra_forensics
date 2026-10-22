#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:47:12 2026

@author: kathrin
"""

import unittest

from nir_forensics.modeling.metrics import calculate_metric

y_true = ["A", "A", "A","B"]
y_pred = ["A", "A", "B","B"]
#class A: F1 = 0.8
#class B: F1 ≈ 0.667
#Macro-F1 ≈ 0.733
#Weighted ~ 0.766

class TestMetrics(unittest.TestCase):
    
    def test_calculate_f1_macro(self):
        """Macro F1 is calculated correctly"""
        f1_macro = calculate_metric(y_true, y_pred, 'macro')
        self.assertAlmostEqual(f1_macro, 0.733, places=3)
        
    def test_calculate_f1_weighted(self):
        """Weighted F1 is calculated correctly"""
        f1_weighted = calculate_metric(y_true, y_pred, 'weighted')
        self.assertAlmostEqual(f1_weighted, 0.7666, places=3)