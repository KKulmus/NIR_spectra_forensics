#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:05:46 2026

@author: kathrin
"""



import numpy as np

def pooling_folds(results):
    true_labels = [result["true_labels"] for result in results]
    true_labels = np.concatenate(true_labels)
    predictions = [result["predictions"] for result in results]
    predictions = np.concatenate(predictions)
    
    return true_labels, predictions