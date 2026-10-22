#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:06:27 2026

@author: kathrin
"""


from sklearn.metrics import f1_score


def calculate_metric(test_target, pred_target, average):
    """Calculates the F1-score for predicted target values.

    Args:
        test_target (pd.Series): True target values of the test dataset.
        pred_target (array-like): Predicted target values from a model.
        average (str): Averaging method for the F1-score, e.g. "macro"
            or "weighted".

    Returns:
        float: F1-score calculated using the specified averaging method.
    """
    
    return f1_score(test_target, pred_target, average=average)

