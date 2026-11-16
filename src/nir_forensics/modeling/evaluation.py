#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:05:46 2026

@author: kathrin
"""



import numpy as np
import pandas as pd

from nir_forensics.modeling.metrics import (calculate_fold_confusion_matrices, 
                                            all_labels, 
                                            calculate_fold_metrics,
                                            calculate_pooled_metrics)

#from nir_forensics.modeling.folds_utils import pooling_folds
def aggregate_fold_metrics(results, instrument):
    """Aggregate fold-level evaluation metrics for one instrument.
        
        Calculates class-wise Recall and Support for each fold based on the
        confusion matrices. Aggregates the per-class values across folds by
        computing their mean and standard deviation.
        
        Additionally, aggregates Macro-F1 and Weighted-F1 scores across folds
        by computing their mean and standard deviation.
        
        Args:
            results (list of dicts): List of dictionaries containing the true labels and
            predictions for each fold.
            instrument (str): Name of the spectrometer instrument.
        
        Returns:
            Tuple containing four pandas DataFrames:
            - df_cm_folds: Class-wise Recall and Support for each fold.
            - df_cm_stats: Mean and standard deviation of Recall and
            Support for each class across folds.
            - df_fms_stats: Mean and standard deviation of Macro-F1 and
            Weighted-F1 across folds.
            - df_fms_folds: Macro-F1 and Weighted-F1 scores for each fold.
    """
    
    cms = calculate_fold_confusion_matrices(results)
    cm_fold_stats = []
    for cm in cms:
        fold = cm["fold"]
        cm_matrix = cm["confusion_matrix"]
        support = np.sum(cm_matrix, axis=1)
        for i, substance_type in enumerate(all_labels):
    
                recall = (
                    cm_matrix[i, i] / support[i]
                    if support[i] > 0
                    else np.nan
                )
    
                cm_fold_stats.append({
                    "instrument": instrument,
                    "fold": fold,
                    "type": substance_type,
                    "recall": recall,
                    "support": support[i]
                })
    #recall and support per class and fold
    df_cm_folds = pd.DataFrame(cm_fold_stats)
    #mean and std per class
    df_cm_stats = (
        df_cm_folds
        .groupby(["type"])[["recall", "support"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    df_cm_stats["instrument"] = instrument
    fms = calculate_fold_metrics(results)
    #macro and weighted f1 per fold
    df_fms_folds = pd.DataFrame(fms)
    df_fms_folds["instrument"] = instrument
    #mean and std of f1 
    df_fms_stats = (
        df_fms_folds[["Macro-F1", "Weighted-F1"]]
        .agg(["mean", "std"])
        .reset_index()
        
        )
    df_fms_stats["instrument"] = instrument


    
    return df_cm_folds, df_cm_stats, df_fms_stats, df_fms_folds

def create_model_comparison(results, instrument, model_name):
    """Create a comprehensive evaluation summary for one instrument-model combination.

    Combines fold-based evaluation metrics and pooled evaluation metrics
    into a structured dictionary.
    
    Args:
        results (list of dicts): Fold-wise predictions and true labels.
        instrument (str): Name of the spectrometer instrument.
        model_name (str): Name of the evaluated model.
    
    Returns:
        dict: Evaluation results containing:
            - instrument: Instrument name.
            - model: Model name.
            - fold_evaluation:
                - class_metrics (DataFrame): Class-wise Recall and Support
                  for each fold.
                - class_statistics (DataFrame): Mean and standard deviation
                  of Recall and Support across folds.
                - f1_scores (DataFrame): Macro-F1 and Weighted-F1 per fold.
                - f1_statistics (DataFrame): Mean and standard deviation
                  of Macro-F1 and Weighted-F1 across folds.
            - pooled_metrics (dict): Pooled Macro-F1 and Weighted-F1 scores.
    """
    df_cm_folds, df_cm_stats, df_fms_stats, df_fms_folds = aggregate_fold_metrics(results, instrument)
    pooled_metrics = calculate_pooled_metrics(results, instrument, model_name)
    
    comparison = {
    "instrument": instrument,
    "model": model_name,

    "fold_evaluation": {
        "class_metrics": df_cm_folds,
        "class_statistics": df_cm_stats,
        "f1_scores": df_fms_folds,
        "f1_statistics": df_fms_stats
    },

    "pooled_metrics": pooled_metrics
}
    return comparison

def create_model_comparison_table(evaluation_results):
    """Create a comparison table for all evaluated instruments and models.

    Args:
        evaluation_results (dict): Results from run_model_evaluation().

    Returns:
        pd.DataFrame: Comparison of fold statistics and pooled metrics.
    """

    comparison = []

    for (instrument, model_name), result in evaluation_results.items():

        metrics = result["metrics"]

        f1_statistics = metrics["fold_evaluation"]["f1_statistics"]
        pooled_metrics = metrics["pooled_metrics"]

        mean_values = f1_statistics.loc[
            f1_statistics["index"] == "mean"
        ].iloc[0]

        std_values = f1_statistics.loc[
            f1_statistics["index"] == "std"
        ].iloc[0]

        comparison.append({
            "instrument": instrument,
            "model": model_name,
            "Macro-F1 mean": mean_values["Macro-F1"],
            "Macro-F1 std": std_values["Macro-F1"],
            "Weighted-F1 mean": mean_values["Weighted-F1"],
            "Weighted-F1 std": std_values["Weighted-F1"],
            "Macro-F1 pooled": pooled_metrics["Macro-F1"],
            "Weighted-F1 pooled": pooled_metrics["Weighted-F1"]
        })

    return pd.DataFrame(comparison)
    