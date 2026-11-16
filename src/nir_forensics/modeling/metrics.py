#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:06:27 2026

@author: kathrin
"""


from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
from nir_forensics.modeling.folds_utils import pooling_folds

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

def calculate_metric(test_target, pred_target, average, labels = None):
    """Calculates the F1-score for predicted target values.

    Args:
        test_target (pd.Series): True target values of the test dataset.
        pred_target (array-like): Predicted target values from a model.
        average (str): Averaging method for the F1-score, e.g. "macro"
            or "weighted".
        labels (list) : optional. Force F1 for all labels, even if they don't exist in fold

    Returns:
        float: F1-score calculated using the specified averaging method.
    """
    
    return f1_score(test_target, pred_target, average=average, labels=labels, zero_division=0)


def calculate_classification_metrics(true_labels, predictions, zero_division = 0):
    """Calculates classification_report for true_label - predicted label.
    
    Args:
        true_labels (array-like): array-like of true labels in the test-fold
        prediction (array-like): array-like of predicted labels from test-fold
        
    Returns:
        classification-report as dictionary
    """
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
    report = classification_report(
        true_labels,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=zero_division
    )
    return report

def calculate_confusion_matrix(results):
    """takes the results list, calls pooling and returns the confusion matrix
        
        Args:
            results (list of dicts): list of dictionaries with model predicions and true labels
            
        Returns:
            confusion_matrix (np.array) of pooled folds for results
    """
    
    
    true_labels, predictions = pooling_folds(results)
        
        
    conf_matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=all_labels,
        normalize="true"
    )
    
    return conf_matrix

def calculate_fold_confusion_matrices(results):
    """Calculated the confusion matrix for each fold.
        
        Args: 
            results (list of dicts) : results from fold-wise prediction of a model
            
        Returns: 
            list with a confusion matrix for each fold
    """
    cm_list = []
    for result in results:
            fold = result["fold"]
            fold_true = result["true_labels"]
            fold_pred = result["predictions"]

            cm = confusion_matrix(
                fold_true,
                fold_pred,
                labels=all_labels
            )
            cm_list.append({
                        "fold": fold,
                        "confusion_matrix" : cm
                })
    return cm_list

def calculate_fold_metrics(results):
    """Calculates the macro and weighted f1-scores for each fold in results seaparately.
        
        Args:
            results (list of dicts) : results from fold-wise prediction of a model
            
        Returns:
            list of dictionaries with fold number, macro-f1 and weighted-f1 for each fold
    """
    f1_metrics = []
    for result in results:
            fold = result["fold"]
            fold_true = result["true_labels"]
            fold_pred = result["predictions"]
            macro = calculate_metric(fold_true, fold_pred, "macro", labels = all_labels)
            weighted = calculate_metric(fold_true, fold_pred, "weighted", labels = all_labels)
            f1_metrics.append({"fold": fold,
                             "Macro-F1" : macro,
                             "Weighted-F1" : weighted
                             } )
    return f1_metrics

def calculate_pooled_metrics(results, instrument, model_name):
    """Calculates the macro and weighted f1-scores pooled folds.
        
        Args:
            results (list of dicts) : results from fold-wise prediction of a model
            instrument (str) : name of the instrumen
            model_name (str) : name of the model
            
        Returns:
            dictionariy with instrument, model_name, macro-f1 and weighted-f1 for each fold
    """
    true_labels, predictions = pooling_folds(results)
    macro = calculate_metric(true_labels, predictions, "macro", labels = all_labels)
    weighted = calculate_metric(true_labels, predictions, "weighted", labels = all_labels)
    f1_metrics = {"instrument": instrument,
                  "model_name" : model_name,
                     "Macro-F1" : macro,
                     "Weighted-F1" : weighted
                     }
    return f1_metrics
    

def compare_class_distributions(results, instrument):
    """Determine a dataframe with relative occurrence of true substance, predicted substance
        and difference between both.
        
        Args:
            results (list of dicts) : list of dictionaries with true and predicted labels per fold
            instrument (str) : Name of instrument for labeling
            
        Returns:
            pd.DataFrame with substance as index, and columns: relative occurrence of substance true,
                    relative occurrence of sustance predicted and difference beween both;
                    if difference is positive, occurrence is overestimated else underestimated
        
        """
    true_labels, predictions = pooling_folds(results)
    true_distribution = (
    pd.Series(true_labels)
    .value_counts(normalize=True)
    )

    predicted_distribution = (
        pd.Series(predictions)
        .value_counts(normalize=True)
    )

    distribution_comparison = pd.concat(
        [true_distribution, predicted_distribution],
        axis=1,
        keys=["true", "predicted"]
    ).fillna(0)
    
    distribution_comparison[f"difference {instrument}"] = (
        distribution_comparison["predicted"]
        - distribution_comparison["true"]
    )
    distribution_comparison = distribution_comparison.reindex(all_labels, fill_value=0)
    return distribution_comparison



    
    