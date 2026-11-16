#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:09:16 2026

@author: kathrin
"""

from nir_forensics.modeling.ml_models import fit_dummy_classifier, fit_random_forest_classifier
from nir_forensics.modeling.split import create_cv_splits, split_features_target
from nir_forensics.modeling.dataset import build_modeling_dataset
from nir_forensics.modeling.evaluation import create_model_comparison, create_model_comparison_table
from nir_forensics.modeling.metrics import calculate_confusion_matrix
from nir_forensics.plots import plot_confusion_matrix_heatmap
from nir_forensics.paths import MODELS_DIR, PREDICTIONS_DIR


import pandas as pd
import joblib



def prepare_dataset(instrument: str = None, dataset: pd.DataFrame = None):
    if instrument is not None and dataset is not None:
        raise ValueError("Only one of instrument or dataset can be given.")

    
    if dataset is None and instrument is None:
        raise ValueError("Either instrument or dataset must be given.")
    
    if dataset is None:
        dataset = build_modeling_dataset(instrument)
    train_indices, test_indices = create_cv_splits(dataset)
    return dataset, train_indices, test_indices

def make_predictions(dataset, train_indices, test_indices,
                    model_name, model_func,  model_params=None, instrument=None):
    """Makes 5-fold predictions for chosen model
        
    Args:
        dataset (pd.DataFrame) : dataset from build_modelling_dataset in question
        train_indices (list) : indices in dataset of train data 
        test_indices (list) : indices in dataset of test data
        model_name (str) : name of the model to be trained e.g. rfc, dummy
        model_func (function) : defines which function is used for training and predictions
        model_params (dict) : dictionary with parameters for the model default = None
                                model_params must use the names given in the function to call as keys!
        instrument (str) : string with name of instrument for saving and loading of model/prediction
                            default = None: if testdataset is given directely
        
        
    Returns: 
        list of dictionaries with number of fold, predictions and true labels
         
    """
    
    if instrument is not None:
        model_dir = MODELS_DIR / instrument / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = PREDICTIONS_DIR / instrument / model_name
        predictions_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    if model_params is None:
        model_params = {}
    
    results = []
    for i, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
        trainset = dataset.iloc[train_idx]
        testset = dataset.iloc[test_idx]
        train_features, train_target, test_features, test_target = split_features_target(trainset, testset)
        if instrument is not None:
            model_path = model_dir / f"fold_{i}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
            else:
                    model = model_func(train_features, train_target, **model_params)
                    joblib.dump(model, model_path)
        else:
            model = model_func(train_features, train_target, **model_params)
        
        predictions = model.predict(test_features)
        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_.tolist()
        wavelengths = train_features.columns.tolist()
        result = {"fold": i,
                        "predictions": predictions,
                        "true_labels": test_target,
                        "wavelengths": wavelengths,
                        "feature_importances": feature_importances
                        }
        if instrument is not None:
            prediction_path = predictions_dir / f"fold_{i}.pkl"
            joblib.dump(result, prediction_path)
        results.append(result)
        if instrument is not None:
            prediction_path = predictions_dir / "all_folds.pkl"
            joblib.dump(results, prediction_path)
    return results

def run_model_evaluation(instruments=None, model_functions=None, dataset=None):
    """Run model evaluation for multiple instruments or a given dataset.

    Args:
        instruments (list): Instruments to evaluate.
        model_config (dict): Models to evaluate: name, according function, model_params.
                             Example: model_functions = {
                                 "Dummy": {
                                     "func": fit_dummy_classifier,
                                     "params": {}
                                 },
                                 "RFC_default": {
                                     "func": fit_random_forest_classifier,
                                     "params": {}
                                 },
                                 "RFC_balanced": {
                                     "func": fit_random_forest_classifier,
                                     "params": {"class_weight": "balanced"}
                                 }
                             }
        dataset (pd.DataFrame, optional): Dataset for direct evaluation.
                                          If given, instruments must be None.
                    

    Returns:
        dict: Evaluation results for each instrument-model combination.
        
    get result of single instrument (example): evaluation_results[("MicroNIR", "rfc")]
        only metrics: evaluation_results[("MicroNIR", "rfc")]["metrics"]
        figures: evaluation_results[("MicroNIR", "rfc")]["figures"]["confusion_matrix"]
    """
   
    if model_functions is None:
        raise ValueError("model_functions must be provided.")

    if instruments is not None and dataset is not None:
        raise ValueError("Only one of instruments or dataset can be given.")

    if instruments is None and dataset is None:
        raise ValueError("Either instruments or dataset must be given.")

    evaluation_results = {}

    if dataset is not None:
        instruments = [None]
        datasets = {None: prepare_dataset(dataset=dataset)}
    else:
        datasets = {
            instrument: prepare_dataset(instrument=instrument)
            for instrument in instruments
        }

    for instrument in instruments:

        current_dataset, train_indices, test_indices = datasets[instrument]

        for model_name, model_config in model_functions.items():
            model_func = model_config["func"]
            model_params = model_config["params"]

            if instrument is not None:
                predictions_dir = PREDICTIONS_DIR / instrument / model_name
                prediction_path = predictions_dir / "all_folds.pkl"
            else:
                prediction_path = None

            if prediction_path is not None and prediction_path.exists():
                results = joblib.load(prediction_path)
            else:
                results = make_predictions(
                    current_dataset,
                    train_indices,
                    test_indices,
                    model_name,
                    model_func,
                    model_params=model_params,
                    instrument=instrument
                )

            model_comparison = create_model_comparison(
                results,
                instrument,
                model_name
            )

            conf_matrix = calculate_confusion_matrix(results)

            fig_cm = plot_confusion_matrix_heatmap(
                conf_matrix,
                instrument or "Test dataset"
            )

            fig_cm.axes[0].set_title(
                f"{instrument or 'Test dataset'} – {model_name}"
            )

            evaluation_results[(instrument, model_name)] = {
                "metrics": model_comparison,
                "predictions": results,
                "confusion_matrix": conf_matrix,
                "figures": {
                    "confusion_matrix": fig_cm
                }
            }

    return evaluation_results


if __name__ == "__main__":
    model_functions = {
        "Dummy": {
            "func": fit_dummy_classifier,
            "params": {}
        },
        "RFC_default": {
            "func": fit_random_forest_classifier,
            "params": {}
        },
        "RFC_balanced": {
            "func": fit_random_forest_classifier,
            "params": {"class_weight": "balanced"}
        }
    }

    out = run_model_evaluation(
        instruments=["MicroNIR", "NeoSpectra", "Scio", "NIRONE"],
        model_functions=model_functions
    )

    df_comparison = create_model_comparison_table(out)

    df_comparison = df_comparison.sort_values(
        ["instrument", "model"]
    ).reset_index(drop=True)

