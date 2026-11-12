#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:09:16 2026

@author: kathrin
"""

from nir_forensics.modeling.ml_models import fit_dummy_classifier, fit_random_forest_classifier
from nir_forensics.modeling.split import create_cv_splits, split_features_target
from nir_forensics.modeling.dataset import build_modeling_dataset
import pandas as pd


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
                    model_name, model_params=None):
    """Makes 5-fold predictions for chosen model
        
    Args:
        dataset (pd.DataFrame) : dataset from build_modelling_dataset in question
        train_indices (list) : indices in dataset of train data 
        test_indices (list) : indices in dataset of test data
        model_name (str) : name of the model to be trained e.g. rfc, dummy
        model_params (dict) : dictionary with parameters for the model default = None
                                model_params must use the names given in the function to call as keys!
        
    Returns: 
        list of dictionaries with number of fold, predictions and true labels
         
    """
        
    models = {"dummy": fit_dummy_classifier,
              "rfc": fit_random_forest_classifier
              }
    if model_name not in models:
        raise ValueError(f"Incorrect model name, valid names are: {list(models)}")
    if model_params is None:
        model_params = {}
    
    results = []
    for i, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
        trainset = dataset.iloc[train_idx]
        testset = dataset.iloc[test_idx]
        train_features, train_target, test_features, test_target = split_features_target(trainset, testset)
        
        model = models[model_name](train_features, train_target,**model_params) 
        predictions = model.predict(test_features)
        results.append({"fold": i,
                        "predictions": predictions,
                        "true_labels": test_target})
    return results




# instruments = ["nirone", "micronir", "neospectra", "scio"]

# 

# fig, axs = plt.subplots(2, 2, figsize=(18, 16))
# axs = axs.flat

# cm_fold_stats = []
# recall_rfc = []
# recall_rfc_balanced = []
# for ax, instrument in zip(axs, instruments):

#     results = make_predictions(instrument)

#     # Check priority class
#     #for i, result in enumerate(results):
#         #print(instrument, i, result["predictions_dummy"][0])

#     # Results for dummy classifier
#     dummy_f1 = metrics_list("predictions_dummy", results)
#     df_dummy_f1 = pd.DataFrame(dummy_f1).set_index("fold")
#     df_dummy_f1["Model"] = "Dummy"

#     mean_std_dummy = df_dummy_f1.drop("Model", axis=1).agg(["mean", "std"])

#     # Results for random forest classifier
#     rfc_f1 = metrics_list("predictions_rfc", results)
#     df_rfc_f1 = pd.DataFrame(rfc_f1).set_index("fold")
#     df_rfc_f1["Model"] = "Random Forest"
    
#     # Results for random forest classifier
#     rfc_balanced_f1 = metrics_list("predictions_rfc_balanced", results)
#     df_rfc_balanced_f1 = pd.DataFrame(rfc_balanced_f1).set_index("fold")
#     df_rfc_balanced_f1["Model"] = "Random Forest balanced"

#     df_compare = pd.concat([df_dummy_f1, df_rfc_f1,df_rfc_balanced_f1])
    
#     mean_std_rfc = df_rfc_f1.drop("Model", axis=1).agg(["mean", "std"])
#     mean_std_rfc_balanced = df_rfc_balanced_f1.drop("Model", axis=1).agg(["mean", "std"])

#     # Confusion matrix (pooled test predictions)
#     true = np.concatenate([result["true_labels"] for result in results])
#     pred = np.concatenate([result["predictions_rfc"] for result in results])
#     pred_balanced = np.concatenate([result["predictions_rfc_balanced"] for result in results])
    
#     recall_rfc.append(recall_score(true, pred, average = "weighted"))
#     recall_rfc_balanced.append(recall_score(true, pred_balanced, average = "weighted"))
    
    
#     conf_matrix = confusion_matrix(
#         true,
#         pred,
#         labels=all_labels,
#         normalize="true"
#     )

#     # Plot confusion matrix
#     sns.heatmap(
#         conf_matrix,
#         annot=True,
#         fmt=".2f",
#         cmap="Blues",
#         xticklabels=all_labels,
#         yticklabels=all_labels,
#         ax=ax,
#         vmin=0,
#         vmax=1
#     )

#     ax.set_title(instrument)
#     ax.set_xlabel("Predicted label")
#     ax.set_ylabel("True label")

#     # Confusion matrices per CV fold
#    

#         # Class-wise recall per fold
#         for i, substance_type in enumerate(all_labels):

#             recall = (
#                 cm[i, i] / support[i]
#                 if support[i] > 0
#                 else np.nan
#             )

#             cm_fold_stats.append({
#                 "instrument": instrument,
#                 "fold": fold,
#                 "type": substance_type,
#                 "recall": recall,
#                 "support": support[i]
#             })

#     #true class distribution in dataset, all testfolds and predicted distribution in all testfolds
#     true_distribution = (
#     pd.Series(true)
#     .value_counts(normalize=True)
#     )

#     predicted_distribution = (
#         pd.Series(pred)
#         .value_counts(normalize=True)
#     )

#     distribution_comparison = pd.concat(
#         [true_distribution, predicted_distribution],
#         axis=1,
#         keys=["true", "predicted"]
#     ).fillna(0)
    
#     distribution_comparison[f"difference {instrument}"] = (
#         distribution_comparison["predicted"]
#         - distribution_comparison["true"]
#     )
    
#     #print(distribution_comparison)
    
# fig.tight_layout()
# #fig.savefig(PROJECT_ROOT / "figs" /"confusion_matrices.png", dpi=300,
# #    bbox_inches="tight" )
# plt.close()
# #plt.show()


# # DataFrame with recall and support per fold
# df_cm_folds = pd.DataFrame(cm_fold_stats)
# for instrument, data in df_cm_folds.groupby("instrument"):
#     support_table = data.pivot(
#         index="type",
#         columns="fold",
#         values="support"
#     )

#     plt.figure(figsize=(8, 5))
#     sns.heatmap(
#         support_table,
#         annot=True,
#         fmt=".0f",
#         cmap="YlGnBu",
#         linewidths=0.5
#     )
#     plt.title(f"Test-set support per fold – {instrument}")
#     plt.xlabel("CV fold")
#     plt.ylabel("Substance class")
#     plt.tight_layout()
#     plt.savefig(PROJECT_ROOT / "figs" /f"support_heatmap_{instrument}.png", dpi=300)
#     plt.close()

# # Mean and standard deviation per instrument and class
# df_cm_stats = (
#     df_cm_folds
#     .groupby(["instrument", "type"])["recall"]
#     .agg(["mean", "std"])
#     .reset_index()
# )


# # Transposed DataFrame: instrument | mean | std
# df_cm_wide = df_cm_stats.pivot(
#     index="type",
#     columns="instrument",
#     values=["mean", "std"]
# )

# df_cm_wide = df_cm_wide.swaplevel(0, 1, axis=1)
# df_cm_wide = df_cm_wide.sort_index(axis=1, level=0)

# # with pd.option_context('display.max_rows', None,
# #                        'display.max_columns', None,
# #                        'display.precision', 3,
# #                        ):
# #     print(df_cm_wide)