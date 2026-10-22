#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:54:26 2026

@author: kathrin
"""

from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

def split_dataset(dataset: pd.DataFrame, n_splits = 1, test_size = 0.2, random_state = 42):
    """Performs a train-test split of a dataset of one instrument 
    holds the groups together. Important for T and P samples to avoid data leakage
    Args: 
        dataset (pd.DataFrame) : preprocessed dataframe of a single instrument (output from build_modeling_data) 
        n_splits (int): number of train-test splits for cross-validation, defaults to 1
        test_size (float): portion of groups in test_data w.r.t. total size in percent/100, defaults to 0.2
        random_state (int): arbitrary number to guarantee reproducibility, defaults to 42
    Returns: 
        Two DataFrames with train and test data, together they have alle rows of dataset and are disjunct
    """
    
    gss = GroupShuffleSplit(n_splits = n_splits, 
                            test_size = test_size, 
                            random_state = random_state)

    train_index, test_index = next(gss.split(dataset, groups=dataset["group"]))
    train_data = dataset.iloc[train_index]
    test_data = dataset.iloc[test_index]
    
    return train_data, test_data

def split_features_target(trainset: pd.DataFrame, testset: pd.DataFrame, target = "type"):
    """Separates target and features in given trainset and testset 
        Args: 
            trainset (pd.DataFrame) : Dataset for training
            testset (pd.DataFrame) : Dataset for testing
            target (str) : target column in train- and testset, defaults to type
        Returns:
            Tuple with four entries: train_features, train_target, test_features, test_target
    """
    if target not in trainset.columns.tolist():
        raise ValueError(f"{target} is no column name of trainset. Typo?")
    if target not in testset.columns.tolist():
        raise ValueError(f"{target} is no column name of testset. Typo?")
    train_features = trainset.drop([target,"group"], axis = 1)
    test_features = testset.drop([target,"group"], axis = 1)

    train_target = trainset[target]
    test_target = testset[target]
    
    return train_features, train_target, test_features, test_target
