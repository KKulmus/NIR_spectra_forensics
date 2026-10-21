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
