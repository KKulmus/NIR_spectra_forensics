#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:06:54 2026

@author: kathrin
"""

import unittest


from nir_forensics.paths import FIXTURE_DIR
from nir_forensics.modeling.split import split_dataset, split_features_target, create_cv_splits
from nir_forensics.modeling.dataset import build_modeling_dataset
from pandas.testing import assert_frame_equal
import pandas as pd
import re
import numpy as np



dataset = build_modeling_dataset(instrument = "MicroNIR",
                            metadata_path = FIXTURE_DIR / "metadata.xlsx",
                            avg_dir = FIXTURE_DIR,
                            expected_rows = {"MicroNIR":4})



class TestSplit(unittest.TestCase):

    def test_split_completeness(self):
        """is split disjunct and complete""" 
        train, test = split_dataset(dataset)
        self.assertEqual(test.shape[0]+train.shape[0], dataset.shape[0])
        self.assertEqual(len(list(set(test["group"].unique().tolist()).intersection(set(train["group"].unique().tolist())))), 0)
        
    def test_split_reproducibility(self):
        """two calls of split_dataset yield same test/train-split"""
        train1, test1 = split_dataset(dataset) 
        train2, test2 = split_dataset(dataset) 
        assert_frame_equal(train1, train2)
        assert_frame_equal(test1, test2)
        
    def test_split_features(self):
        """Feature have no columns target and group """
        train, test = split_dataset(dataset)
        train_features, train_target, test_features, test_target = split_features_target(train, test)
        self.assertNotIn("type", train_features.columns.tolist())
        self.assertNotIn("type", test_features.columns.tolist())
        self.assertNotIn("group", train_features.columns.tolist())
        self.assertNotIn("group", test_features.columns.tolist())
    
    def test_split_targets(self):
        train, test = split_dataset(dataset)
        train_features, train_target, test_features, test_target = split_features_target(train, test)
        self.assertEqual(type(train_target), pd.Series)
        self.assertEqual(type(test_target), pd.Series)
        
        self.assertEqual(train_target.name, 'type')
        self.assertEqual(test_target.name, 'type')
        
        train_list = train_target.tolist()
        test_list = test_target.tolist()
        result = train_list + test_list
        result = ["" if pd.isna(x) else x for x in result]
        expected = ['other', 'cocaine', '' ,'cocaine']
        self.assertEqual(sorted(result), sorted(expected))
        
class TestCVSplit(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load data-files once (using the cache)"""
        cls.df = pd.read_excel(FIXTURE_DIR / "cv_folding_fixture.xlsx") 
    
    def test__cv_split_number_of_folds(self):
        """Tests whether number of folds == n_split = 5"""
        train_index, test_index = create_cv_splits(self.df, n_splits = 5, shuffle = True, random_state = 42)
        self.assertEqual(len(train_index),5)
        self.assertEqual(len(test_index),5)
    
    def test_cv_split_t_p_separation(self):
        """Tests whether t and p-samples with same group are both in train or in test set"""
        train_index, test_index = create_cv_splits(self.df, n_splits = 5, shuffle = True, random_state = 42)
        
        tp_list_train = []
        for i in range(len(train_index)):
            groups = self.df.iloc[train_index[i]]["group"]
            tp_list_train.append(groups[groups.str.startswith("E")])


        tp_list_test = []
        for i in range(len(test_index)):
            groups = self.df.iloc[test_index[i]]["group"]
            tp_list_test.append(groups[groups.str.startswith("E")])
        
        for l in range(len(tp_list_test)):
            self.assertEqual(len(set(tp_list_test[l])&set(tp_list_train[l])),0)
    
    def test_cv_split_reproducibility(self):
        """Tests whether setting a random state yield the same folds"""
        train_index_1, test_index_1 = create_cv_splits(self.df, n_splits = 5, shuffle = True, random_state = 42)
        train_index_2, test_index_2 = create_cv_splits(self.df, n_splits = 5, shuffle = True, random_state = 42)
        for l in range(len(train_index_1)):
            np.testing.assert_array_equal(train_index_1[l], train_index_2[l])
        for l in range(len(test_index_1)):
            np.testing.assert_array_equal(test_index_1[l], test_index_2[l])
            
    def test_cv_split_test_train_disjoint(self):
        """Test whether the test-train indices of each fold are disjoint"""
        train_index, test_index = create_cv_splits(self.df, n_splits = 5, shuffle = True, random_state = 42)
        for l in range(len(train_index)):
            self.assertEqual(len(set(train_index[l]) & set(test_index[l])), 0)
        
        