#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:06:54 2026

@author: kathrin
"""

import unittest


from nir_forensics.paths import FIXTURE_DIR
from nir_forensics.modeling.split import split_dataset, split_features_target
from nir_forensics.modeling.dataset import build_modeling_dataset
from pandas.testing import assert_frame_equal
import pandas as pd



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
        
        