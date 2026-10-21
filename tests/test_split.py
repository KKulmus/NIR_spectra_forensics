#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:06:54 2026

@author: kathrin
"""

import unittest


from nir_forensics.paths import FIXTURE_DIR
from nir_forensics.modeling.split import split_dataset
from nir_forensics.modeling.dataset import build_modeling_dataset
from pandas.testing import assert_frame_equal



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