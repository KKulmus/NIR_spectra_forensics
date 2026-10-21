#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:21:25 2026

@author: kathrin

Test suite for function: groups.get_codes().
Includes code validation and group assignment for valid and invalid entries.
Checks whether input data remains unchanged. 
"""

import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from nir_forensics.modeling.groups import get_codes

#Dataframe with valid entries
dic_valid = {"code":["T7", "P7", "T8", "P8", "Y7"],
       "set": ["T",  "P", "T", "P",  "Y"],
       "nr" : ["7", "7", "8", "8",  "7"]
       }
df_valid = pd.DataFrame(dic_valid)

# DataFrame with one enty with code != set + nr
dic_unvalid_nr =  {"code":["T7", "X3"],
       "set": ["T",  "X"],
       "nr" : ["7", "4"]
       }
df_invalid_nr = pd.DataFrame(dic_unvalid_nr)

# DataFrame with empty strings
dic_missing_val = {"code":["T7", "", "T7", "X3"],
       "set": ["T",  "X", "",  "X"],
       "nr" : ["7", "3", "7", ""]
       }
df_missing_val = pd.DataFrame(dic_missing_val)

# DataFrame with None as entry
dic_none_val = {"code":["T7", "X3"],
       "set": ["T",  "X"],
       "nr" : ["7", None]
       }
df_none_val = pd.DataFrame(dic_none_val)

# Empty DataFrame
dic_empty = {"code":[],
       "set": [],
       "nr" : []
       }
df_empty = pd.DataFrame(dic_empty)


class TestGroups(unittest.TestCase):
    def test_valid_groups(self):
        """Valid codes receive the expected group assignments."""
        df_result = get_codes(df_valid)
        expected_groups = ["E7", "E7", "E8", "E8", "Y7"]
        self.assertEqual(df_result["group"].tolist(), expected_groups)
        
    def test_invalid_nr(self):
        """Invalid codes are excluded and trigger a warning"""
        with self.assertWarnsRegex(
            UserWarning,
            r"Manual inspection is required "
            "before using them." # The regex to match the message
        ):
            df_result = get_codes(df_invalid_nr)
            expected_groups = ["E7"]
            self.assertEqual(df_result["group"].tolist(), expected_groups)
            
    def test_get_codes_does_not_change_df_valid(self):
        """get_codes() does not modify original dataframe"""
        df_valid_copy = df_valid.copy()
        get_codes(df_valid)
        assert_frame_equal(df_valid_copy, df_valid)
        
    def test_missing_values(self):
        """Rows with empty strings are excluded"""
        df_result = get_codes(df_missing_val)
        expected = {"code":["T7"],
               "set": ["T"],
               "nr" : ["7"],
               "group" : ["E7"]}
        df_expected = pd.DataFrame(expected)
        assert_frame_equal(df_result, df_expected)
        
    def test_none_values(self):
        """Rows with None are excluded"""
        df_result = get_codes(df_none_val)
        expected = {"code":["T7"],
                "set": ["T"],
                "nr" : ["7"],
                "group" : ["E7"]}
        df_expected = pd.DataFrame(expected)
        assert_frame_equal(df_result, df_expected)   
        
    def test_empty_dataframe(self):
        """get_codes(empty_DataFrame) returns empty DataFrame with new column group """
        df_code_empty = get_codes(df_empty)
        df_res = df_empty.copy()
        df_res["group"] = []
        df_res["group"] = df_res["group"].astype(object)
        assert_frame_equal(df_code_empty, df_res)
        
        
        
        
        
        
        
        
        
        
        
        