#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:36:46 2026

@author: kathrin
"""

import unittest

import pandas as pd
from nir_forensics.paths import FIXTURE_DIR
from nir_forensics.modeling.dataset import build_modeling_dataset







class TestNormalizeCode(unittest.TestCase):

    def test_invalid_instrument(self):
        """Instrument = foo: raise valueError"""
        
        self.assertRaises(ValueError, build_modeling_dataset, instrument = "foo",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":4})
        
        
    def test_parquet_not_exist(self):
        """parquet file for instrument is not available: raise ValueError"""
        self.assertRaises(ValueError, build_modeling_dataset,instrument = "Scio",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":4})
        
    def test_missing(self):
        """no metadata for a spectrum: warnings.warn"""
        self.assertWarns(Warning, build_modeling_dataset,instrument = "MicroNIR",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":4})
    def test_expected_rows(self):
        """expected rows and returned rows are not equal: raise ValueError"""
        self.assertRaises(ValueError, build_modeling_dataset,instrument = "Scio",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":14})
    def test_positive(self):
        """create dataframe correctly, no error"""
        df = build_modeling_dataset(instrument = "MicroNIR",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":4})
        df_parquet = pd.read_parquet(FIXTURE_DIR / "X_MicroNIR_avg.parquet")
        pam7Val = df_parquet.loc["PAM7", 740.0]
        self.assertEqual(df.shape,(4,7))
        self.assertNotIn("X12", df["code"].tolist())
        self.assertEqual(list(df[df["code"]=="T1"]["type"])[0],"other")
        
        self.assertAlmostEqual(list(df.loc[df["code"]=="PAM7", 740.0])[0],pam7Val, delta=1e-12)
       
        
         
         
 