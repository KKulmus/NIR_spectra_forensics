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







class TestModelingData(unittest.TestCase):

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
        # self.assertWarns(Warning, build_modeling_dataset,instrument = "MicroNIR",
        #                             metadata_path = FIXTURE_DIR / "metadata.xlsx",
        #                             avg_dir = FIXTURE_DIR,
        #                             expected_rows = {"MicroNIR":4})
        with self.assertWarnsRegex(
            UserWarning,
            r"spectra without metadata: {'X12'}" # The regex to match the message
        ):
            build_modeling_dataset(instrument = "MicroNIR",
                                         metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                         avg_dir = FIXTURE_DIR,
                                         expected_rows = {"MicroNIR":4})
    def test_expected_rows(self):
        """expected rows and returned rows are not equal: raise ValueError"""
        self.assertRaises(ValueError, build_modeling_dataset,instrument = "MicroNIR",
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
        df_parquet.columns = pd.to_numeric(df_parquet.columns)
        
        
        numerical_cols = [c for c in df_parquet.columns if isinstance(c, (int, float))]
        self.assertTrue(all(isinstance(c, float) for c in numerical_cols),
                        msg=f"Wavelengths should be float, got: {[(c, type(c).__name__) for c in numerical_cols]}"
                            )
       
        self.assertEqual(df.shape,(4,10))
        self.assertNotIn("X12", df["code"].tolist())
        self.assertEqual(list(df[df["code"]=="T1"]["type"])[0],"other")
        
        self.assertAlmostEqual(list(df.loc[df["code"]=="PAM7", 740.0])[0],
                               df_parquet.loc["PAM7", 740.0], delta=1e-12)
       
        
    def test_column_group(self):
        """Test whether column group is added"""
        df = build_modeling_dataset(instrument = "MicroNIR",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":4})

        column_list = df.columns.tolist()
        self.assertIn("group", column_list)
        
    def test_group_assignement(self):
        """Group assignement okay? T1 -> E1"""
        df = build_modeling_dataset(instrument = "MicroNIR",
                                    metadata_path = FIXTURE_DIR / "metadata.xlsx",
                                    avg_dir = FIXTURE_DIR,
                                    expected_rows = {"MicroNIR":4})

        expected_groups = ['E1', 'E2', 'E1', 'PAM7']
        expected_codes = ['T1', 'T2', 'P1', 'PAM7']
        self.assertEqual(df["group"].tolist(), expected_groups)
        self.assertEqual(df["code"].tolist(), expected_codes)

         
 