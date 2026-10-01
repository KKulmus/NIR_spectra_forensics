import unittest


from nir_forensics.dataView import ReadData
import numpy as np
import pandas as pd
from nir_forensics.dataCache import load_avg
from nir_forensics.paths import FIXTURE_DIR, PROJECT_ROOT

HAS_RAW_DATA = (PROJECT_ROOT / "dataset" / "datafiles_raw" / "T_Scio.xlsx").exists() 

@unittest.skipUnless(HAS_RAW_DATA, "Lizenzdaten (Kranenburg, DOI 10.21942/uva.21252300) nicht verfügbar")
class TestNormalizeCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load data-files once (using the cache)"""
        cls.rd_t   = ReadData.load("T_Scio.xlsx")
        cls.rd_pam = ReadData.load("PAM_Scio.xlsx")

    def _code_map(self, rd):
        """Helper: Raw Code -> noramlized code.
           Example: 
               input: T1, PAM001
               output {T1:T1, PAM001:PAM1}
             
        """
        raw  = rd.df.iloc[:, 0].astype(str).str.strip()
        return dict(zip(raw, rd.df["code_norm"]))

    def test_letters_and_digits_survive(self):
        """Genuin codes are unchanged"""
        m = self._code_map(self.rd_t)
        for raw, expected in [("T0", "T0"), ("T10", "T10"),
                              ("T101", "T101"), ("T150", "T150")]:
            with self.subTest(code=raw):
                self.assertEqual(m.get(raw), expected)

    def test_leading_zeros_are_stripped(self):
        """PAM001 -> PAM1: convention from scio files."""
        m = self._code_map(self.rd_pam)
        actual_keys = sorted(m.keys())
        # Diagnose-Hilfe: zeigt dir im Fehlerfall, welche Keys existieren
        self.assertIn("PAM001", actual_keys,
                      msg=f"'PAM001' not in code-column! Available Keys: {actual_keys[:15]}...")
        self.assertEqual(m["PAM001"], "PAM1")

    def test_zero_only_digits_do_not_vanish(self):
        """PAM000 -> PAM0: '0' may not get lost."""
        m = self._code_map(self.rd_pam)
        self.assertEqual(m["PAM000"], "PAM0")
    def test_get_sample_returns_numeric_spectra(self):
        """Regression: only spectral_cols are numeric; code_norm (string) not in spectrum."""
        rd = ReadData.load("T_Scio.xlsx")
        sample = rd.get_sample("T0")
        vals = sample[rd.spectral_cols]
        self.assertTrue(vals.dtypes.map(lambda d: np.issubdtype(d, np.floating)).all())

@unittest.skipUnless(HAS_RAW_DATA, "Lizenzdaten (Kranenburg, DOI 10.21942/uva.21252300) nicht verfügbar")
class TestAvgCache(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load data ONCE for all Tests"""
        cls.rd_t = ReadData.load("T_Scio.xlsx")
        cls.avg_file_t101 = cls.rd_t.get_avg_spectrum("T101")
          
        
    def test_compare_liveAvg_cacheAvg(self):
        """AVG for Scio: live-vs. Cache -> exists cache? is it equal to live?"""
        cache_avg = load_avg(self.rd_t).loc["T101"]
        pd.testing.assert_series_equal(cache_avg, self.avg_file_t101, check_names = False)
            
        
        
class TestLoaderFixture(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load data-files once (using the cache)"""
        cls.rd_t   = ReadData.load(filename = "T_Scio_fixture.xlsx", path = FIXTURE_DIR )
        
    def _code_map(self, rd):
        """Helper: Raw Code -> noramlized code.
           Example: 
               input: T1, PAM001
               output {T1:T1, PAM001:PAM1}
             
        """
        raw  = rd.df.iloc[:, 0].astype(str).str.strip()
        return dict(zip(raw, rd.df["code_norm"]))

    def test_letters_and_digits_survive(self):
        """Genuin codes are unchanged"""
        m = self._code_map(self.rd_t)
        for raw, expected in [("T0", "T0"), ("T10", "T10"),
                              ("T101", "T101"), ("T150", "T150")]:
            with self.subTest(code=raw):
                self.assertEqual(m.get(raw), expected)

    def test_leading_zeros_are_stripped(self):
        """PAM001 -> PAM1: convention from scio files."""
        m = self._code_map(self.rd_t)
        actual_keys = sorted(m.keys())
        # Diagnose-Hilfe: zeigt dir im Fehlerfall, welche Keys existieren
        self.assertIn("PAM001", actual_keys,
                      msg=f"'PAM001' not in code-column! Available Keys: {actual_keys[:15]}...")
        self.assertEqual(m["PAM001"], "PAM1")

    def test_zero_only_digits_do_not_vanish(self):
        """PAM000 -> PAM0: '0' may not get lost."""
        m = self._code_map(self.rd_t)
        self.assertEqual(m["PAM000"], "PAM0")
        
    def test_get_sample_returns_numeric_spectra(self):
        """Regression: only spectral_cols are numeric; code_norm (string) not in spectrum."""
        sample = self.rd_t.get_sample("T0")
        vals = sample[self.rd_t.spectral_cols]
        self.assertTrue(vals.dtypes.map(lambda d: np.issubdtype(d, np.floating)).all())

    
    def test_get_available_samples(self):
        """Replicats only returned once?"""
        self.assertEqual(self.rd_t.get_available_samples(), ['T0','T10','T101','T150', 'PAM001', 'PAM000'])
        
    def test_compare_manualAvg_functionAvg(self):
        """AVG for Scio: manual vs. function are they "equal"?"""
        avg_from_func = self.rd_t.get_avg_spectrum("T101").iloc[0]
        manual = 0.3952071834766561
        self.assertAlmostEqual(avg_from_func,manual, delta=1e-12)
            
        

if __name__ == "__main__":
    unittest.main(verbosity=2)
