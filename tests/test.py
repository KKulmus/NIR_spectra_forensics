import unittest
from dataView import ReadData
import numpy as np

class TestNormalizeCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Lädt die Dateien EINMAL für alle Tests (Cache macht's billig)."""
        cls.rd_t   = ReadData.load("T_Scio.xlsx")
        cls.rd_pam = ReadData.load("PAM_Scio.xlsx")

    def _code_map(self, rd):
        """Helper: Roher Code -> normalisierter Code."""
        raw  = rd.df.iloc[:, 0].astype(str).str.strip()
        return dict(zip(raw, rd.df["code_norm"]))

    def test_letters_and_digits_survive(self):
        """Normale Codes bleiben unverändert."""
        m = self._code_map(self.rd_t)
        for raw, expected in [("T0", "T0"), ("T10", "T10"),
                              ("T101", "T101"), ("T150", "T150")]:
            with self.subTest(code=raw):
                self.assertEqual(m.get(raw), expected)

    def test_leading_zeros_are_stripped(self):
        """PAM001 -> PAM1: Konvention aus den Scio-Dateien."""
        m = self._code_map(self.rd_pam)
        actual_keys = sorted(m.keys())
        # Diagnose-Hilfe: zeigt dir im Fehlerfall, welche Keys existieren
        self.assertIn("PAM001", actual_keys,
                      msg=f"'PAM001' nicht in Codespalte! Vorhandene Keys: {actual_keys[:15]}...")
        self.assertEqual(m["PAM001"], "PAM1")

    def test_zero_only_digits_do_not_vanish(self):
        """PAM000 -> PAM0: Ziffer '0' darf nicht verschwinden."""
        m = self._code_map(self.rd_pam)
        self.assertEqual(m["PAM000"], "PAM0")
    def test_get_sample_returns_numeric_spectra(self):
        """Regression: nur spectral_cols sind numerisch; code_norm scheidet aus dem Spektrum aus."""
        rd = ReadData.load("T_Scio.xlsx")
        sample = rd.get_sample("T0")
        vals = sample[rd.spectral_cols]
        self.assertTrue(vals.dtypes.map(lambda d: np.issubdtype(d, np.floating)).all())



if __name__ == "__main__":
    unittest.main(verbosity=2)
