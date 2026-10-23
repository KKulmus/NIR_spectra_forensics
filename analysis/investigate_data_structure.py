#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate the structure of the NIR modeling datasets and diagnose the
historical NeoSpectra wavelength-column issue.

The investigation found that the NeoSpectra modeling dataset initially
contained duplicate wavelength columns caused by tiny floating-point
differences in wavelength labels across raw files. Rounding wavelength
labels to eight decimal places resolved the column mismatch.

This script is retained as an analysis record. The regression test in
tests/test_modeling.py checks the corrected dataset structure.
"""

import numpy as np
import pandas as pd

from nir_forensics.dataView import ReadData
from nir_forensics.modeling.dataset import build_modeling_dataset
from nir_forensics.paths import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "dataset" / "datafiles_raw"


def get_spectral_columns(df: pd.DataFrame) -> list:
    """Return columns whose labels are numeric wavelength values."""
    return [column for column in df.columns if isinstance(column, (int, float))]


def inspect_dataset_overview() -> None:
    """Print sample counts and class distributions for each instrument."""
    instruments = {
        "SCiO": "scio",
        "ASD": "ASD",
        "NeoSpectra": "neospectra",
        "MicroNIR": "micronir",
        "NIRone": "NIRONE",
    }

    datasets = {
        name: build_modeling_dataset(instrument_key)
        for name, instrument_key in instruments.items()
    }

    # Compare sample counts and the distribution of target classes.
    print("Sample counts by instrument:")
    print(pd.Series({name: len(df) for name, df in datasets.items()}))

    class_counts = pd.DataFrame({
        name: df["type"].value_counts()
        for name, df in datasets.items()
    }).fillna(0).astype(int)

    print("\nClass counts by instrument:")
    print(class_counts)

    class_proportions = (
        class_counts.div(class_counts.sum(axis=0), axis=1) * 100
    )
    print("\nClass proportions by instrument (%):")
    print(class_proportions.round(1))


def inspect_shared_sample_codes() -> None:
    """Check whether NeoSpectra and MicroNIR contain the same sample codes."""
    neo_df = build_modeling_dataset("neospectra")
    micro_df = build_modeling_dataset("micronir")

    neo_codes = set(neo_df["code"])
    micro_codes = set(micro_df["code"])

    print("\nSample-code comparison: NeoSpectra vs MicroNIR")
    print("NeoSpectra unique codes:", len(neo_codes))
    print("MicroNIR unique codes:", len(micro_codes))
    print("Codes shared by both datasets:", len(neo_codes & micro_codes))
    print("Codes only in NeoSpectra:", sorted(neo_codes - micro_codes))
    print("Codes only in MicroNIR:", sorted(micro_codes - neo_codes))


def investigate_neospectra_wavelength_grids() -> None:
    """Compare wavelength grids in the raw NeoSpectra files."""
    files = sorted(RAW_DIR.glob("*_NeoSpectra.xlsx"))
    if not files:
        print("\nNo raw NeoSpectra files found; skipping wavelength-grid analysis.")
        return

    reference_file = files[0]
    reference = ReadData.load(reference_file.name).wl

    print("\nHistorical NeoSpectra wavelength-grid investigation")
    print("Reference file:", reference_file.name)
    print("Number of reference wavelengths:", len(reference))

    # Compare each raw file with the reference wavelength grid.
    for file in files[1:]:
        wavelengths = ReadData.load(file.name).wl

        if len(wavelengths) != len(reference):
            print(
                f"{file.name}: {len(wavelengths)} wavelengths; "
                f"expected {len(reference)}"
            )
            continue

        max_difference = np.max(np.abs(wavelengths - reference))
        same_after_rounding = np.array_equal(
            np.round(wavelengths, 8),
            np.round(reference, 8),
        )
        print(
            f"{file.name}: maximum absolute difference = "
            f"{max_difference:.3e}; identical after rounding to 8 decimals = "
            f"{same_after_rounding}"
        )


def validate_current_neospectra_dataset() -> None:
    """Report the current NeoSpectra dataset shape and missing spectral values."""
    df = build_modeling_dataset("neospectra")
    spectral_columns = get_spectral_columns(df)
    spectral_values = df[spectral_columns]

    print("\nCurrent NeoSpectra modeling dataset")
    print("Dataset shape:", df.shape)
    print("Number of spectral columns:", len(spectral_columns))
    print("Missing spectral values:", int(spectral_values.isna().sum().sum()))


if __name__ == "__main__":
    inspect_dataset_overview()
    inspect_shared_sample_codes()
    investigate_neospectra_wavelength_grids()
    validate_current_neospectra_dataset()