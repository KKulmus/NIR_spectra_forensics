# dataCache.py
"""Erzeugt replikat-gemittelte Spektren aus den Rohdateien (einmal rechnen, immer lesen).

Erzeugt: dataset/datafiles_avg/<name>_avg.parquet
Aufruf:  python dataCache.py         (oder import + build_all())
"""
from pathlib import Path
import pandas as pd
from dataView import ReadData

RAW_DIR  = Path("dataset/datafiles_raw")
AVG_DIR  = Path("dataset/datafiles_avg")

def build_avg_file(rd: ReadData, out_dir="dataset/datafiles_avg") -> Path:
    """Replicate-averaged, transformierte Spektren (Absorbanz) -> Parquet.
    
    Nutzt ReadData.get_avg_spectrum(), damit Transform-Reihenfolge (-log vor
    Mittelung, nur beim SCiO) garantiert identisch zum interaktiven Pfad ist.
    """
    AVG_DIR.mkdir(exist_ok=True)
    rows = {code: rd.get_avg_spectrum(code) for code in rd.df["code_norm"].unique()}
    avg = pd.DataFrame(rows).T          # Zeilen: code_norm, Spalten: Wellenlängen
    out = AVG_DIR / rd.filename.replace(".xlsx", "_avg.parquet")
    avg.to_parquet(out)
    return out

def load_avg(filename: str) -> pd.DataFrame:
    out = AVG_DIR / rd.filename.replace(".xlsx", "_avg.parquet")
    df = pd.read_parquet(out)
    return df
    
rd = ReadData("T_Scio.xlsx")
build_avg_file(rd)
df = load_avg("T_Scio.xlsx")
print(df.head())
