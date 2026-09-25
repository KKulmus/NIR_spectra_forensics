# dataCache.py
"""Erzeugt replikat-gemittelte Spektren aus den Rohdateien (einmal rechnen, immer lesen).

Erzeugt: dataset/datafiles_avg/<name>_avg.parquet
Aufruf:  python dataCache.py         (oder import + build_all())
"""
from pathlib import Path
import pandas as pd
from nir_forensics.dataView import ReadData
import json
from datetime import datetime
from nir_forensics.paths import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "dataset" / "datafiles_raw"
AVG_DIR = PROJECT_ROOT / "dataset" / "datafiles_avg"

def build_path(rd: ReadData, AVG_DIR, parquet = True) -> str:
    if parquet:
        return AVG_DIR / rd.filename.replace(".xlsx", "_avg.parquet")
    else:
        return AVG_DIR/ rd.filename.replace(".xlsx", "_manifest.json")

def build_avg_file(rd: ReadData, out_dir="dataset/datafiles_avg") -> Path:
    """Replicate-averaged, transformierte Spektren (Absorbanz) -> Parquet.
        .json-manifest, das Zeit, Quelle und Prepossing Version speichert
    Nutzt ReadData.get_avg_spectrum(), damit Transform-Reihenfolge (-log vor
    Mittelung, nur beim SCiO) garantiert identisch zum interaktiven Pfad ist.
    """
    AVG_DIR.mkdir(exist_ok=True)
    rows = {code: rd.get_avg_spectrum(code) for code in rd.df["code_norm"].unique()}
    avg = pd.DataFrame(rows).T          # Zeilen: code_norm, Spalten: Wellenlängen
    out = build_path(rd, AVG_DIR, parquet = True)
    out_json = build_path(rd, AVG_DIR, parquet = False)
    manifest = {"created" : datetime.now().isoformat(),
                "source": rd.filename,
                "preprocessing_version": "1"} #händisch hochgezählt
    avg.to_parquet(out)
    with open(out_json, "w") as file:
        json.dump(manifest, file)
    
    return out

def load_avg(rd: ReadData) -> pd.DataFrame:
    out = build_path(rd, AVG_DIR, parquet = True)
    df = pd.read_parquet(out)
    return df

if __name__ == "__main__":     
    rd = ReadData("T_Scio.xlsx")
    build_avg_file(rd)
    df = load_avg("T_Scio.xlsx")
    print(df.head())
