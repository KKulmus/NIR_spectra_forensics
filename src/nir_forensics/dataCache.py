# dataCache.py
"""Erzeugt replikat-gemittelte Spektren aus den Rohdateien 
(einmal rechnen, immer lesen).

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
PREPROCESSING_VERSION = "3" 

def build_path(rd: ReadData, AVG_DIR = AVG_DIR, parquet = True) -> str:
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
                "preprocessing_version": PREPROCESSING_VERSION} #händisch hochgezählt
    avg.to_parquet(out)
    with open(out_json, "w") as file:
        json.dump(manifest, file)
    
    return out

def load_avg(rd: ReadData, instrument) -> pd.DataFrame:
    out = build_path(rd, AVG_DIR, parquet = True)
    if out.exists():
        df = pd.read_parquet(out)
    else:
        build_all(instrument)
        out = build_path(rd, AVG_DIR, parquet = True)
        df = pd.read_parquet(out) 
    return df

def build_all(instrument = None):
    """Build averaged parquet files for all raw Excel files.
    
    Args:
        instrument: Optional filter ('scio', 'micronir', etc.). If None, process all.
    """
    files = sorted(RAW_DIR.glob("*.xlsx"))
        
    if instrument is not None: 
        files = [f for f in files if instrument.upper() in f.stem.upper().split('_')[-1]]
    rebuilt_count = 0
    for file in files:
        rd = ReadData.load(file.name)
        parquet_path = build_path(rd, parquet = True)
        manifest_path = build_path(rd, parquet = False)
       
        needs_rebuild = False
        if not parquet_path.exists():
            needs_rebuild = True
        elif manifest_path.exists():
           with open(manifest_path, 'r') as f:
                manifest_dict = json.load(f)
           if manifest_dict.get("preprocessing_version") != PREPROCESSING_VERSION:
               needs_rebuild = True
        if needs_rebuild:
            build_avg_file(rd)
            rebuilt_count += 1
    print(f"Processed {len(files)} files, rebuilt {rebuilt_count}")
        
            
       
                
            
    
    

if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("*.xlsx"))
    for file in files:
        rd = ReadData.load(file.name)
        build_avg_file(rd)
        df = load_avg(rd)
    print(df.head())
