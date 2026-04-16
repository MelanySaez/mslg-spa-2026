"""Prepara datasets para fine-tuning BART español→glosas (dos rondas).

Fuentes:
  - esp-lsm glosses corpus.csv  (3000 pares, root)
  - enfoque3/data/MSLG_SPA_train.txt  (490 pares, TSV)

Salida:
  - enfoque2/pruebas/data/round1/{train,val,test}.csv   (70/10/20)
  - enfoque2/pruebas/data/round2/{train,val}.csv         (400/90)
"""

import argparse
import json
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRUEBAS_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PRUEBAS_DIR, "data")
PROJECT_ROOT = os.path.dirname(os.path.dirname(PRUEBAS_DIR))

DEFAULT_ROUND1_SRC = os.path.join(PROJECT_ROOT, "esp-lsm glosses corpus.csv")
DEFAULT_ROUND2_SRC = os.path.join(PROJECT_ROOT, "enfoque3", "data", "MSLG_SPA_train.txt")


def load_round1_corpus(path):
    """Lee esp-lsm CSV, limpia BOM + cols vacías, renombra a source/target."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df[["esp", "lsm"]].dropna()
    df = df.rename(columns={"esp": "source", "lsm": "target"})
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")]
    return df.reset_index(drop=True)


def load_round2_tsv(path):
    """Lee MSLG_SPA_train.txt TSV, mapea SPA→source, MSLG→target."""
    df = pd.read_csv(path, sep="\t")
    df = df[["SPA", "MSLG"]].dropna()
    df = df.rename(columns={"SPA": "source", "MSLG": "target"})
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    return df.reset_index(drop=True)


def split_round1(df):
    df_trainval, df_test = train_test_split(
        df, test_size=0.20, random_state=SEED, shuffle=True
    )
    val_rel = 0.10 / 0.80
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_rel, random_state=SEED, shuffle=True
    )
    return df_train, df_val, df_test


def split_round2(df, train_n=400, val_n=90):
    df_train = df.iloc[:train_n].copy()
    df_val = df.iloc[train_n:train_n + val_n].copy()
    return df_train, df_val


def _save(df, out_dir, name):
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def prepare_round1(src_path):
    print(f"[R1] Cargando {src_path}")
    df = load_round1_corpus(src_path)
    print(f"[R1] Tras limpieza: {len(df)}")
    train, val, test = split_round1(df)
    print(f"[R1] Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    out_dir = os.path.join(DATA_DIR, "round1")
    os.makedirs(out_dir, exist_ok=True)
    _save(train, out_dir, "train")
    _save(val, out_dir, "val")
    _save(test, out_dir, "test")
    stats = {
        "total": len(df),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def prepare_round2(src_path, test_path=None):
    print(f"[R2] Cargando {src_path}")
    df = load_round2_tsv(src_path)
    print(f"[R2] Total: {len(df)}")
    train, val = split_round2(df)
    print(f"[R2] Train:{len(train)} Val:{len(val)}")
    out_dir = os.path.join(DATA_DIR, "round2")
    os.makedirs(out_dir, exist_ok=True)
    _save(train, out_dir, "train")
    _save(val, out_dir, "val")
    if test_path and os.path.exists(test_path):
        shutil.copy(test_path, os.path.join(out_dir, "test.csv"))
        print(f"[R2] Test externo copiado: {test_path}")
    else:
        print("[R2] Sin test externo. Coloca en data/round2/test.csv antes de evaluar.")
    stats = {"total": len(df), "train": len(train), "val": len(val)}
    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--round1-src", default=DEFAULT_ROUND1_SRC)
    p.add_argument("--round2-src", default=DEFAULT_ROUND2_SRC)
    p.add_argument("--round2-test", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    s1 = prepare_round1(args.round1_src)
    s2 = prepare_round2(args.round2_src, args.round2_test)
    print(f"\nR1 → train:{s1['train']} val:{s1['val']} test:{s1['test']}")
    print(f"R2 → train:{s2['train']} val:{s2['val']}")


if __name__ == "__main__":
    main()
