"""Carga y split del dataset MSLG_SPA_train.txt."""

import csv
import random

import config


def load_dataset(path=None):
    """Lee el TSV y retorna lista de dicts {id, spa, mslg}."""
    path = path or config.DATASET_PATH
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({
                "id": row["ID"],
                "spa": row["SPA"].strip(),
                "mslg": row["MSLG"].strip(),
            })
    return data


def split_dataset(data=None):
    """Shuffle con seed fija y split en pool (400) + validación (90)."""
    if data is None:
        data = load_dataset()

    shuffled = list(data)
    random.seed(config.RANDOM_SEED)
    random.shuffle(shuffled)

    pool = shuffled[: config.TRAIN_SPLIT]
    val = shuffled[config.TRAIN_SPLIT : config.TRAIN_SPLIT + config.VAL_SPLIT]

    print(f"Dataset cargado: {len(data)} pares total")
    print(f"  Pool de ejemplos: {len(pool)}")
    print(f"  Validación:       {len(val)}")
    return pool, val


if __name__ == "__main__":
    pool, val = split_dataset()
    print(f"\nPrimer ejemplo del pool: {pool[0]}")
    print(f"Primer ejemplo de val:  {val[0]}")
