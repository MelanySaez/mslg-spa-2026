"""Shim a enfoque7/data_loader.py (mismo split y seed que 7.1) + cargador de test set oficial."""

import csv
import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "data_loader.py")
_spec = importlib.util.spec_from_file_location("enfoque7_data_loader", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_dataset = _mod.load_dataset
split_dataset = _mod.split_dataset


def load_test(path: str, source_col: str) -> list:
    """Lee TSV oficial de test (ID + source_col), preserva orden de archivo.

    Retorna lista de dicts {id, source}. El orden es CRÍTICO: la submission
    debe tener una línea por instancia en el mismo orden que el archivo.
    """
    items = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = row.get("ID")
            src = row.get(source_col)
            if not rid or not src:
                continue
            items.append({"id": rid.strip(), "source": src.strip()})
    return items
