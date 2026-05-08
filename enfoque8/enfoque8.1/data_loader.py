"""Carga de dataset y test set — shim a enfoque7.2/data_loader.py.

Reutiliza el mismo split (pool/val) que enfoque7.2 para comparabilidad directa.
Incluye load_test() para leer MSLG2SPA_test.txt preservando orden de archivo.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "enfoque7", "enfoque7.2", "data_loader.py")
_spec = importlib.util.spec_from_file_location("e72_data_loader", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_dataset = _mod.load_dataset
split_dataset = _mod.split_dataset
load_test = _mod.load_test
