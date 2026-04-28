"""Shim a enfoque7/data_loader.py.

Reusa el split SPA/MSLG idéntico al de 7.1 (mismo seed, mismo pool, mismo val).
Esto garantiza que las métricas reversas se computan sobre los mismos IDs que
la baseline directa, permitiendo comparación uno-a-uno.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "data_loader.py")
_spec = importlib.util.spec_from_file_location("enfoque7_data_loader", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_dataset = _mod.load_dataset
split_dataset = _mod.split_dataset
