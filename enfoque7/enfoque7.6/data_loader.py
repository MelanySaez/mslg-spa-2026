"""Shim a enfoque7/data_loader.py (mismo split y seed que 7.2)."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "data_loader.py")
_spec = importlib.util.spec_from_file_location("enfoque7_data_loader", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_dataset = _mod.load_dataset
split_dataset = _mod.split_dataset
