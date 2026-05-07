"""Shim a enfoque3/embedding_index.py"""
import importlib.util
import os
_path = os.path.join(os.path.dirname(__file__), "..", "embedding_index.py")
_spec = importlib.util.spec_from_file_location("e3_embedding_index", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
for k, v in _mod.__dict__.items():
    if not k.startswith("_"):
        locals()[k] = v
