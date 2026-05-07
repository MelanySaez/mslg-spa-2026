"""Shim a enfoque3/post_processor.py"""
import importlib.util
import os
_path = os.path.join(os.path.dirname(__file__), "..", "post_processor.py")
_spec = importlib.util.spec_from_file_location("e3_post_processor", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
for k, v in _mod.__dict__.items():
    if not k.startswith("_"):
        locals()[k] = v
