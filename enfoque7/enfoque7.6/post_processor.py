"""Shim a enfoque7.2/post_processor.py (preserva acentos/capitalización SPA)."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque7.2",
                     "post_processor.py")
_spec = importlib.util.spec_from_file_location("enfoque72_post_processor", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean = _mod.clean
