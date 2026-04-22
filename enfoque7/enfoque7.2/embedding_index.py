"""Shim a enfoque7/embedding_index.py."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "embedding_index.py")
_spec = importlib.util.spec_from_file_location("enfoque7_embedding_index", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EmbeddingIndex = _mod.EmbeddingIndex
