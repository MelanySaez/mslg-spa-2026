"""Shim a enfoque7.2/embedding_index.py (índice MSLG-indexed para sentido reverso)."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque7.2",
                     "embedding_index.py")
_spec = importlib.util.spec_from_file_location("enfoque72_embedding_index", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EmbeddingIndex = _mod.EmbeddingIndex
