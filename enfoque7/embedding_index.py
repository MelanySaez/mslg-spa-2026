"""Índice de embeddings — reutiliza enfoque6/embedding_index.py.

Requerido solo para la variante few_shot_diverse (selección por k-means
sobre embeddings del pool).
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque6", "embedding_index.py")
_spec = importlib.util.spec_from_file_location("enfoque6_embedding_index", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EmbeddingIndex = _mod.EmbeddingIndex
