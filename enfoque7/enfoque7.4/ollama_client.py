"""Shim a enfoque7.3/ollama_client.py.

El cliente acepta `(system_prompt, user_prompt)` y desactiva el modo de
razonamiento (`think=False`) para que deepseek-r1:32b devuelva la traducción
directa sin emitir el bloque <think>...</think>.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque7.3",
                     "ollama_client.py")
_spec = importlib.util.spec_from_file_location("enfoque73_ollama_client", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

translate = _mod.translate
