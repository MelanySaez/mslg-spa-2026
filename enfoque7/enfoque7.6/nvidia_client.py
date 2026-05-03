"""Shim a enfoque7.5/nvidia_client.py.

Cliente compartido para la API de NVIDIA NIM (chat completions). Acepta
`(system_prompt, user_prompt)` y autentica con `NVIDIA_API_KEY` desde el
config local de cada enfoque.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque7.5",
                     "nvidia_client.py")
_spec = importlib.util.spec_from_file_location("enfoque75_nvidia_client", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

translate = _mod.translate
