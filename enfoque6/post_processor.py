"""Post-procesamiento — extiende enfoque3/post_processor.py con soporte DeepSeek-R1."""

import importlib.util
import os
import re

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque3", "post_processor.py")
_spec = importlib.util.spec_from_file_location("enfoque3_post_processor", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_clean_base = _mod.clean

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean(raw_response: str) -> str:
    """Elimina bloques <think> de DeepSeek-R1 y luego aplica limpieza base."""
    text = _THINK_PATTERN.sub("", raw_response).strip()
    return _clean_base(text)
