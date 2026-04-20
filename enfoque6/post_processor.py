"""Post-procesamiento — extiende enfoque3.post_processor con soporte DeepSeek-R1."""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "enfoque3"))
from post_processor import clean as _clean_base  # noqa: E402

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean(raw_response: str) -> str:
    """Elimina bloques <think> de DeepSeek-R1 y luego aplica limpieza base."""
    text = _THINK_PATTERN.sub("", raw_response).strip()
    return _clean_base(text)
