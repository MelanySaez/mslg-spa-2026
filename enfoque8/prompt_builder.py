"""Prompt builder enfoque 8: Many-Shot In-Context Learning."""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_here, "..", "enfoque7", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("e7_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

PROMPT_BASE = _pb.PROMPT_BASE

def build_many_shot(sentence: str, examples_pool: list):
    """
    Construye un prompt de sistema pasando TODO el pool de ejemplos.
    Para validación serán 400 ejemplos. Para test, serán 490.
    """
    examples_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in examples_pool
    )
    
    system = f"{PROMPT_BASE}\n\nEJEMPLOS (Contexto completo del corpus):\n{examples_text}"
    user = f'Traduce:\nSPA: "{sentence}"\nMSLG:'
    
    return system, user
