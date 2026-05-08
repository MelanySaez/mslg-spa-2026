"""Prompt builder enfoque8.1: Many-Shot MSLG → SPA.

Reutiliza PROMPT_BASE de enfoque7.2/prompt_builder.py (instrucciones inversas
para glosa → español) y construye el bloque de ejemplos con TODO el pool.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_here, "..", "..", "enfoque7", "enfoque7.2", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("e72_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

PROMPT_BASE = _pb.PROMPT_BASE


def build_many_shot(mslg: str, examples_pool: list):
    """
    Construye prompt pasando TODO el pool de ejemplos en el system.
    Validación: ~400 ejemplos. Submission: ~490 (train+val completo).
    """
    examples_text = "\n".join(
        f'MSLG: "{ex["mslg"]}" → SPA: "{ex["spa"]}"'
        for ex in examples_pool
    )

    system = f"{PROMPT_BASE}\n\nEJEMPLOS (Contexto completo del corpus):\n{examples_text}"
    user = f'Traduce:\nMSLG: "{mslg}"\nSPA:'

    return system, user
