"""Prompt builder enfoque7.1 — hereda de enfoque7/prompt_builder.py y añade
la variante híbrida RAG + curriculum.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_here, "..", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("enfoque7_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

# Re-export constantes y builders existentes
PROMPT_BASE = _pb.PROMPT_BASE
FIXED_EXAMPLES = _pb.FIXED_EXAMPLES
LSM_GLOSSARY = _pb.LSM_GLOSSARY
NEGATIVE_EXAMPLES = _pb.NEGATIVE_EXAMPLES
COT_INSTRUCTIONS = _pb.COT_INSTRUCTIONS

build_zero_shot = _pb.build_zero_shot
build_zero_shot_cot = _pb.build_zero_shot_cot
build_zero_shot_glossary = _pb.build_zero_shot_glossary
build_zero_shot_full = _pb.build_zero_shot_full
build_few_shot = _pb.build_few_shot
build_few_shot_cot = _pb.build_few_shot_cot
build_few_shot_negative = _pb.build_few_shot_negative
build_few_shot_curriculum = _pb.build_few_shot_curriculum
build_few_shot_diverse = _pb.build_few_shot_diverse
build_few_shot_full = _pb.build_few_shot_full
build_few_shot_rag = _pb.build_few_shot_rag


def build_few_shot_rag_curriculum(sentence: str, retrieved_examples: list):
    """RAG + curriculum: retrieval dinámico top-k por embeddings + orden
    ascendente por longitud del spa (recency bias a favor del ejemplo más
    complejo, que queda PEGADO a la query).
    """
    ordered = sorted(retrieved_examples, key=lambda e: len(e["spa"]))
    system = _pb._compose_system(
        _pb.PROMPT_BASE,
        _pb._examples_block(
            ordered,
            header="EJEMPLOS (recuperados y ordenados de simple a complejo):",
        ),
    )
    return system, _pb._user(sentence)
