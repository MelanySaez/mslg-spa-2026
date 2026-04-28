"""Shim a enfoque7.1/prompt_builder.py.

Reusa el mismo conjunto de builders SPA→MSLG (incluyendo el constructor
`build_few_shot_rag_curriculum`) que ganó en enfoque7.1. Lo único que cambia
en enfoque7.3 es el cliente del modelo (Ollama en vez de Anthropic).
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_here, "..", "enfoque7.1", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("enfoque71_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

# Re-export constantes
PROMPT_BASE = _pb.PROMPT_BASE
FIXED_EXAMPLES = _pb.FIXED_EXAMPLES
LSM_GLOSSARY = _pb.LSM_GLOSSARY
NEGATIVE_EXAMPLES = _pb.NEGATIVE_EXAMPLES
COT_INSTRUCTIONS = _pb.COT_INSTRUCTIONS

# Re-export builders
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
build_few_shot_rag_curriculum = _pb.build_few_shot_rag_curriculum
