"""Shim a enfoque7.1/prompt_builder.py (incluye build_few_shot_rag_curriculum)."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque7.1", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("enfoque7_1_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

# Re-export de todo lo relevante
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
build_few_shot_rag_curriculum = _pb.build_few_shot_rag_curriculum
