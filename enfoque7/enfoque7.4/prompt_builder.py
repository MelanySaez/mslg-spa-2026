"""Shim a enfoque7.2/prompt_builder.py (sentido reverso MSLG → SPA).

Reusa el PROMPT_BASE invertido y los builders (`build_few_shot_rag_curriculum`,
etc.) que ganaron en enfoque7.2. El único cambio en enfoque7.4 es el cliente
del modelo (Ollama deepseek-r1:32b en vez de Anthropic Haiku 4.5).
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_here, "..", "enfoque7.2", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("enfoque72_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

# Re-export constantes
PROMPT_BASE = _pb.PROMPT_BASE
FIXED_EXAMPLES = _pb.FIXED_EXAMPLES
NEGATIVE_EXAMPLES = _pb.NEGATIVE_EXAMPLES

# Re-export builders
build_zero_shot = _pb.build_zero_shot
build_few_shot = _pb.build_few_shot
build_few_shot_rag = _pb.build_few_shot_rag
build_few_shot_rag_curriculum = _pb.build_few_shot_rag_curriculum
