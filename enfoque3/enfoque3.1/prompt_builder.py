"""Prompt builder enfoque 3.1"""
import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("e3_prompt_builder", _path)
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

PROMPT_BASE = _pb.PROMPT_BASE
FIXED_EXAMPLES = _pb.FIXED_EXAMPLES

def build_rag(sentence, retrieved_examples):
    examples_text = "\n".join(f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"' for ex in retrieved_examples)
    system = f"{PROMPT_BASE}\n\nEJEMPLOS (similares a la oración a traducir):\n{examples_text}"
    user = f'Traduce:\nSPA: "{sentence}"\nMSLG:'
    return system, user
