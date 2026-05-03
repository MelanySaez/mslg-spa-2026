"""Shim a enfoque7/evaluator.py (BLEU + METEOR + chrF)."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "evaluator.py")
_spec = importlib.util.spec_from_file_location("enfoque7_evaluator", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

evaluate = _mod.evaluate
