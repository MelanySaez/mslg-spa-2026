"""Evaluación — reutiliza enfoque3/evaluator.py sin colisión de nombres."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque3", "evaluator.py")
_spec = importlib.util.spec_from_file_location("enfoque3_evaluator", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

evaluate = _mod.evaluate
