"""Evaluación — reutiliza enfoque3.evaluator directamente."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "enfoque3"))
from evaluator import evaluate  # noqa: F401, E402
