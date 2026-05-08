"""Evaluador MSLG→SPA — shim a enfoque7.2/evaluator.py.

Calcula BLEU, METEOR, chrF y COMET (métrica obligatoria del subtask MSLG2SPA).
Espera resultados con claves: 'mslg', 'spa_real', 'spa_pred'.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "enfoque7", "enfoque7.2", "evaluator.py")
_spec = importlib.util.spec_from_file_location("e72_evaluator", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

evaluate = _mod.evaluate
