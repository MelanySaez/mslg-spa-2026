"""Carga y split del dataset — reutiliza enfoque6/data_loader.py.

enfoque6/data_loader.py hace `import config`; al ejecutarse desde enfoque7
con sys.path[0]=enfoque7/, resuelve enfoque7/config.py (mismos atributos
DATASET_PATH, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED).
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque6", "data_loader.py")
_spec = importlib.util.spec_from_file_location("enfoque6_data_loader", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_dataset = _mod.load_dataset
split_dataset = _mod.split_dataset
