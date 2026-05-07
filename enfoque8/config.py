"""Configuración global — Enfoque 8: Many-Shot In-Context Learning (Anthropic)."""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_parent_config = os.path.join(_here, "..", "enfoque7", "config.py")
_spec = importlib.util.spec_from_file_location("e7_config", _parent_config)
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Re-export
ANTHROPIC_API_KEY = _c.ANTHROPIC_API_KEY
ANTHROPIC_MODEL = _c.ANTHROPIC_MODEL
TEMPERATURE = _c.TEMPERATURE
MAX_TOKENS = _c.MAX_TOKENS
ENABLE_PROMPT_CACHE = _c.ENABLE_PROMPT_CACHE
PROJECT_ROOT = _c.PROJECT_ROOT
DATASET_PATH = _c.DATASET_PATH
TRAIN_SPLIT = _c.TRAIN_SPLIT
VAL_SPLIT = _c.VAL_SPLIT
RANDOM_SEED = _c.RANDOM_SEED
ANTHROPIC_TIMEOUT = _c.ANTHROPIC_TIMEOUT
ANTHROPIC_MAX_RETRIES = _c.ANTHROPIC_MAX_RETRIES
SUBMISSIONS_DIR = _c.SUBMISSIONS_DIR

# Overrides
BASE_DIR = _here
RESULTS_SUBDIR = "many-shot"
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)

RUN_TEST = os.environ.get("RUN_TEST", "true").lower() == "true"
RUN_VAL = os.environ.get("RUN_VAL", "true").lower() == "true"
TEST_PATH = os.environ.get("TEST_PATH", os.path.join(PROJECT_ROOT, "SPA2MSLG_test.txt"))

EXPERIMENTS = [
    {"name": "many-shot-all", "type": "many_shot"},
]
