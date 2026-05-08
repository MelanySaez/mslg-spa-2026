"""Configuración enfoque8.1 — Many-Shot MSLG → SPA (reverso de enfoque8)."""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_e8_config = os.path.join(_here, "..", "config.py")
_spec = importlib.util.spec_from_file_location("e8_config", _e8_config)
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Re-export desde enfoque8/config.py (que re-exporta desde enfoque7/config.py)
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

# Overrides
BASE_DIR = _here
RESULTS_SUBDIR = "many-shot"
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

# Dirección inversa
DIRECTION = "mslg2spa"
SUBTASK = "MSLG2SPA"
TEAM_NAME = os.environ.get("TEAM_NAME", "VerbaNexAI")
SOLUTION_NAME = os.environ.get("SOLUTION_NAME", "ManyShot")
SUBMISSION_INCLUDE_ID = os.environ.get("SUBMISSION_INCLUDE_ID", "false").lower() == "true"

TEST_PATH = os.environ.get("TEST_PATH", os.path.join(PROJECT_ROOT, "MSLG2SPA_test.txt"))
TEST_SOURCE_COL = "MSLG"

RUN_TEST = os.environ.get("RUN_TEST", "true").lower() == "true"
RUN_VAL = os.environ.get("RUN_VAL", "true").lower() == "true"

# COMET (obligatorio para MSLG2SPA según bases de la tarea)
ENABLE_COMET = os.environ.get("ENABLE_COMET", "true").lower() == "true"
COMET_MODEL = os.environ.get("COMET_MODEL", "Unbabel/wmt22-comet-da")
COMET_BATCH_SIZE = int(os.environ.get("COMET_BATCH_SIZE", "8"))
COMET_BIN = os.environ.get("COMET_BIN", "comet-score")
COMET_GPUS = int(os.environ.get("COMET_GPUS", "0"))

EXPERIMENTS = [
    {"name": "many-shot-all-mslg2spa", "type": "many_shot"},
]
