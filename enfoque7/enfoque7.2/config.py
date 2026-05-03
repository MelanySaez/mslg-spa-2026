"""Configuración enfoque7.2 — pipeline reverso MSLG → SPA.

Hereda de enfoque7/config.py (modelo, claves API, dataset, splits, seed) y
sobreescribe RESULTS_SUBDIR + EXPERIMENTS para correr la versión reversa del
mejor experimento de 7.1: few-shot-10-rag-curriculum aplicado a MSLG → SPA.

El split es idéntico al de 7.1 (mismo pool y val por seed), así los IDs
evaluados coinciden y el conjunto es directamente comparable cara a cara.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_e7_config = os.path.join(_here, "..", "config.py")
_spec = importlib.util.spec_from_file_location("enfoque7_config", _e7_config)
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Re-export de constantes de enfoque7/config.py
ANTHROPIC_API_KEY = _c.ANTHROPIC_API_KEY
ANTHROPIC_MODEL = _c.ANTHROPIC_MODEL
TEMPERATURE = _c.TEMPERATURE
MAX_TOKENS = _c.MAX_TOKENS
ENABLE_PROMPT_CACHE = _c.ENABLE_PROMPT_CACHE
DATASET_PATH = _c.DATASET_PATH
PROJECT_ROOT = _c.PROJECT_ROOT
TRAIN_SPLIT = _c.TRAIN_SPLIT
VAL_SPLIT = _c.VAL_SPLIT
RANDOM_SEED = _c.RANDOM_SEED
EMBEDDING_MODEL = _c.EMBEDDING_MODEL
ANTHROPIC_TIMEOUT = _c.ANTHROPIC_TIMEOUT
ANTHROPIC_MAX_RETRIES = _c.ANTHROPIC_MAX_RETRIES
SUBMISSIONS_DIR = _c.SUBMISSIONS_DIR

# ── Overrides ─────────────────────────────────────────────────────────────────
BASE_DIR = _here
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "reverse")
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)

# Dirección de traducción: este enfoque usa MSLG como query y SPA como target.
DIRECTION = "mslg2spa"

# ── Submission MSLG-SPA 2026 ──────────────────────────────────────────────────
# Subtask oficial: MSLG2SPA (Gloss-to-Spanish). Métricas: BLEU, METEOR, chrF,
# COMET. La salida .txt sigue el formato 'TeamName_SolutionName_MSLG2SPA.txt'.
SUBTASK = "MSLG2SPA"
TEAM_NAME = os.environ.get("TEAM_NAME", "PrismaticVision")
SOLUTION_NAME = os.environ.get("SOLUTION_NAME", "FewShot10RagCurriculum")
# Si True, anteponer el ID en la línea como mecanismo de verificación opcional.
SUBMISSION_INCLUDE_ID = os.environ.get(
    "SUBMISSION_INCLUDE_ID", "false").lower() == "true"

# Test set oficial (244 pares). TSV con columnas: ID, MSLG.
TEST_PATH = os.environ.get(
    "TEST_PATH", os.path.join(PROJECT_ROOT, "MSLG2SPA_test.txt"))
TEST_SOURCE_COL = "MSLG"

# Modo de ejecución del main.
#   RUN_TEST=true  -> inferencia sobre TEST_PATH y dump de submission .txt
#   RUN_VAL=true   -> evaluación sobre split val (BLEU/METEOR/chrF/COMET)
RUN_TEST = os.environ.get("RUN_TEST", "true").lower() == "true"
RUN_VAL = os.environ.get("RUN_VAL", "false").lower() == "true"

# ── COMET (solo MSLG2SPA según la actividad) ─────────────────────────────────
# COMET 2.x choca con pandas>=3 / torch>=2 dentro del proyecto, así que se
# invoca el CLI 'comet-score' (instalado vía 'uv tool install unbabel-comet').
ENABLE_COMET = os.environ.get("ENABLE_COMET", "true").lower() == "true"
COMET_MODEL = os.environ.get("COMET_MODEL", "Unbabel/wmt22-comet-da")
COMET_BATCH_SIZE = int(os.environ.get("COMET_BATCH_SIZE", "8"))
COMET_BIN = os.environ.get("COMET_BIN", "comet-score")
COMET_GPUS = int(os.environ.get("COMET_GPUS", "0"))

# Único experimento: réplica del ganador de 7.1 en sentido reverso.
EXPERIMENTS = [
    {"name": "few-shot-10-rag-curriculum-mslg2spa",
     "type": "few_shot_rag_curriculum_reverse",
     "k": 10},
]
