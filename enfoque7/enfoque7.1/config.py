"""Configuración enfoque7.1 — hereda de enfoque7/config.py y sobreescribe
RESULTS_SUBDIR + EXPERIMENTS para el híbrido RAG + curriculum.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_e7_config = os.path.join(_here, "..", "config.py")
_spec = importlib.util.spec_from_file_location("enfoque7_config", _e7_config)
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Re-export de todas las constantes de enfoque7/config.py
ANTHROPIC_API_KEY = _c.ANTHROPIC_API_KEY
ANTHROPIC_MODEL = _c.ANTHROPIC_MODEL
TEMPERATURE = _c.TEMPERATURE
MAX_TOKENS = _c.MAX_TOKENS
ENABLE_PROMPT_CACHE = _c.ENABLE_PROMPT_CACHE
DATASET_PATH = _c.DATASET_PATH
TRAIN_SPLIT = _c.TRAIN_SPLIT
VAL_SPLIT = _c.VAL_SPLIT
RANDOM_SEED = _c.RANDOM_SEED
EMBEDDING_MODEL = _c.EMBEDDING_MODEL
ANTHROPIC_TIMEOUT = _c.ANTHROPIC_TIMEOUT
ANTHROPIC_MAX_RETRIES = _c.ANTHROPIC_MAX_RETRIES
SUBMISSIONS_DIR = _c.SUBMISSIONS_DIR

# ── Overrides ─────────────────────────────────────────────────────────────────
BASE_DIR = _here
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "rag-curriculum")
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)

# ── Submission MSLG-SPA 2026 ──────────────────────────────────────────────────
# Subtask oficial: SPA2MSLG (Spanish-to-Gloss). Métricas: BLEU, METEOR, chrF
# (COMET NO aplica en este subtask por las reglas de la actividad).
# La salida .txt sigue el formato 'TeamName_SolutionName_SPA2MSLG.txt'.
DIRECTION = "spa2mslg"
SUBTASK = "SPA2MSLG"
TEAM_NAME = os.environ.get("TEAM_NAME", "PrismaticVision")
SOLUTION_NAME = os.environ.get("SOLUTION_NAME", "FewShot10RagCurriculum")
SUBMISSION_INCLUDE_ID = os.environ.get(
    "SUBMISSION_INCLUDE_ID", "false").lower() == "true"

# Un único experimento: híbrido RAG + curriculum k=10
EXPERIMENTS = [
    {"name": "few-shot-10-rag-curriculum",
     "type": "few_shot_rag_curriculum",
     "k": 10},
]
