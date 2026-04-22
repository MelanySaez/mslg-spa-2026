"""Configuración enfoque7.2 — plan de escalado en 3 pasos sobre
few_shot_rag_curriculum.

Hereda de enfoque7/config.py. Sobreescribe:
  - RESULTS_DIR → enfoque7.2/results/scaling/
  - EXPERIMENTS → sweep de k (8, 10, 12, 15) + mejor-k con Self-Consistency N=3.
  - SC_TEMPERATURE → temperatura para llamadas con Self-Consistency.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_e7_config = os.path.join(_here, "..", "config.py")
_spec = importlib.util.spec_from_file_location("enfoque7_config", _e7_config)
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Re-export de enfoque7/config.py
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
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "scaling")
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)

# Self-Consistency: temperature para las N llamadas.
SC_TEMPERATURE = float(os.environ.get("SC_TEMPERATURE", "0.3"))

# Matriz de escalado:
#   Paso 1: sweep de k (8, 10, 12, 15) sobre rag_curriculum con post-proc v2.
#   Paso 2: post_processor v2 se aplica automáticamente a TODOS (hard rules).
#   Paso 3: Self-Consistency N=3 sobre k=10 (ampliable tras ver resultados).
EXPERIMENTS = [
    # Paso 1 — sweep de k
    {"name": "rag-curriculum-k8",      "type": "few_shot_rag_curriculum", "k": 8,  "sc_n": 1},
    {"name": "rag-curriculum-k10",     "type": "few_shot_rag_curriculum", "k": 10, "sc_n": 1},
    {"name": "rag-curriculum-k12",     "type": "few_shot_rag_curriculum", "k": 12, "sc_n": 1},
    {"name": "rag-curriculum-k15",     "type": "few_shot_rag_curriculum", "k": 15, "sc_n": 1},
    # Paso 3 — Self-Consistency sobre k=10 (ajusta manualmente si otro k gana)
    {"name": "rag-curriculum-k10-sc3", "type": "few_shot_rag_curriculum", "k": 10, "sc_n": 3},
]
