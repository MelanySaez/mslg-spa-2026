"""Configuración enfoque7.4 — réplica local de enfoque7.2 (reverso MSLG→SPA).

Mismo experimento ganador (`few-shot-10-rag-curriculum-mslg2spa`, dirección
MSLG→SPA) que enfoque7.2, pero servido por un modelo LOCAL vía Ollama
(deepseek-r1:32b). Usa el mismo split y seed que 7.2 para que las métricas
sean comparables uno-a-uno con la versión Anthropic.

COMET (métrica obligatoria del subtask MSLG2SPA) sigue invocándose vía el CLI
`comet-score` exactamente igual que en 7.2 (controlado por ENABLE_COMET).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _here = Path(__file__).resolve().parent
    for candidate in (_here / ".env", _here.parent / ".env",
                      _here.parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    pass


# ── Modelo Ollama ─────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:32b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "1024"))

# ── Datos (mismo split que enfoque7.2) ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "enfoque3", "data", "MSLG_SPA_train.txt")
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "reverse")
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# Dirección de traducción
DIRECTION = "mslg2spa"

# ── Submission MSLG-SPA 2026 ──────────────────────────────────────────────────
SUBTASK = "MSLG2SPA"
TEAM_NAME = os.environ.get("TEAM_NAME", "PrismaticVision")
SOLUTION_NAME = os.environ.get(
    "SOLUTION_NAME", "FewShot10RagCurriculumDeepseekR1_32b"
)
SUBMISSION_INCLUDE_ID = os.environ.get(
    "SUBMISSION_INCLUDE_ID", "false").lower() == "true"

# ── COMET (solo MSLG2SPA según la actividad) ─────────────────────────────────
ENABLE_COMET = os.environ.get("ENABLE_COMET", "true").lower() == "true"
COMET_MODEL = os.environ.get("COMET_MODEL", "Unbabel/wmt22-comet-da")
COMET_BATCH_SIZE = int(os.environ.get("COMET_BATCH_SIZE", "8"))
COMET_BIN = os.environ.get("COMET_BIN", "comet-score")
COMET_GPUS = int(os.environ.get("COMET_GPUS", "0"))

# ── Retry / timeout ───────────────────────────────────────────────────────────
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "240"))
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "3"))

# ── Único experimento: réplica del ganador de 7.2 sobre modelo local ──────────
EXPERIMENTS = [
    {"name": "few-shot-10-rag-curriculum-mslg2spa",
     "type": "few_shot_rag_curriculum_reverse",
     "k": 10},
]
