"""Configuración enfoque7.6 — réplica de enfoque7.2 (reverso MSLG→SPA) sobre NVIDIA NIM.

Mismo experimento ganador (`few-shot-10-rag-curriculum-mslg2spa`, dirección
MSLG→SPA) que enfoque7.2 (Anthropic) y 7.4 (Ollama local), pero servido por
la API de NVIDIA NIM con `google/gemma-4-31b-it` (31B instruction-tuned).
Modo `enable_thinking` desactivado por defecto.

COMET (métrica obligatoria del subtask MSLG2SPA) sigue invocándose vía el CLI
`comet-score` exactamente igual que en 7.2 / 7.4 (controlado por ENABLE_COMET).
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


# ── Modelo NVIDIA NIM (HTTP + SSE OpenAI-compatible) ─────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "google/gemma-4-31b-it")
NVIDIA_URL = os.environ.get(
    "NVIDIA_URL", "https://integrate.api.nvidia.com/v1/chat/completions"
)
TEMPERATURE = float(os.environ.get("NVIDIA_TEMPERATURE", "1"))
TOP_P = float(os.environ.get("NVIDIA_TOP_P", "0.95"))
MAX_TOKENS = int(os.environ.get("NVIDIA_MAX_TOKENS", "1024"))
NVIDIA_STREAM = os.environ.get("NVIDIA_STREAM", "true").lower() == "true"
NVIDIA_ENABLE_THINKING = os.environ.get(
    "NVIDIA_ENABLE_THINKING", "false").lower() == "true"

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
    "SOLUTION_NAME", "FewShot10RagCurriculumGemma4_31B"
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
NVIDIA_TIMEOUT = int(os.environ.get("NVIDIA_TIMEOUT", "600"))
NVIDIA_MAX_RETRIES = int(os.environ.get("NVIDIA_MAX_RETRIES", "3"))

# ── Único experimento: réplica del ganador de 7.2 sobre NVIDIA NIM ────────────
EXPERIMENTS = [
    {"name": "few-shot-10-rag-curriculum-mslg2spa",
     "type": "few_shot_rag_curriculum_reverse",
     "k": 10},
]
