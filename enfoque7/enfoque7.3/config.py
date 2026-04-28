"""Configuración enfoque7.3 — réplica local de enfoque7.1 (RAG + curriculum).

Mismo experimento ganador (`few-shot-10-rag-curriculum`, dirección SPA→MSLG)
pero servido por un modelo LOCAL vía Ollama (deepseek-r1:32b) en lugar de la
API de Anthropic. Esto permite reproducir el pipeline sin costo de API y
comparar el desempeño de un razonador open-source contra Claude Haiku 4.5.

El split y el seed son los mismos que enfoque7.1 (heredados desde
enfoque7/data_loader.py), por lo que las métricas son comparables uno-a-uno.
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

# ── Datos (mismo split que enfoque7.1) ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "enfoque3", "data", "MSLG_SPA_train.txt")
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "rag-curriculum")
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# ── Embeddings (para retrieval RAG) ───────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# ── Submission MSLG-SPA 2026 ──────────────────────────────────────────────────
DIRECTION = "spa2mslg"
SUBTASK = "SPA2MSLG"
TEAM_NAME = os.environ.get("TEAM_NAME", "PrismaticVision")
SOLUTION_NAME = os.environ.get(
    "SOLUTION_NAME", "FewShot10RagCurriculumDeepseekR1_32b"
)
SUBMISSION_INCLUDE_ID = os.environ.get(
    "SUBMISSION_INCLUDE_ID", "false").lower() == "true"

# ── Retry / timeout ───────────────────────────────────────────────────────────
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "240"))
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "3"))

# ── Único experimento: réplica del ganador de 7.1 sobre modelo local ──────────
EXPERIMENTS = [
    {"name": "few-shot-10-rag-curriculum",
     "type": "few_shot_rag_curriculum",
     "k": 10},
]
