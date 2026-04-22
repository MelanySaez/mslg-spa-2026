"""Configuración global — Enfoque 7: Zero-shot & Few-shot vía Anthropic API (Claude Haiku 4.5).

Carga la clave de API y parámetros desde variables de entorno (.env).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Busca .env en el directorio del paquete y en la raíz del proyecto.
    _here = Path(__file__).resolve().parent
    for candidate in (_here / ".env", _here.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    # python-dotenv opcional: si no está instalado, se usan las env vars del sistema.
    pass


# ── API Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
TEMPERATURE = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024"))
ENABLE_PROMPT_CACHE = os.environ.get("ANTHROPIC_PROMPT_CACHE", "true").lower() == "true"

# ── Datos ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "enfoque3", "data", "MSLG_SPA_train.txt")
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "improved")
RESULTS_DIR = os.path.join(BASE_DIR, "results", RESULTS_SUBDIR)
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# Embeddings (para few-shot-diverse)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Experimentos ──────────────────────────────────────────────────────────────
# Matriz reducida (5 variantes efectivas, no 11). Eliminadas: zero_shot_cot,
# zero_shot_glossary, zero_shot_full, few_shot_cot, few_shot_negative,
# few_shot_full — sin ganancia demostrada y costo duplicado.
# Añadida: few_shot_rag (retrieval real sobre pool del train).
EXPERIMENTS = [
    # Control: zero-shot con reglas corregidas
    {"name": "zero-shot-rules",        "type": "zero_shot",           "k": 0},
    # Baseline mínimo few-shot
    {"name": "few-shot-5-rules",       "type": "few_shot",            "k": 5},
    # Ganador previo (curriculum k=10) con ejemplos corpus-faithful
    {"name": "few-shot-10-curriculum", "type": "few_shot_curriculum", "k": 10},
    # Cobertura diversa vía k-means
    {"name": "few-shot-10-diverse",    "type": "few_shot_diverse",    "k": 10},
    # NUEVA: RAG top-k sobre pool real del train (dominante teórico)
    {"name": "few-shot-5-rag",         "type": "few_shot_rag",        "k": 5},
]

# ── Retry / timeout ───────────────────────────────────────────────────────────
ANTHROPIC_TIMEOUT = 120
ANTHROPIC_MAX_RETRIES = 4
