"""Configuración global — Enfoque 6: Reglas LSM + Híbrido (deepseek-r1:70b)."""

import os

# Modelo Ollama
OLLAMA_MODEL = "deepseek-r1:32b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.1
MAX_TOKENS = 2048

# Datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "enfoque3", "data", "MSLG_SPA_train.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# Embeddings (para RAG)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Experimentos a ejecutar.
# Baselines originales comentados para acelerar la comparación del nuevo batch.
# Descomentar si se quieren re-correr junto con las variantes nuevas.
EXPERIMENTS = [
    # ── Baselines previos ──────────────────────────────────────────────
    # {"name": "zero-shot-rules", "type": "zero_shot", "k": 0},
    # {"name": "few-shot-5-rules", "type": "few_shot", "k": 5},
    # {"name": "few-shot-10-rules", "type": "few_shot", "k": 10},
    # {"name": "hybrid-zero", "type": "hybrid_zero", "k": 0},
    # {"name": "hybrid-few-5", "type": "hybrid_few", "k": 5},
    # {"name": "hybrid-few-10", "type": "hybrid_few", "k": 10},
    # {"name": "rag-hybrid-7", "type": "rag_hybrid", "k": 7},

    # ── Zero-shot mejorado ─────────────────────────────────────────────
    {"name": "zero-shot-cot", "type": "zero_shot_cot", "k": 0},
    {"name": "zero-shot-glossary", "type": "zero_shot_glossary", "k": 0},
    {"name": "zero-shot-full", "type": "zero_shot_full", "k": 0},

    # ── Few-shot mejorado (k=10 como baseline fuerte) ──────────────────
    {"name": "few-shot-10-cot", "type": "few_shot_cot", "k": 10},
    {"name": "few-shot-10-negative", "type": "few_shot_negative", "k": 10},
    {"name": "few-shot-10-curriculum", "type": "few_shot_curriculum", "k": 10},
    {"name": "few-shot-10-diverse", "type": "few_shot_diverse", "k": 10},
    {"name": "few-shot-10-full", "type": "few_shot_full", "k": 10},

    # ── Escalado de k (curva de saturación) ────────────────────────────
    {"name": "few-shot-15-rules", "type": "few_shot", "k": 15},
    {"name": "few-shot-20-rules", "type": "few_shot", "k": 20},
    {"name": "few-shot-15-diverse", "type": "few_shot_diverse", "k": 15},
]

# Retry / timeout para Ollama
OLLAMA_TIMEOUT = 240
OLLAMA_MAX_RETRIES = 3
