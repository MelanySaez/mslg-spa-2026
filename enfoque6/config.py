"""Configuración global — Enfoque 6: Reglas LSM + Híbrido (deepseek-r1:70b)."""

import os

# Modelo Ollama
OLLAMA_MODEL = "deepseek-r1:70b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.1
MAX_TOKENS = 256

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

# Experimentos a ejecutar
EXPERIMENTS = [
    {"name": "zero-shot-rules",    "type": "zero_shot",   "k": 0},
    {"name": "few-shot-5-rules",   "type": "few_shot",    "k": 5},
    {"name": "few-shot-10-rules",  "type": "few_shot",    "k": 10},
    {"name": "hybrid-zero",        "type": "hybrid_zero", "k": 0},
    {"name": "hybrid-few-5",       "type": "hybrid_few",  "k": 5},
    {"name": "hybrid-few-10",      "type": "hybrid_few",  "k": 10},
    {"name": "rag-hybrid-7",       "type": "rag_hybrid",  "k": 7},
]

# Retry / timeout para Ollama
OLLAMA_TIMEOUT = 240
OLLAMA_MAX_RETRIES = 3
