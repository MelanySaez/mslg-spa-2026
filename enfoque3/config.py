"""Configuración global del pipeline SPA→MSLG."""

import os

# Modelo Ollama
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.1
MAX_TOKENS = 150

# Datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "MSLG_SPA_train.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# Embeddings (para RAG)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Experimentos a ejecutar
EXPERIMENTS = [
    {"name": "zero-shot",   "type": "zero_shot", "k": 0},
    {"name": "few-shot-3",  "type": "few_shot",  "k": 3},
    {"name": "few-shot-5",  "type": "few_shot",  "k": 5},
    {"name": "few-shot-10", "type": "few_shot",  "k": 10},
    {"name": "few-shot-15", "type": "few_shot",  "k": 15},
    {"name": "rag-5",       "type": "rag",        "k": 5},
    {"name": "rag-7",       "type": "rag",        "k": 7},
    {"name": "rag-10",      "type": "rag",        "k": 10},
]

# Retry / timeout para Ollama
OLLAMA_TIMEOUT = 120
OLLAMA_MAX_RETRIES = 3
