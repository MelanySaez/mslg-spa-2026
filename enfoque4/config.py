"""Configuración local de enfoque4.

Reexporta constantes de enfoque3 (el paquete __init__.py ya añadió enfoque3
al sys.path) para garantizar que el split del dataset y los parámetros del
modelo sean idénticos — de esa forma las métricas son directamente
comparables con los resultados de enfoque3.
"""

import os

import config as _e3_config  # enfoque3/config.py (vía sys.path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

OLLAMA_MODEL = _e3_config.OLLAMA_MODEL
OLLAMA_URL = _e3_config.OLLAMA_URL
TEMPERATURE = _e3_config.TEMPERATURE
MAX_TOKENS = _e3_config.MAX_TOKENS
OLLAMA_TIMEOUT = _e3_config.OLLAMA_TIMEOUT
OLLAMA_MAX_RETRIES = _e3_config.OLLAMA_MAX_RETRIES

DATASET_PATH = _e3_config.DATASET_PATH
TRAIN_SPLIT = _e3_config.TRAIN_SPLIT
VAL_SPLIT = _e3_config.VAL_SPLIT
RANDOM_SEED = _e3_config.RANDOM_SEED

EMBEDDING_MODEL = _e3_config.EMBEDDING_MODEL

SPACY_MODEL = "es_core_news_lg"

# Retrieval híbrido (BM25 + denso + cross-encoder rerank).
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_ENABLED = True
RETRIEVAL_CANDIDATES = 30   # top-N por cada ranker antes del RRF
RERANK_POOL = 15            # candidatos reordenados por el cross-encoder
LENGTH_TOLERANCE = 0.75     # |len(cand) - len(query)| / len(query) permitido

EXPERIMENTS = [
    {"name": "fol-rag-10",      "type": "fol_rag",      "k": 10},
    {"name": "enriched-rag-10", "type": "enriched_rag",  "k": 10},
]
