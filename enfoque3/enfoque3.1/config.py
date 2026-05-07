"""Configuración para enfoque 3.1: Anthropic API y solo RAG-10."""

import importlib.util
import os
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_e3_config = os.path.join(_here, "..", "config.py")
_spec = importlib.util.spec_from_file_location("enfoque3_config", _e3_config)
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Constants from e3
OLLAMA_MODEL = _c.OLLAMA_MODEL
OLLAMA_URL = _c.OLLAMA_URL
TEMPERATURE = _c.TEMPERATURE
MAX_TOKENS = _c.MAX_TOKENS
BASE_DIR = _here
DATASET_PATH = _c.DATASET_PATH
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TRAIN_SPLIT = _c.TRAIN_SPLIT
VAL_SPLIT = _c.VAL_SPLIT
RANDOM_SEED = _c.RANDOM_SEED
EMBEDDING_MODEL = _c.EMBEDDING_MODEL

# Anthropic Logic
try:
    from dotenv import load_dotenv
    for candidate in (
        Path(_here) / ".env",
        Path(_here).parent / ".env",
        Path(_here).parent.parent / "enfoque7" / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_TIMEOUT = int(os.environ.get("ANTHROPIC_TIMEOUT", "120"))
ANTHROPIC_MAX_RETRIES = int(os.environ.get("ANTHROPIC_MAX_RETRIES", "4"))
ENABLE_PROMPT_CACHE = os.environ.get("ANTHROPIC_PROMPT_CACHE", "true").lower() == "true"

EXPERIMENTS = [
    {"name": "rag-10", "type": "rag", "k": 10},
]
