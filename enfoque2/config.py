"""Configuración de enfoque2: fine-tuning de bart-base-spanish para SPA→MSLG."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Datos (mismos que enfoque1/3/4) ──
DATASET_PATH = os.path.join(
    os.path.dirname(BASE_DIR), "enfoque3", "data", "MSLG_SPA_train.txt"
)
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# ── Modelo ──
MODEL_NAME = "vgaraujov/bart-base-spanish"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128

# ── Entrenamiento ──
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16 = True
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_bleu"

OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")
