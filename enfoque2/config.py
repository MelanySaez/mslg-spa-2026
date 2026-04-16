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
TASK_PREFIX = "Traducir a glosas MSLG: "

# ── Entrenamiento ──
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 8   # batch efectivo = 8 × 8 = 64, pero 8× más updates
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
LABEL_SMOOTHING = 0.1
WARMUP_STEPS = 100
FP16 = True
OPTIM = "adamw_torch_fused"
LR_SCHEDULER = "cosine"
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_bleu_4"

# ── Generación ──
NUM_BEAMS = 8
NO_REPEAT_NGRAM_SIZE = 3
LENGTH_PENALTY = 0.8

OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")
