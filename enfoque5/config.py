"""Configuración de enfoque5: fine-tuning con augmentación sintética vía reglas."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Datos originales ──
ORIGINAL_DATASET = os.path.join(
    os.path.dirname(BASE_DIR), "enfoque3", "data", "MSLG_SPA_train.txt"
)
AUGMENTED_DATASET = os.path.join(DATA_DIR, "augmented_train.txt")
TRAIN_SPLIT = 400
VAL_SPLIT = 90
RANDOM_SEED = 42

# ── Augmentación ──
AUGMENTATIONS_PER_SAMPLE = 4     # variaciones por oración original
WORD_DROP_PROB = 0.15            # prob de eliminar cada palabra
WORD_SWAP_PROB = 0.10            # prob de swap con palabra adyacente
SPACY_MODEL = "es_core_news_lg"

# ── Modelo (mismo que enfoque2) ──
MODEL_NAME = "vgaraujov/bart-base-spanish"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128
TASK_PREFIX = "Traducir a glosas MSLG: "

# ── Entrenamiento ──
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 8
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
