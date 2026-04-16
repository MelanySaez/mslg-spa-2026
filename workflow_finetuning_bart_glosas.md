# Workflow de Fine-Tuning: `vgaraujov/bart-base-spanish` → Traducción Español a Glosas

> **Objetivo:** Ajustar el modelo BART en español para traducir oraciones en español a representaciones en glosas (notación morfológica/lingüística). Se ejecutarán dos rondas de fine-tuning secuenciales.

---

## Índice

1. [Prerequisitos y Entorno](#1-prerequisitos-y-entorno)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Preparación de Datos](#3-preparación-de-datos)
4. [Fine-Tuning Ronda 1 (Dataset 3.000 registros)](#4-fine-tuning-ronda-1-dataset-3000-registros)
5. [Evaluación con Test Set — Ronda 1](#5-evaluación-con-test-set--ronda-1)
6. [Fine-Tuning Ronda 2 (Dataset 490 registros)](#6-fine-tuning-ronda-2-dataset-490-registros)
7. [Evaluación Final con Test Set — Ronda 2](#7-evaluación-final-con-test-set--ronda-2)
8. [Instrucciones para Agente de Código (Claude Code / OpenCode)](#8-instrucciones-para-agente-de-código-claude-code--opencode)

---

## 1. Prerequisitos y Entorno

### 1.1 Requerimientos del Sistema

- Python >= 3.10
- CUDA >= 11.8 (si se usa GPU; recomendado para entrenamiento)
- RAM >= 16 GB
- GPU VRAM >= 8 GB (recomendado para BART base)

### 1.2 Instrucciones para el Agente

```
Crea un entorno virtual de Python llamado 'glosas_env', actívalo e instala 
las siguientes dependencias:

pip install transformers==4.40.0
pip install datasets==2.19.0
pip install torch==2.2.2
pip install accelerate==0.29.3
pip install evaluate==0.4.1
pip install sacrebleu==2.4.0
pip install rouge-score==0.1.2
pip install sentencepiece==0.2.0
pip install pandas==2.2.2
pip install scikit-learn==1.4.2
pip install tensorboard==2.16.2
pip install tqdm==4.66.2

Verifica que todas las instalaciones sean exitosas ejecutando:
python -c "import transformers; import datasets; import torch; print('OK')"
```

---

## 2. Estructura del Proyecto

### 2.1 Instrucciones para el Agente

```
Crea la siguiente estructura de directorios en el proyecto:

finetuning_bart_glosas/
├── data/
│   ├── raw/
│   │   ├── dataset_3000.csv        ← Dataset base (3.000 registros)
│   │   └── dataset_490.csv         ← Dataset especializado (490 registros)
│   ├── round1/
│   │   ├── train.csv               ← 70% = 2.100 registros
│   │   ├── val.csv                 ← 10% = 300 registros
│   │   └── test.csv                ← 20% = 600 registros
│   └── round2/
│       ├── train.csv               ← 400 registros
│       ├── val.csv                 ← 90 registros
│       └── test.csv                ← Dataset de test externo (se provee por separado)
├── scripts/
│   ├── prepare_data.py
│   ├── train_round1.py
│   ├── evaluate.py
│   └── train_round2.py
├── models/
│   ├── round1/                     ← Checkpoints y modelo final Ronda 1
│   └── round2/                     ← Checkpoints y modelo final Ronda 2
├── results/
│   ├── round1_test_results.json
│   └── round2_test_results.json
└── logs/
    ├── round1/
    └── round2/

Ejecuta: mkdir -p finetuning_bart_glosas/{data/{raw,round1,round2},scripts,models/{round1,round2},results,logs/{round1,round2}}
```

---

## 3. Preparación de Datos

### 3.1 Formato Esperado del Dataset

Los archivos CSV deben tener **dos columnas obligatorias**:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| `source` | Oración en español | `"El niño come manzanas"` |
| `target` | Glosa correspondiente | `"DET.MASC.SG niño comer.PRS.3SG manzana.PL"` |

### 3.2 Script de Preparación de Datos

> **Instrucción para el agente:** Crea el archivo `finetuning_bart_glosas/scripts/prepare_data.py` con el siguiente contenido:

```python
"""
prepare_data.py
Divide los datasets para las dos rondas de fine-tuning.
Ronda 1: 3.000 registros → 70/10/20 (train/val/test)
Ronda 2: 490 registros   → 400/90 (train/val) + test externo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# ─── Configuración ────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = "data"

# ─── Ronda 1: Dataset de 3.000 registros ──────────────────────────────────────
def prepare_round1(input_path: str):
    print(f"[Ronda 1] Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path)
    
    assert "source" in df.columns and "target" in df.columns, \
        "El CSV debe tener columnas 'source' y 'target'"
    
    print(f"[Ronda 1] Total de registros: {len(df)}")
    
    # Primera división: separar test (20%)
    df_trainval, df_test = train_test_split(
        df, test_size=0.20, random_state=SEED, shuffle=True
    )
    
    # Segunda división: separar val (10% del total = ~12.5% del trainval)
    val_size_relative = 0.10 / 0.80  # ~0.125
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size_relative, random_state=SEED, shuffle=True
    )
    
    print(f"[Ronda 1] Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    # Guardar splits
    out_dir = os.path.join(DATA_DIR, "round1")
    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    
    # Guardar estadísticas
    stats = {
        "total": len(df),
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
        "train_pct": round(len(df_train)/len(df)*100, 1),
        "val_pct": round(len(df_val)/len(df)*100, 1),
        "test_pct": round(len(df_test)/len(df)*100, 1),
    }
    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Ronda 1] Splits guardados en: {out_dir}/")
    return stats

# ─── Ronda 2: Dataset de 490 registros ────────────────────────────────────────
def prepare_round2(input_path: str, test_path: str = None):
    print(f"[Ronda 2] Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path)
    
    assert "source" in df.columns and "target" in df.columns, \
        "El CSV debe tener columnas 'source' y 'target'"
    
    print(f"[Ronda 2] Total de registros: {len(df)}")
    
    # División fija: 400 train / 90 val
    df_train = df.iloc[:400].copy()
    df_val   = df.iloc[400:490].copy()
    
    # Alternativa con shuffle (descomentar si se prefiere aleatorio):
    # df_train, df_val = train_test_split(df, test_size=90, random_state=SEED, shuffle=True)
    
    print(f"[Ronda 2] Train: {len(df_train)} | Val: {len(df_val)}")
    
    out_dir = os.path.join(DATA_DIR, "round2")
    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    
    if test_path and os.path.exists(test_path):
        import shutil
        shutil.copy(test_path, os.path.join(out_dir, "test.csv"))
        print(f"[Ronda 2] Test externo copiado desde: {test_path}")
    else:
        print("[Ronda 2] ADVERTENCIA: No se proporcionó test set externo.")
        print("          Coloca tu archivo de test en data/round2/test.csv antes de evaluar.")
    
    stats = {
        "total": len(df),
        "train": len(df_train),
        "val": len(df_val),
    }
    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Ronda 2] Splits guardados en: {out_dir}/")
    return stats

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1_input", default="data/raw/dataset_3000.csv")
    parser.add_argument("--round2_input", default="data/raw/dataset_490.csv")
    parser.add_argument("--round2_test",  default=None, 
                        help="Ruta opcional al test set externo para ronda 2")
    args = parser.parse_args()
    
    stats1 = prepare_round1(args.round1_input)
    stats2 = prepare_round2(args.round2_input, args.round2_test)
    
    print("\n✅ Preparación de datos completada.")
    print(f"   Ronda 1 → train:{stats1['train']} / val:{stats1['val']} / test:{stats1['test']}")
    print(f"   Ronda 2 → train:{stats2['train']} / val:{stats2['val']}")
```

**Ejecución:**
```bash
cd finetuning_bart_glosas
python scripts/prepare_data.py \
  --round1_input data/raw/dataset_3000.csv \
  --round2_input data/raw/dataset_490.csv
```

---

## 4. Fine-Tuning Ronda 1 (Dataset 3.000 registros)

### 4.1 Configuración del Entrenamiento

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Modelo base | `vgaraujov/bart-base-spanish` | Modelo BART preentrenado en español |
| Learning rate | `5e-5` | Estándar para fine-tuning de seq2seq |
| Batch size (train) | `16` | Balance memoria/velocidad para BART base |
| Batch size (eval) | `16` | — |
| Épocas | `10` | Suficiente para convergencia inicial con 2.100 muestras |
| Max source length | `128` | Oraciones en español usualmente < 128 tokens |
| Max target length | `128` | Glosas suelen ser más cortas que la fuente |
| Warmup steps | `200` | ~10% de los pasos totales |
| Weight decay | `0.01` | Regularización estándar |
| Beam search | `4` | Balance entre calidad y velocidad |
| Save strategy | `epoch` | Guardar checkpoint por época |
| Eval strategy | `epoch` | Evaluar en cada época |
| Load best model | `True` | Cargar el mejor modelo al final |
| Metric for best | `eval_loss` | Minimizar pérdida de validación |

### 4.2 Script de Entrenamiento Ronda 1

> **Instrucción para el agente:** Crea el archivo `finetuning_bart_glosas/scripts/train_round1.py` con el siguiente contenido:

```python
"""
train_round1.py
Fine-tuning Ronda 1: vgaraujov/bart-base-spanish sobre 3.000 registros español→glosas.
"""

import os
import json
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ─── Configuración ────────────────────────────────────────────────────────────
MODEL_NAME      = "vgaraujov/bart-base-spanish"
DATA_DIR        = "data/round1"
OUTPUT_DIR      = "models/round1"
LOGS_DIR        = "logs/round1"
MAX_SRC_LENGTH  = 128
MAX_TGT_LENGTH  = 128
BATCH_SIZE      = 16
NUM_EPOCHS      = 10
LEARNING_RATE   = 5e-5
WARMUP_STEPS    = 200
WEIGHT_DECAY    = 0.01
SEED            = 42

# ─── Carga del tokenizer y modelo ─────────────────────────────────────────────
print(f"[Ronda 1] Cargando tokenizer y modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"[Ronda 1] Parámetros del modelo: {model.num_parameters():,}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Ronda 1] Dispositivo: {device}")

# ─── Carga y tokenización del dataset ─────────────────────────────────────────
def load_csv_as_dataset(split: str) -> Dataset:
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    return Dataset.from_pandas(df[["source", "target"]].dropna().reset_index(drop=True))

print("[Ronda 1] Cargando datasets...")
raw_datasets = DatasetDict({
    "train": load_csv_as_dataset("train"),
    "validation": load_csv_as_dataset("val"),
    "test": load_csv_as_dataset("test"),
})
print(f"[Ronda 1] Train: {len(raw_datasets['train'])} | "
      f"Val: {len(raw_datasets['validation'])} | "
      f"Test: {len(raw_datasets['test'])}")

def preprocess(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_SRC_LENGTH,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=MAX_TGT_LENGTH,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("[Ronda 1] Tokenizando datasets...")
tokenized = raw_datasets.map(
    preprocess,
    batched=True,
    remove_columns=["source", "target"],
    desc="Tokenizando",
)

# ─── Métricas ─────────────────────────────────────────────────────────────────
bleu   = evaluate.load("sacrebleu")
rouge  = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels         = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    bleu_result  = bleu.compute(predictions=decoded_preds,
                                references=[[l] for l in decoded_labels])
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels)
    return {
        "bleu":   round(bleu_result["score"], 4),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
    }

# ─── Data Collator ─────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

# ─── Argumentos de entrenamiento ──────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    predict_with_generate=True,
    generation_max_length=MAX_TGT_LENGTH,
    generation_num_beams=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=LOGS_DIR,
    logging_steps=50,
    save_total_limit=3,
    seed=SEED,
    report_to="tensorboard",
    fp16=torch.cuda.is_available(),  # Mixed precision si hay GPU
)

# ─── Trainer ──────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─── Entrenamiento ─────────────────────────────────────────────────────────────
print("\n[Ronda 1] Iniciando entrenamiento...")
train_result = trainer.train()

# ─── Guardado del modelo final ─────────────────────────────────────────────────
print("[Ronda 1] Guardando modelo final...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Guardar métricas de entrenamiento
with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
    json.dump(train_result.metrics, f, indent=2)

print(f"\n✅ [Ronda 1] Entrenamiento completado. Modelo guardado en: {OUTPUT_DIR}/")
print(f"   Métricas finales: {train_result.metrics}")
```

**Ejecución:**
```bash
cd finetuning_bart_glosas
python scripts/train_round1.py
```

> **Nota:** Si se dispone de múltiples GPUs, se puede usar `accelerate launch scripts/train_round1.py` para distributed training.

---

## 5. Evaluación con Test Set — Ronda 1

### 5.1 Script de Evaluación

> **Instrucción para el agente:** Crea el archivo `finetuning_bart_glosas/scripts/evaluate.py` con el siguiente contenido:

```python
"""
evaluate.py
Evalúa un modelo fine-tuneado sobre un test set y genera ejemplos de predicción.
Sirve tanto para Ronda 1 como para Ronda 2.
"""

import os
import json
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate as hf_evaluate

# ─── Argumentos ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",   required=True, help="Ruta al modelo fine-tuneado")
parser.add_argument("--test_csv",    required=True, help="Ruta al test set (.csv)")
parser.add_argument("--output_file", required=True, help="Ruta para guardar resultados (.json)")
parser.add_argument("--batch_size",  type=int, default=16)
parser.add_argument("--num_beams",   type=int, default=4)
parser.add_argument("--max_src_len", type=int, default=128)
parser.add_argument("--max_tgt_len", type=int, default=128)
parser.add_argument("--num_examples",type=int, default=20, 
                    help="Número de ejemplos detallados a guardar")
args = parser.parse_args()

# ─── Carga modelo ─────────────────────────────────────────────────────────────
print(f"[Evaluación] Cargando modelo desde: {args.model_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"[Evaluación] Dispositivo: {device}")

# ─── Carga dataset ────────────────────────────────────────────────────────────
print(f"[Evaluación] Cargando test set desde: {args.test_csv}")
df = pd.read_csv(args.test_csv).dropna(subset=["source", "target"])
print(f"[Evaluación] Registros en test: {len(df)}")

# ─── Inferencia por lotes ─────────────────────────────────────────────────────
all_predictions = []
all_references  = []

sources = df["source"].tolist()
targets = df["target"].tolist()

print("[Evaluación] Generando predicciones...")
for i in range(0, len(sources), args.batch_size):
    batch_src = sources[i:i+args.batch_size]
    
    inputs = tokenizer(
        batch_src,
        max_length=args.max_src_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=args.num_beams,
            max_length=args.max_tgt_len,
            early_stopping=True,
        )
    
    preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
    all_predictions.extend([p.strip() for p in preds])
    all_references.extend(targets[i:i+args.batch_size])
    
    if (i // args.batch_size) % 5 == 0:
        print(f"   Procesados: {min(i+args.batch_size, len(sources))}/{len(sources)}")

# ─── Cálculo de métricas ──────────────────────────────────────────────────────
print("[Evaluación] Calculando métricas...")
bleu  = hf_evaluate.load("sacrebleu")
rouge = hf_evaluate.load("rouge")

bleu_result  = bleu.compute(
    predictions=all_predictions,
    references=[[r] for r in all_references]
)
rouge_result = rouge.compute(
    predictions=all_predictions,
    references=all_references
)

metrics = {
    "total_samples": len(all_predictions),
    "bleu":          round(bleu_result["score"], 4),
    "bleu_bp":       round(bleu_result["bp"], 4),
    "rouge1":        round(rouge_result["rouge1"], 4),
    "rouge2":        round(rouge_result["rouge2"], 4),
    "rougeL":        round(rouge_result["rougeL"], 4),
}

print("\n─── Resultados ───────────────────────────────────")
for k, v in metrics.items():
    print(f"   {k:20s}: {v}")
print("──────────────────────────────────────────────────")

# ─── Ejemplos detallados ──────────────────────────────────────────────────────
examples = []
for i in range(min(args.num_examples, len(all_predictions))):
    examples.append({
        "source":     sources[i],
        "reference":  all_references[i],
        "prediction": all_predictions[i],
        "exact_match": all_predictions[i].strip() == all_references[i].strip(),
    })

# ─── Guardar resultados ───────────────────────────────────────────────────────
output = {
    "model":    args.model_dir,
    "test_csv": args.test_csv,
    "metrics":  metrics,
    "examples": examples,
}
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Resultados guardados en: {args.output_file}")
```

**Ejecución para Ronda 1:**
```bash
cd finetuning_bart_glosas
python scripts/evaluate.py \
  --model_dir models/round1 \
  --test_csv  data/round1/test.csv \
  --output_file results/round1_test_results.json \
  --num_examples 30
```

### 5.2 Métricas a Observar

| Métrica | Descripción | Umbral sugerido |
|---------|-------------|-----------------|
| **BLEU** | Coincidencia de n-gramas con referencias | > 30 es aceptable para traducción especializada |
| **ROUGE-1** | Solapamiento de unigramas | > 0.5 indica buena cobertura |
| **ROUGE-L** | Subsecuencia más larga común | > 0.4 es razonable |
| **Exact Match** | % de predicciones idénticas a referencia | Métrica complementaria para glosas |

---

## 6. Fine-Tuning Ronda 2 (Dataset 490 registros)

### 6.1 Estrategia de Entrenamiento

La Ronda 2 parte del modelo guardado en `models/round1/` y continúa el ajuste con el dataset especializado de 490 registros. Se recomienda usar un **learning rate menor** para evitar el olvido catastrófico del conocimiento adquirido en Ronda 1.

| Parámetro | Ronda 1 | Ronda 2 | Justificación |
|-----------|---------|---------|---------------|
| Learning rate | `5e-5` | `2e-5` | Menor LR para preservar conocimiento previo |
| Épocas | `10` | `20` | Más épocas para un dataset más pequeño |
| Warmup steps | `200` | `50` | Proporcional al tamaño del dataset |
| Early stopping patience | `3` | `5` | Mayor paciencia para dataset pequeño |
| Batch size | `16` | `8` | Dataset pequeño; puede reducirse si hay overfitting |

### 6.2 Script de Entrenamiento Ronda 2

> **Instrucción para el agente:** Crea el archivo `finetuning_bart_glosas/scripts/train_round2.py` con el siguiente contenido:

```python
"""
train_round2.py
Fine-tuning Ronda 2: continúa desde models/round1/ sobre 490 registros.
"""

import os
import json
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ─── Configuración ────────────────────────────────────────────────────────────
BASE_MODEL_DIR  = "models/round1"   # Punto de partida: modelo de Ronda 1
DATA_DIR        = "data/round2"
OUTPUT_DIR      = "models/round2"
LOGS_DIR        = "logs/round2"
MAX_SRC_LENGTH  = 128
MAX_TGT_LENGTH  = 128
BATCH_SIZE      = 8
NUM_EPOCHS      = 20
LEARNING_RATE   = 2e-5             # Menor que Ronda 1
WARMUP_STEPS    = 50
WEIGHT_DECAY    = 0.01
SEED            = 42

# ─── Carga del modelo desde Ronda 1 ───────────────────────────────────────────
print(f"[Ronda 2] Cargando tokenizer y modelo desde: {BASE_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)

print(f"[Ronda 2] Parámetros del modelo: {model.num_parameters():,}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Ronda 2] Dispositivo: {device}")

# ─── Carga y tokenización del dataset ─────────────────────────────────────────
def load_csv_as_dataset(split: str) -> Dataset:
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    return Dataset.from_pandas(df[["source", "target"]].dropna().reset_index(drop=True))

print("[Ronda 2] Cargando datasets...")
raw_datasets = DatasetDict({
    "train":      load_csv_as_dataset("train"),
    "validation": load_csv_as_dataset("val"),
})
print(f"[Ronda 2] Train: {len(raw_datasets['train'])} | "
      f"Val: {len(raw_datasets['validation'])}")

def preprocess(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_SRC_LENGTH,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=MAX_TGT_LENGTH,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("[Ronda 2] Tokenizando datasets...")
tokenized = raw_datasets.map(
    preprocess,
    batched=True,
    remove_columns=["source", "target"],
    desc="Tokenizando",
)

# ─── Métricas ─────────────────────────────────────────────────────────────────
bleu  = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels         = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    bleu_result  = bleu.compute(predictions=decoded_preds,
                                references=[[l] for l in decoded_labels])
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels)
    return {
        "bleu":   round(bleu_result["score"], 4),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
    }

# ─── Data Collator ─────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

# ─── Argumentos de entrenamiento ──────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    predict_with_generate=True,
    generation_max_length=MAX_TGT_LENGTH,
    generation_num_beams=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=LOGS_DIR,
    logging_steps=20,
    save_total_limit=5,
    seed=SEED,
    report_to="tensorboard",
    fp16=torch.cuda.is_available(),
)

# ─── Trainer ──────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# ─── Entrenamiento ─────────────────────────────────────────────────────────────
print("\n[Ronda 2] Iniciando entrenamiento (continuación desde Ronda 1)...")
train_result = trainer.train()

# ─── Guardado del modelo final ─────────────────────────────────────────────────
print("[Ronda 2] Guardando modelo final...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
    json.dump(train_result.metrics, f, indent=2)

print(f"\n✅ [Ronda 2] Entrenamiento completado. Modelo guardado en: {OUTPUT_DIR}/")
print(f"   Métricas finales: {train_result.metrics}")
```

**Ejecución:**
```bash
cd finetuning_bart_glosas
python scripts/train_round2.py
```

---

## 7. Evaluación Final con Test Set — Ronda 2

**Ejecución:**
```bash
cd finetuning_bart_glosas
python scripts/evaluate.py \
  --model_dir models/round2 \
  --test_csv  data/round2/test.csv \
  --output_file results/round2_test_results.json \
  --num_examples 30
```

### 7.1 Comparación de Resultados entre Rondas

> **Instrucción para el agente:** Crea el archivo `finetuning_bart_glosas/scripts/compare_results.py` con el siguiente contenido:

```python
"""
compare_results.py
Compara métricas entre Ronda 1 y Ronda 2.
"""
import json

def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["metrics"]

r1 = load_metrics("results/round1_test_results.json")
r2 = load_metrics("results/round2_test_results.json")

print("\n╔══════════════════════════════════════════════╗")
print("║     Comparación de Resultados por Ronda      ║")
print("╠══════════════════════════════════════════════╣")
print(f"{'Métrica':<20} {'Ronda 1':>10} {'Ronda 2':>10} {'Δ':>10}")
print("─" * 50)
for k in ["bleu", "rouge1", "rouge2", "rougeL"]:
    v1 = r1.get(k, 0)
    v2 = r2.get(k, 0)
    delta = v2 - v1
    sign  = "+" if delta >= 0 else ""
    print(f"{k:<20} {v1:>10.4f} {v2:>10.4f} {sign+str(round(delta,4)):>10}")
print("╚══════════════════════════════════════════════╝\n")
```

**Ejecución:**
```bash
cd finetuning_bart_glosas
python scripts/compare_results.py
```

---

## 8. Instrucciones para Agente de Código (Claude Code / OpenCode)

A continuación, las instrucciones en orden secuencial para ejecutar el workflow completo a través de un agente de código en terminal.

---

### PASO 0 — Verificar entorno

```
Verifica que Python >= 3.10 esté instalado. Muestra la versión con:
  python --version
Si no está disponible, instálalo o usa conda/pyenv para crear un entorno con Python 3.10.
```

### PASO 1 — Crear entorno virtual e instalar dependencias

```
Crea un entorno virtual llamado 'glosas_env':
  python -m venv glosas_env

Actívalo:
  source glosas_env/bin/activate    (Linux/Mac)
  glosas_env\Scripts\activate       (Windows)

Instala las dependencias:
  pip install transformers==4.40.0 datasets==2.19.0 torch==2.2.2 \
              accelerate==0.29.3 evaluate==0.4.1 sacrebleu==2.4.0 \
              rouge-score==0.1.2 sentencepiece==0.2.0 pandas==2.2.2 \
              scikit-learn==1.4.2 tensorboard==2.16.2 tqdm==4.66.2

Verifica con:
  python -c "import transformers; import datasets; import torch; print('Entorno OK')"
```

### PASO 2 — Crear estructura de directorios

```
Desde el directorio de trabajo, ejecuta:
  mkdir -p finetuning_bart_glosas/{data/{raw,round1,round2},scripts,models/{round1,round2},results,logs/{round1,round2}}

Luego entra al proyecto:
  cd finetuning_bart_glosas
```

### PASO 3 — Colocar los datasets

```
Copia tus archivos CSV en las rutas correctas:
  - Dataset 3.000 registros → data/raw/dataset_3000.csv
  - Dataset 490 registros   → data/raw/dataset_490.csv
  - Dataset de test (Ronda 2) → (se puede agregar después)

Verifica que ambos archivos tengan las columnas 'source' y 'target':
  python -c "
  import pandas as pd
  df1 = pd.read_csv('data/raw/dataset_3000.csv')
  df2 = pd.read_csv('data/raw/dataset_490.csv')
  print('Dataset 3000:', df1.shape, list(df1.columns))
  print('Dataset 490:', df2.shape, list(df2.columns))
  "
```

### PASO 4 — Crear todos los scripts

```
Crea los siguientes archivos Python en la carpeta scripts/:
  - scripts/prepare_data.py    (contenido en sección 3.2)
  - scripts/train_round1.py    (contenido en sección 4.2)
  - scripts/evaluate.py        (contenido en sección 5.1)
  - scripts/train_round2.py    (contenido en sección 6.2)
  - scripts/compare_results.py (contenido en sección 7.1)
```

### PASO 5 — Preparar los datos

```
Ejecuta el script de preparación de datos:
  python scripts/prepare_data.py \
    --round1_input data/raw/dataset_3000.csv \
    --round2_input data/raw/dataset_490.csv

Verifica que se crearon los archivos:
  ls data/round1/
  ls data/round2/
```

### PASO 6 — Ejecutar Fine-Tuning Ronda 1

```
Inicia el entrenamiento de Ronda 1:
  python scripts/train_round1.py

Este proceso puede tomar entre 30 minutos y varias horas dependiendo del hardware.
Puedes monitorear el progreso con TensorBoard en otra terminal:
  tensorboard --logdir logs/round1

Al finalizar, verifica que el modelo fue guardado:
  ls models/round1/
```

### PASO 7 — Evaluar con Test Set (Ronda 1)

```
Ejecuta la evaluación sobre el test set de Ronda 1:
  python scripts/evaluate.py \
    --model_dir models/round1 \
    --test_csv  data/round1/test.csv \
    --output_file results/round1_test_results.json \
    --num_examples 30

Revisa los resultados:
  python -c "
  import json
  with open('results/round1_test_results.json') as f:
      r = json.load(f)
  print('Métricas Ronda 1:')
  for k,v in r['metrics'].items():
      print(f'  {k}: {v}')
  print()
  print('Primeros 3 ejemplos:')
  for ex in r['examples'][:3]:
      print(f'  SRC: {ex[\"source\"]}')
      print(f'  REF: {ex[\"reference\"]}')
      print(f'  PRD: {ex[\"prediction\"]}')
      print()
  "
```

### PASO 8 — Agregar test set externo para Ronda 2 (si aplica)

```
Si tienes un test set externo para la Ronda 2, cópialo a:
  cp /ruta/a/tu/test_externo.csv data/round2/test.csv

Verifica que tenga las columnas correctas:
  python -c "
  import pandas as pd
  df = pd.read_csv('data/round2/test.csv')
  print('Test Ronda 2:', df.shape, list(df.columns))
  "
```

### PASO 9 — Ejecutar Fine-Tuning Ronda 2

```
Inicia el entrenamiento de Ronda 2 (continúa desde el modelo de Ronda 1):
  python scripts/train_round2.py

Monitorea con TensorBoard:
  tensorboard --logdir logs/round2

Al finalizar, verifica:
  ls models/round2/
```

### PASO 10 — Evaluación Final (Ronda 2)

```
Ejecuta la evaluación final:
  python scripts/evaluate.py \
    --model_dir models/round2 \
    --test_csv  data/round2/test.csv \
    --output_file results/round2_test_results.json \
    --num_examples 30

Compara ambas rondas:
  python scripts/compare_results.py
```

### PASO 11 — Verificar resultado final

```
Lista todos los artefactos generados:
  find . -name "*.json" | sort
  find models/ -name "*.bin" -o -name "*.safetensors" | sort

El modelo final listo para producción se encuentra en: models/round2/
```

---

## Notas Finales

**Sobre el modelo base `vgaraujov/bart-base-spanish`:**
El modelo usa el tokenizer de BART adaptado al español. Si el tokenizador reporta `as_target_tokenizer()` como deprecado en versiones recientes de Transformers, reemplaza ese bloque por:

```python
labels = tokenizer(
    text_target=examples["target"],
    max_length=MAX_TGT_LENGTH,
    truncation=True,
    padding=False,
)
```

**Sobre el formato de las glosas:**
Asegúrate de que las glosas en tu dataset sean consistentes en el uso de separadores (espacios, guiones, puntos) para que BLEU y ROUGE sean comparables entre registros.

**Sobre el riesgo de overfitting en Ronda 2:**
Con solo 400 registros de entrenamiento, el modelo puede sobreajustarse rápidamente. Si el `eval_loss` aumenta después de pocas épocas, el early stopping actuará. Si necesitas más regularización, incrementa `weight_decay` a `0.05` en `train_round2.py`.

---

*Documento generado para el proyecto de traducción español → glosas sobre `vgaraujov/bart-base-spanish`.*
