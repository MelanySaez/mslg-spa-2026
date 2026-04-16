"""Fine-tuning Ronda 2: continúa desde models/round1/ sobre 490 pares SPA→MSLG."""

import json
import os

import numpy as np
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRUEBAS_DIR = os.path.dirname(BASE_DIR)
BASE_MODEL_DIR = os.path.join(PRUEBAS_DIR, "models", "round1")
DATA_DIR = os.path.join(PRUEBAS_DIR, "data", "round2")
OUTPUT_DIR = os.path.join(PRUEBAS_DIR, "models", "round2")
LOGS_DIR = os.path.join(PRUEBAS_DIR, "logs", "round2")

MAX_SRC_LENGTH = 128
MAX_TGT_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
SEED = 42


def load_csv_as_dataset(split):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    df = df[["source", "target"]].dropna().reset_index(drop=True)
    return Dataset.from_pandas(df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(BASE_MODEL_DIR):
        print(f"[R2] Falta modelo de Ronda 1 en {BASE_MODEL_DIR}")
        print("     Ejecuta primero: uv run python scripts/train_round1.py")
        return

    print(f"[R2] Cargando modelo de Ronda 1: {BASE_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[R2] Dispositivo: {device}")

    print("[R2] Cargando datasets...")
    raw = DatasetDict({
        "train": load_csv_as_dataset("train"),
        "validation": load_csv_as_dataset("val"),
    })
    print(f"[R2] Train:{len(raw['train'])} Val:{len(raw['validation'])}")

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["source"],
            max_length=MAX_SRC_LENGTH,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=MAX_TGT_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[R2] Tokenizando...")
    tokenized = raw.map(
        preprocess, batched=True, remove_columns=["source", "target"], desc="Tokenizando"
    )

    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        bleu_r = bleu.compute(
            predictions=decoded_preds, references=[[l] for l in decoded_labels]
        )
        rouge_r = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "bleu": round(bleu_r["score"], 4),
            "rouge1": round(rouge_r["rouge1"], 4),
            "rougeL": round(rouge_r["rougeL"], 4),
        }

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
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
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\n[R2] Continuando desde Ronda 1...")
    train_result = trainer.train()

    print("[R2] Guardando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f"\n[R2] Modelo guardado en: {OUTPUT_DIR}")
    print(f"   Métricas: {train_result.metrics}")


if __name__ == "__main__":
    main()
