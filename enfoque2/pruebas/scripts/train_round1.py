"""Fine-tuning Ronda 1: vgaraujov/bart-base-spanish sobre 3000 pares esp→lsm."""

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
DATA_DIR = os.path.join(PRUEBAS_DIR, "data", "round1")
OUTPUT_DIR = os.path.join(PRUEBAS_DIR, "models", "round1")
LOGS_DIR = os.path.join(PRUEBAS_DIR, "logs", "round1")

MODEL_NAME = "vgaraujov/bart-base-spanish"
MAX_SRC_LENGTH = 128
MAX_TGT_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
SEED = 42


def load_csv_as_dataset(split):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    df = df[["source", "target"]].dropna().reset_index(drop=True)
    return Dataset.from_pandas(df)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"[R1] Cargando tokenizer y modelo: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"[R1] Params: {model.num_parameters():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[R1] Dispositivo: {device}")

    print("[R1] Cargando datasets...")
    raw = DatasetDict({
        "train": load_csv_as_dataset("train"),
        "validation": load_csv_as_dataset("val"),
        "test": load_csv_as_dataset("test"),
    })
    print(f"[R1] Train:{len(raw['train'])} Val:{len(raw['validation'])} Test:{len(raw['test'])}")

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

    print("[R1] Tokenizando...")
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
        logging_steps=50,
        save_total_limit=3,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n[R1] Entrenando...")
    train_result = trainer.train()

    print("[R1] Guardando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f"\n[R1] Modelo guardado en: {OUTPUT_DIR}")
    print(f"   Métricas: {train_result.metrics}")


if __name__ == "__main__":
    main()
