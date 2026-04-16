"""Fine-tuning de bart-base-spanish para traducción SPA→MSLG."""

import os
from collections import Counter

import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from enfoque2 import config
from enfoque2.data_loader import cargar_dataset, split_train_val, MSLGDataset


def _ngram_precision(pred_tokens, ref_tokens, n):
    """Calcula precisión de n-gramas (clipped) para un par pred/ref."""
    pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
    if not pred_ngrams:
        return 0.0
    ref_counts = Counter(ref_ngrams)
    clipped = 0
    pred_counts = Counter(pred_ngrams)
    for ng, count in pred_counts.items():
        clipped += min(count, ref_counts.get(ng, 0))
    return clipped / len(pred_ngrams)


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    for pred, ref in zip(decoded_preds, decoded_labels):
        pred_tok = pred.split()
        ref_tok = ref.split()
        for n in range(1, 5):
            bleu_scores[n].append(_ngram_precision(pred_tok, ref_tok, n))

    results = {}
    for n in range(1, 5):
        scores = bleu_scores[n]
        results[f"bleu_{n}"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    return results


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Cargando tokenizer y modelo: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)

    print("Cargando dataset...")
    pares = cargar_dataset(config.DATASET_PATH)
    train_pares, val_pares = split_train_val(
        pares, train_n=config.TRAIN_SPLIT, seed=config.RANDOM_SEED
    )
    print(f"  Train: {len(train_pares)} | Val: {len(val_pares)}")

    train_dataset = MSLGDataset(
        train_pares, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN,
        task_prefix=config.TASK_PREFIX,
    )
    val_dataset = MSLGDataset(
        val_pares, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN,
        task_prefix=config.TASK_PREFIX,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        label_smoothing_factor=config.LABEL_SMOOTHING,
        warmup_steps=config.WARMUP_STEPS,
        fp16=config.FP16,
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER,
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_LEN,
        generation_num_beams=config.NUM_BEAMS,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )

    print("\nIniciando fine-tuning...")
    trainer.train()

    print("\nGuardando mejor modelo...")
    best_dir = os.path.join(config.OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Modelo guardado en {best_dir}")

    print("\nEvaluación final en validación:")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
