"""Fine-tuning con datos augmentados. Reutiliza lógica de enfoque2."""

import os

import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from enfoque5 import config
from enfoque2.data_loader import MSLGDataset
from enfoque2.train import compute_metrics, _ngram_precision


def cargar_augmented(ruta_tsv, train_n):
    """Carga dataset augmentado. Train = primeros train_n, val = resto."""
    import csv
    pares = []
    with open(ruta_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fila in reader:
            pares.append({
                "id": fila["ID"].strip(),
                "mslg": fila["MSLG"].strip(),
                "spa": fila["SPA"].strip(),
            })
    return pares[:train_n], pares[train_n:]


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(config.AUGMENTED_DATASET):
        print("Dataset augmentado no encontrado. Ejecuta primero:")
        print("  python -m enfoque5.augment")
        return

    print(f"Cargando tokenizer y modelo: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)

    # Contar líneas para saber cuántas son train
    with open(config.AUGMENTED_DATASET, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # sin header
    train_n = total_lines - config.VAL_SPLIT

    print(f"Cargando dataset augmentado: {config.AUGMENTED_DATASET}")
    train_pares, val_pares = cargar_augmented(config.AUGMENTED_DATASET, train_n)
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

    print(f"\nIniciando fine-tuning ({len(train_pares)} muestras)...")
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
