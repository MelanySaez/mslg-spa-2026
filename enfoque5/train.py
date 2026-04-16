"""Fine-tuning con curriculum: pretrain (gold×N + silver) → finetune (solo gold)."""

import argparse
import os
import random

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
from enfoque2.train import compute_metrics, _ngram_precision  # noqa: F401


def cargar_augmented(ruta_tsv, train_n):
    """Carga dataset augmentado. Train = primeros train_n, val = resto."""
    import csv

    pares = []
    with open(ruta_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fila in reader:
            pares.append(
                {
                    "id": fila["ID"].strip(),
                    "mslg": fila["MSLG"].strip(),
                    "spa": fila["SPA"].strip(),
                }
            )
    return pares[:train_n], pares[train_n:]


def separar_gold_silver(pares):
    """Gold = IDs sin prefijo AUG. Silver = IDs con AUG (generados por enfoque1)."""
    gold = [p for p in pares if not p["id"].startswith("AUG")]
    silver = [p for p in pares if p["id"].startswith("AUG")]
    return gold, silver


def build_training_args(output_dir, epochs, lr, warmup_steps, phase_label):
    return Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, phase_label),
        num_train_epochs=epochs,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        weight_decay=config.WEIGHT_DECAY,
        label_smoothing_factor=config.LABEL_SMOOTHING,
        warmup_steps=warmup_steps,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena SPA->MSLG con curriculum gold-upsampled + finetune gold-only"
    )
    parser.add_argument("--dataset-path", default=config.AUGMENTED_DATASET)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.dataset_path):
        print("Dataset augmentado no encontrado. Ejecuta primero:")
        print("  python -m enfoque5.augment")
        return

    print(f"Cargando tokenizer y modelo: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)

    with open(args.dataset_path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # sin header
    train_n = total_lines - config.VAL_SPLIT

    print(f"Cargando dataset augmentado: {args.dataset_path}")
    train_pares, val_pares = cargar_augmented(args.dataset_path, train_n)
    gold_train, silver_train = separar_gold_silver(train_pares)
    print(
        f"  Gold: {len(gold_train)} | Silver: {len(silver_train)} | Val: {len(val_pares)}"
    )

    # Upsample gold para equilibrar señal vs silver durante pretrain
    pretrain_pares = gold_train * config.GOLD_UPSAMPLE_FACTOR + silver_train
    random.seed(config.RANDOM_SEED)
    random.shuffle(pretrain_pares)
    print(
        f"  Pretrain: {len(pretrain_pares)} (gold ×{config.GOLD_UPSAMPLE_FACTOR} + silver)"
    )

    pretrain_dataset = MSLGDataset(
        pretrain_pares,
        tokenizer,
        config.MAX_SOURCE_LEN,
        config.MAX_TARGET_LEN,
        task_prefix=config.TASK_PREFIX,
    )
    val_dataset = MSLGDataset(
        val_pares,
        tokenizer,
        config.MAX_SOURCE_LEN,
        config.MAX_TARGET_LEN,
        task_prefix=config.TASK_PREFIX,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    # ── Fase 1: pretrain ──
    print("\n[Fase 1] Pretrain sobre gold upsampled + silver...")
    pretrain_args = build_training_args(
        args.output_dir,
        config.EPOCHS,
        config.LEARNING_RATE,
        config.WARMUP_STEPS,
        "pretrain",
    )
    pretrain_trainer = Seq2SeqTrainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE
            )
        ],
    )
    pretrain_trainer.train()

    pretrain_best_dir = os.path.join(args.output_dir, "pretrain_best")
    pretrain_trainer.save_model(pretrain_best_dir)
    tokenizer.save_pretrained(pretrain_best_dir)
    print(f"Pretrain best → {pretrain_best_dir}")

    del model, pretrain_trainer

    # ── Fase 2: fine-tune solo gold ──
    print("\n[Fase 2] Fine-tune sobre solo gold...")
    model_ft = AutoModelForSeq2SeqLM.from_pretrained(pretrain_best_dir)

    gold_dataset = MSLGDataset(
        gold_train,
        tokenizer,
        config.MAX_SOURCE_LEN,
        config.MAX_TARGET_LEN,
        task_prefix=config.TASK_PREFIX,
    )

    finetune_args = build_training_args(
        args.output_dir,
        config.FINETUNE_EPOCHS,
        config.FINETUNE_LEARNING_RATE,
        config.FINETUNE_WARMUP_STEPS,
        "finetune",
    )
    finetune_trainer = Seq2SeqTrainer(
        model=model_ft,
        args=finetune_args,
        train_dataset=gold_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.FINETUNE_EARLY_STOPPING_PATIENCE
            )
        ],
    )
    finetune_trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    finetune_trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nModelo final guardado en {best_dir}")

    print("\nEvaluación final en validación:")
    metrics = finetune_trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
