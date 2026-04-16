"""Fine-tuning de bart-base-spanish para traducción SPA→MSLG."""

import os

import numpy as np
import sacrebleu
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from enfoque2 import config
from enfoque2.data_loader import cargar_dataset, split_train_val, MSLGDataset


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    return {
        "bleu": round(bleu.score, 4),
        "chrf": round(chrf.score, 4),
    }


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
        train_pares, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN
    )
    val_dataset = MSLGDataset(
        val_pares, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=int(config.WARMUP_RATIO * config.EPOCHS * (config.TRAIN_SPLIT // config.BATCH_SIZE)),
        fp16=config.FP16,
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_LEN,
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
