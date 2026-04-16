"""Evaluación del modelo fine-tuned: genera predicciones + métricas BLEU/chrF/METEOR."""

import os

import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from enfoque2 import config
from enfoque2.data_loader import cargar_dataset, split_train_val

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def generar_predicciones(model, tokenizer, pares):
    predicciones = []
    for par in pares:
        inputs = tokenizer(
            par["spa"],
            max_length=config.MAX_SOURCE_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_length=config.MAX_TARGET_LEN,
            num_beams=4,
            early_stopping=True,
        )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicciones.append(pred.strip())
    return predicciones


def evaluar(predicciones, referencias):
    bleu = sacrebleu.corpus_bleu(predicciones, [referencias])
    chrf = sacrebleu.corpus_chrf(predicciones, [referencias])

    meteor_scores = []
    for pred, ref in zip(predicciones, referencias):
        score = meteor_score([ref.split()], pred.split())
        meteor_scores.append(score)
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return {
        "BLEU": round(bleu.score, 4),
        "chrF": round(chrf.score, 4),
        "METEOR": round(meteor_avg * 100, 4),
    }


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(config.OUTPUT_DIR, "best")

    print(f"Cargando modelo desde {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    print("Cargando dataset...")
    pares = cargar_dataset(config.DATASET_PATH)
    _, val_pares = split_train_val(
        pares, train_n=config.TRAIN_SPLIT, seed=config.RANDOM_SEED
    )
    print(f"  Val: {len(val_pares)} pares")

    print("\nGenerando predicciones...")
    referencias = [p["mslg"] for p in val_pares]
    predicciones = generar_predicciones(model, tokenizer, val_pares)

    print("\n── Ejemplos ──")
    for par, pred in zip(val_pares[:8], predicciones[:8]):
        print(f"  SPA:  {par['spa']}")
        print(f"  REF:  {par['mslg']}")
        print(f"  PRED: {pred}")
        print()

    print("── Métricas (SPA→MSLG) ──")
    metricas = evaluar(predicciones, referencias)
    for k, v in metricas.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(config.RESULTS_DIR, "predicciones_val.tsv")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("ID\tSPA\tREF_MSLG\tPRED_MSLG\n")
        for par, pred in zip(val_pares, predicciones):
            f.write(f"{par['id']}\t{par['spa']}\t{par['mslg']}\t{pred}\n")
    print(f"\nPredicciones guardadas en {results_path}")


if __name__ == "__main__":
    main()
