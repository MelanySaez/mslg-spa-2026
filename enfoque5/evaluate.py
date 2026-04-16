"""Evaluación enfoque5. Reutiliza métricas de enfoque2."""

import argparse
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from enfoque5 import config
from enfoque2.data_loader import cargar_dataset, split_train_val
from enfoque2.evaluate import evaluar

import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def generar_predicciones(model, tokenizer, pares):
    """Wrapper → usa config de enfoque5 (mismos params de generación)."""
    predicciones = []
    for par in pares:
        inputs = tokenizer(
            config.TASK_PREFIX + par["spa"],
            max_length=config.MAX_SOURCE_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_length=config.MAX_TARGET_LEN,
            num_beams=config.NUM_BEAMS,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            length_penalty=config.LENGTH_PENALTY,
            early_stopping=True,
        )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicciones.append(pred.strip())
    return predicciones


def parse_args():
    parser = argparse.ArgumentParser(description="Evalúa un modelo de enfoque5")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument(
        "--results-path",
        default=os.path.join(config.RESULTS_DIR, "predicciones_val.tsv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    model_path = os.path.join(args.output_dir, "best")

    print(f"Cargando modelo desde {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # Evaluar sobre val split ORIGINAL (gold), no augmentado
    print("Cargando dataset original...")
    pares = cargar_dataset(config.ORIGINAL_DATASET)
    _, val_pares = split_train_val(
        pares, train_n=config.TRAIN_SPLIT, seed=config.RANDOM_SEED
    )
    print(f"  Val: {len(val_pares)} pares (gold)")

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
    for k, v in sorted(metricas.items()):
        print(f"  {k}: {v}")

    results_path = args.results_path
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("ID\tSPA\tREF_MSLG\tPRED_MSLG\n")
        for par, pred in zip(val_pares, predicciones):
            f.write(f"{par['id']}\t{par['spa']}\t{par['mslg']}\t{pred}\n")
    print(f"\nPredicciones guardadas en {results_path}")


if __name__ == "__main__":
    main()
