"""Genera submission oficial SPA2MSLG para enfoque5."""

import argparse
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from enfoque5 import config
from enfoque2.submit import cargar_test, evaluar

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def generar_predicciones(model, tokenizer, pares):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    model_path = os.path.join(config.OUTPUT_DIR, "best")
    print(f"Cargando modelo desde {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    print(f"Cargando test: {args.test}")
    pares = cargar_test(args.test)
    print(f"  {len(pares)} instancias")

    print("Generando predicciones...")
    predicciones = generar_predicciones(model, tokenizer, pares)

    referencias = [p["mslg"] for p in pares]
    if all(referencias):
        print("\n── Métricas ──")
        metricas = evaluar(predicciones, referencias)
        for k, v in sorted(metricas.items()):
            print(f"  {k}: {v}")

    output_name = f"{args.team}_{args.run}_SPA2MSLG.txt"
    output_path = os.path.join(config.RESULTS_DIR, output_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for par, pred in zip(pares, predicciones):
            f.write(f'"{par["id"]}"\t"{pred}"\n')

    print(f"\nSubmission: {output_path} ({len(predicciones)} líneas)")


if __name__ == "__main__":
    main()
