"""Genera archivo de submission oficial para SPA2MSLG.

Uso:
    python -m enfoque2.submit --test <ruta_test.txt> --team <TeamName> --run <SolutionName>

El archivo de test debe ser el TSV oficial con columnas ID y SPA (sin columna MSLG).
Si el archivo incluye columna MSLG (test con referencia), también se calculan métricas.

Salida: <TeamName>_<SolutionName>_SPA2MSLG.txt
Formato por línea: "InstanceID"\t"SystemOutput"\n
"""

import argparse
import csv
import os

import nltk
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from enfoque2 import config

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def cargar_test(ruta_tsv: str):
    """Carga archivo de test. Acepta columnas: ID, SPA (y opcionalmente MSLG)."""
    pares = []
    with open(ruta_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fila in reader:
            par = {
                "id": fila["ID"].strip(),
                "spa": fila["SPA"].strip(),
                "mslg": fila.get("MSLG", "").strip(),
            }
            pares.append(par)
    return pares


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


def evaluar(predicciones, referencias):
    bleu = sacrebleu.corpus_bleu(predicciones, [referencias])
    chrf = sacrebleu.corpus_chrf(predicciones, [referencias])

    meteor_scores = []
    for pred, ref in zip(predicciones, referencias):
        score = meteor_score([ref.split()], pred.split())
        meteor_scores.append(score)
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return {
        "BLEU":   round(bleu.score, 4),
        "chrF":   round(chrf.score, 4),
        "METEOR": round(meteor_avg * 100, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  required=True, help="Ruta al archivo de test oficial (.txt/.tsv)")
    parser.add_argument("--team",  required=True, help="Nombre del equipo")
    parser.add_argument("--run",   required=True, help="Nombre de la solución/run")
    args = parser.parse_args()

    model_path = os.path.join(config.OUTPUT_DIR, "best")
    print(f"Cargando modelo desde {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    print(f"Cargando test set: {args.test}")
    pares = cargar_test(args.test)
    print(f"  {len(pares)} instancias")

    print("Generando predicciones...")
    predicciones = generar_predicciones(model, tokenizer, pares)

    # ── Métricas (solo si el test tiene referencias) ──
    referencias = [p["mslg"] for p in pares]
    if all(referencias):
        print("\n── Métricas ──")
        metricas = evaluar(predicciones, referencias)
        for k, v in sorted(metricas.items()):
            print(f"  {k}: {v}")

    # ── Guardar submission ──
    output_name = f"{args.team}_{args.run}_SPA2MSLG.txt"
    output_path = os.path.join(config.RESULTS_DIR, output_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for par, pred in zip(pares, predicciones):
            f.write(f'"{par["id"]}"\t"{pred}"\n')

    print(f"\nSubmission guardada en: {output_path}")
    print(f"Instancias escritas: {len(predicciones)}")


if __name__ == "__main__":
    main()
