"""Genera el archivo de submission oficial para MSLG-SPA 2026."""

import argparse
import csv
import os
import time

import config
import embedding_index as emb_mod
import data_loader
import ollama_client
import post_processor
import prompt_builder


def load_test_set(path):
    """Carga el test set (TSV con columnas ID, SPA — sin MSLG)."""
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({
                "id": row["ID"],
                "spa": row["SPA"].strip(),
            })
    return data


def generate(experiment_name, test_path):
    """
    Genera predicciones para el test set usando la configuración del experimento.

    Args:
        experiment_name: nombre del experimento (e.g. "rag-7").
        test_path: ruta al archivo de test set.
    """
    # Encontrar la configuración del experimento
    exp_config = None
    for exp in config.EXPERIMENTS:
        if exp["name"] == experiment_name:
            exp_config = exp
            break

    if exp_config is None:
        raise ValueError(
            f"Experimento '{experiment_name}' no encontrado. "
            f"Disponibles: {[e['name'] for e in config.EXPERIMENTS]}"
        )

    exp_type = exp_config["type"]
    k = exp_config["k"]

    # Cargar pool (para few-shot/RAG)
    pool, _ = data_loader.split_dataset()

    # Cargar test set
    test_data = load_test_set(test_path)
    print(f"Test set cargado: {len(test_data)} oraciones")

    # Inicializar embedding index si es RAG
    emb_index = None
    if exp_type == "rag":
        emb_index = emb_mod.EmbeddingIndex(pool)

    # Traducir
    predictions = []
    start = time.time()

    for i, item in enumerate(test_data):
        spa = item["spa"]

        if exp_type == "zero_shot":
            prompt = prompt_builder.build_zero_shot(spa)
        elif exp_type == "few_shot":
            prompt = prompt_builder.build_few_shot(spa, k=k)
        elif exp_type == "rag":
            ejemplos = emb_index.retrieve(spa, k=k)
            prompt = prompt_builder.build_rag(spa, ejemplos)
        else:
            raise ValueError(f"Tipo desconocido: {exp_type}")

        raw = ollama_client.translate(prompt)
        pred = post_processor.clean(raw)
        predictions.append(pred)

        elapsed = time.time() - start
        print(f"  [{i+1}/{len(test_data)}] ({elapsed:.0f}s) SPA: {spa[:50]}... → {pred[:60]}")

    # Escribir archivo de submission
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    tag = experiment_name.upper().replace("-", "")
    filename = f"UTB_{tag}_SPA2MSLG.txt"
    filepath = os.path.join(config.SUBMISSIONS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f'"{pred}"\n')

    total = time.time() - start
    print(f"\nSubmission generado: {filepath}")
    print(f"  {len(predictions)} predicciones en {total:.1f}s")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Genera submission MSLG-SPA 2026")
    parser.add_argument(
        "--experiment", required=True,
        help="Nombre del experimento a usar (e.g. rag-7)",
    )
    parser.add_argument(
        "--test-file", required=True,
        help="Ruta al archivo de test set TSV",
    )
    args = parser.parse_args()

    generate(args.experiment, args.test_file)


if __name__ == "__main__":
    main()
