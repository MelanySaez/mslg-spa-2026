"""Shim a enfoque3/generate_submission.py adaptado a Anthropic"""
import argparse
import csv
import os
import time

import config
import embedding_index as emb_mod
import data_loader
import anthropic_client
import post_processor
import prompt_builder

def load_test_set(path):
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({"id": row["ID"], "spa": row["SPA"].strip()})
    return data

def generate(experiment_name, test_path):
    exp_config = next((e for e in config.EXPERIMENTS if e["name"] == experiment_name), None)
    if exp_config is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado.")

    exp_type = exp_config["type"]
    k, pool, _ = exp_config["k"], *data_loader.split_dataset()
    
    test_data = load_test_set(test_path)
    print(f"Test set cargado: {len(test_data)} oraciones")

    emb_index = emb_mod.EmbeddingIndex(pool) if exp_type == "rag" else None

    predictions, start = [], time.time()
    for i, item in enumerate(test_data):
        spa = item["spa"]
        if exp_type == "rag":
            ejemplos = emb_index.retrieve(spa, k=k)
            system_prompt, user_prompt = prompt_builder.build_rag(spa, ejemplos)
        else:
             raise ValueError("Solo soportado RAG en este generador.")
             
        raw = anthropic_client.translate(system_prompt, user_prompt)
        pred = post_processor.clean(raw)
        predictions.append(pred)

        elapsed = time.time() - start
        print(f"  [{i+1}/{len(test_data)}] ({elapsed:.0f}s) SPA: {spa[:40]}... → {pred[:50]}")

    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    tag = experiment_name.upper().replace("-", "")
    filepath = os.path.join(config.SUBMISSIONS_DIR, f"UTB_{tag}_SPA2MSLG.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f'"{pred}"\n')

    print(f"\nSubmission generado: {filepath} ({len(predictions)} predicciones en {time.time() - start:.1f}s)")
    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--test-file", required=True)
    args = parser.parse_args()
    generate(args.experiment, args.test_file)
