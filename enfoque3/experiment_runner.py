"""Orquestador de experimentos SPA→MSLG."""

import csv
import json
import os
import time

import config
import data_loader
import embedding_index as emb_mod
import evaluator
import ollama_client
import post_processor
import prompt_builder


def run_experiment(experiment, pool, val, emb_index=None):
    """
    Ejecuta un único experimento sobre el conjunto de validación.

    Args:
        experiment: dict con 'name', 'type', 'k'.
        pool: pool de ejemplos para few-shot.
        val: conjunto de validación.
        emb_index: EmbeddingIndex (solo necesario para tipo 'rag').

    Returns:
        (results, metrics) — lista de resultados y dict de métricas.
    """
    exp_name = experiment["name"]
    exp_type = experiment["type"]
    k = experiment["k"]

    print(f"\n{'='*60}")
    print(f"  Experimento: {exp_name} (tipo={exp_type}, k={k})")
    print(f"{'='*60}")

    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        spa = par["spa"]
        mslg_real = par["mslg"]

        # 1. Construir prompt
        if exp_type == "zero_shot":
            prompt = prompt_builder.build_zero_shot(spa)
        elif exp_type == "few_shot":
            prompt = prompt_builder.build_few_shot(spa, k=k)
        elif exp_type == "rag":
            ejemplos = emb_index.retrieve(spa, k=k)
            prompt = prompt_builder.build_rag(spa, ejemplos)
        else:
            raise ValueError(f"Tipo de experimento desconocido: {exp_type}")

        # 2. Llamar a Ollama
        raw_response = ollama_client.translate(prompt)

        # 3. Post-procesar
        mslg_pred = post_processor.clean(raw_response)

        # 4. Guardar resultado
        results.append({
            "id": par["id"],
            "spa": spa,
            "mslg_real": mslg_real,
            "mslg_pred": mslg_pred,
            "raw_response": raw_response,
        })

        # 5. Progreso
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(val) - i - 1)
        print(
            f"  [{exp_name}] {i+1}/{len(val)} "
            f"({elapsed:.0f}s, ~{remaining:.0f}s restantes) | "
            f"SPA: {spa[:50]}... | PRED: {mslg_pred[:60]}"
        )

    # 6. Evaluar
    metrics = evaluator.evaluate(results)

    total_time = time.time() - start_time
    print(f"\n  === Resultados {exp_name} ({total_time:.1f}s) ===")
    print(f"  BLEU:   {metrics['bleu']:.4f}")
    print(f"  METEOR: {metrics['meteor']:.4f}")
    print(f"  chrF:   {metrics['chrf']:.4f}")

    # 7. Guardar archivos
    _save_results_csv(results, exp_name)
    _save_metrics_json(metrics, exp_name, total_time)

    return results, metrics


def run_all(experiments=None):
    """Ejecuta todos los experimentos y genera tabla resumen."""
    experiments = experiments or config.EXPERIMENTS

    # Cargar datos
    pool, val = data_loader.split_dataset()

    # Determinar si necesitamos embeddings (algún experimento RAG)
    need_rag = any(e["type"] == "rag" for e in experiments)
    emb_index = None
    if need_rag:
        emb_index = emb_mod.EmbeddingIndex(pool)

    summary = []
    for experiment in experiments:
        _, metrics = run_experiment(experiment, pool, val, emb_index)
        summary.append({"name": experiment["name"], **metrics})

    # Tabla resumen
    _print_summary(summary)
    _save_summary_csv(summary)

    return summary


def _save_results_csv(results, exp_name):
    """Guarda resultados detallados por oración."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "spa", "mslg_real", "mslg_pred", "raw_response"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Resultados guardados: {path}")


def _save_metrics_json(metrics, exp_name, total_time):
    """Guarda métricas agregadas en JSON."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json")
    data = {**metrics, "experiment": exp_name, "total_time_seconds": round(total_time, 1)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Métricas guardadas:   {path}")


def _save_summary_csv(summary):
    """Guarda tabla comparativa de todos los experimentos."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nResumen guardado: {path}")


def _print_summary(summary):
    """Imprime tabla resumen formateada."""
    print(f"\n{'='*60}")
    print("  RESUMEN DE EXPERIMENTOS")
    print(f"{'='*60}")
    print(f"  {'Experimento':<18} {'BLEU':>8} {'METEOR':>8} {'chrF':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}")
    for row in summary:
        print(f"  {row['name']:<18} {row['bleu']:>8.4f} {row['meteor']:>8.4f} {row['chrf']:>8.4f}")
    print(f"{'='*60}")
