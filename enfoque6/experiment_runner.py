"""Orquestador de experimentos SPA→MSLG — Enfoque 6.

Tipos de experimento:
  zero_shot   — prompt con reglas LSM extendidas, sin ejemplos.
  few_shot    — reglas + k ejemplos fijos.
  hybrid_zero — reglas + borrador del motor FOL + anotaciones.
  hybrid_few  — reglas + borrador FOL + k ejemplos fijos.
  rag         — reglas + k ejemplos recuperados semánticamente.
  rag_hybrid  — reglas + borrador FOL + k ejemplos recuperados.
"""

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
from rules_engine import RulesEngine


def run_experiment(experiment: dict, pool: list, val: list,
                   engine: RulesEngine = None, emb_index=None):
    """
    Ejecuta un único experimento sobre el conjunto de validación.

    Returns:
        (results, metrics)
    """
    exp_name = experiment["name"]
    exp_type = experiment["type"]
    k = experiment["k"]
    is_hybrid = exp_type in ("hybrid_zero", "hybrid_few", "rag_hybrid")

    print(f"\n{'='*60}")
    print(f"  Experimento: {exp_name} (tipo={exp_type}, k={k})")
    print(f"{'='*60}")

    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        spa = par["spa"]
        mslg_real = par["mslg"]

        # Análisis del motor de reglas (solo para experimentos híbridos)
        analysis = engine.analyze(spa) if (is_hybrid and engine) else None

        # Construir prompt
        if exp_type == "zero_shot":
            prompt = prompt_builder.build_zero_shot(spa)
        elif exp_type == "few_shot":
            prompt = prompt_builder.build_few_shot(spa, k=k)
        elif exp_type == "hybrid_zero":
            prompt = prompt_builder.build_hybrid_zero(spa, analysis)
        elif exp_type == "hybrid_few":
            prompt = prompt_builder.build_hybrid_few(spa, analysis, k=k)
        elif exp_type == "rag":
            ejemplos = emb_index.retrieve(spa, k=k)
            prompt = prompt_builder.build_rag(spa, ejemplos)
        elif exp_type == "rag_hybrid":
            ejemplos = emb_index.retrieve(spa, k=k)
            prompt = prompt_builder.build_rag_hybrid(spa, analysis, ejemplos)
        else:
            raise ValueError(f"Tipo desconocido: {exp_type}")

        raw_response = ollama_client.translate(prompt)
        mslg_pred = post_processor.clean(raw_response)

        result = {
            "id": par["id"],
            "spa": spa,
            "mslg_real": mslg_real,
            "mslg_pred": mslg_pred,
            "raw_response": raw_response,
        }
        if analysis:
            result["draft"] = analysis["draft"]
        results.append(result)

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(val) - i - 1)
        print(
            f"  [{exp_name}] {i+1}/{len(val)} "
            f"({elapsed:.0f}s, ~{remaining:.0f}s restantes) | "
            f"SPA: {spa[:45]}... | PRED: {mslg_pred[:55]}"
        )

    metrics = evaluator.evaluate(results)
    total_time = time.time() - start_time

    print(f"\n  === Resultados {exp_name} ({total_time:.1f}s) ===")
    print(f"  BLEU:   {metrics['bleu']:.4f}")
    print(f"  METEOR: {metrics['meteor']:.4f}")
    print(f"  chrF:   {metrics['chrf']:.4f}")

    _save_results_csv(results, exp_name)
    _save_metrics_json(metrics, exp_name, total_time)

    return results, metrics


def run_all(experiments=None):
    """Ejecuta todos los experimentos y genera tabla resumen."""
    experiments = experiments or config.EXPERIMENTS

    pool, val = data_loader.split_dataset()

    need_hybrid = any(e["type"] in ("hybrid_zero", "hybrid_few", "rag_hybrid")
                      for e in experiments)
    need_rag = any(e["type"] in ("rag", "rag_hybrid") for e in experiments)

    engine = None
    emb_index = None

    if need_hybrid:
        print("\nCargando motor de reglas FOL (spaCy)...")
        engine = RulesEngine(corpus_pairs=pool)

    if need_rag:
        print("\nConstruyendo índice de embeddings...")
        emb_index = emb_mod.EmbeddingIndex(pool)

    summary = []
    for experiment in experiments:
        _, metrics = run_experiment(experiment, pool, val, engine, emb_index)
        summary.append({"name": experiment["name"], **metrics})

    _print_summary(summary)
    _save_summary_csv(summary)

    return summary


def _save_results_csv(results: list, exp_name: str):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    fieldnames = ["id", "spa", "mslg_real", "mslg_pred", "draft", "raw_response"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Resultados guardados: {path}")


def _save_metrics_json(metrics: dict, exp_name: str, total_time: float):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json")
    data = {**metrics, "experiment": exp_name, "total_time_seconds": round(total_time, 1)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Métricas guardadas:   {path}")


def _save_summary_csv(summary: list):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nResumen guardado: {path}")


def _print_summary(summary: list):
    print(f"\n{'='*60}")
    print("  RESUMEN — ENFOQUE 6 (deepseek-r1:70b + Reglas LSM + Híbrido)")
    print(f"{'='*60}")
    print(f"  {'Experimento':<22} {'BLEU':>8} {'METEOR':>8} {'chrF':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for row in summary:
        print(
            f"  {row['name']:<22} {row['bleu']:>8.4f} "
            f"{row['meteor']:>8.4f} {row['chrf']:>8.4f}"
        )
    print(f"{'='*60}")
