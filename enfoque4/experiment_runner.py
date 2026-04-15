"""Orquestador independiente de enfoque4.

Solo ejecuta las configuraciones FOL-RAG-7 y FOL-RAG-10. Reusa la KB
(EmbeddingIndex), el cliente Ollama, el post-procesador y el evaluador
de enfoque3.

Para cada oración de validación:
  1. rule_engine.generar_gloss_fol → candidato FOL.
  2. embedding_index.retrieve(k) → ejemplos semánticamente similares.
  3. prompt_builder.build_fol_rag(spa, ejemplos, gloss_fol) → prompt.
  4. ollama_client.translate → respuesta cruda del LLM.
  5. post_processor.clean → glosa final.
  6. evaluator.evaluate → BLEU / METEOR / chrF.
"""

import csv
import json
import os
import time

import data_loader            # enfoque3
import embedding_index as emb_mod  # enfoque3
import evaluator              # enfoque3
import ollama_client          # enfoque3
import post_processor         # enfoque3

from . import config
from . import prompt_builder as fol_prompts
from . import rule_engine


def run_experiment(experiment, pool, val, emb_index, nlp, dicc_compuestos, nombres_personas):
    exp_name = experiment["name"]
    k = experiment["k"]

    print(f"\n{'=' * 60}")
    print(f"  Experimento: {exp_name} (tipo=fol_rag, k={k})")
    print(f"{'=' * 60}")

    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        spa = par["spa"]
        mslg_real = par["mslg"]

        gloss_fol = rule_engine.generar_gloss_fol(
            spa, nlp, dicc_compuestos, nombres_personas
        )
        ejemplos = emb_index.retrieve(spa, k=k)
        prompt = fol_prompts.build_fol_rag(spa, ejemplos, gloss_fol)

        raw_response = ollama_client.translate(prompt)
        mslg_pred = post_processor.clean(raw_response)

        results.append({
            "id": par["id"],
            "spa": spa,
            "mslg_real": mslg_real,
            "mslg_pred": mslg_pred,
            "gloss_fol": gloss_fol,
            "raw_response": raw_response,
        })

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(val) - i - 1)
        print(
            f"  [{exp_name}] {i + 1}/{len(val)} "
            f"({elapsed:.0f}s, ~{remaining:.0f}s restantes) | "
            f"SPA: {spa[:50]}... | PRED: {mslg_pred[:60]}"
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
    experiments = experiments or config.EXPERIMENTS

    pool, val = data_loader.split_dataset()

    print("\nCargando modelo spaCy para reglas FOL...")
    nlp = rule_engine.cargar_nlp(config.SPACY_MODEL)
    dicc_compuestos = rule_engine.construir_dicc_compuestos(pool)
    nombres_personas = rule_engine.construir_dicc_nombres(pool)
    print(f"  Diccionario de compuestos: {len(dicc_compuestos)} entradas")
    print(f"  Nombres de personas:       {len(nombres_personas)} entradas")

    emb_index = emb_mod.EmbeddingIndex(pool)

    summary = []
    for experiment in experiments:
        _, metrics = run_experiment(
            experiment, pool, val, emb_index, nlp, dicc_compuestos, nombres_personas
        )
        summary.append({"name": experiment["name"], **metrics})

    _print_summary(summary)
    _save_summary_csv(summary)

    return summary


def _save_results_csv(results, exp_name):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "spa", "mslg_real", "mslg_pred", "gloss_fol", "raw_response"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"  Resultados guardados: {path}")


def _save_metrics_json(metrics, exp_name, total_time):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json")
    data = {**metrics, "experiment": exp_name, "total_time_seconds": round(total_time, 1)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Métricas guardadas:   {path}")


def _save_summary_csv(summary):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nResumen guardado: {path}")


def _print_summary(summary):
    print(f"\n{'=' * 60}")
    print("  RESUMEN ENFOQUE4 (FOL-RAG)")
    print(f"{'=' * 60}")
    print(f"  {'Experimento':<18} {'BLEU':>8} {'METEOR':>8} {'chrF':>8}")
    print(f"  {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8}")
    for row in summary:
        print(f"  {row['name']:<18} {row['bleu']:>8.4f} {row['meteor']:>8.4f} {row['chrf']:>8.4f}")
    print(f"{'=' * 60}")
