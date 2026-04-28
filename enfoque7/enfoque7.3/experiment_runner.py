"""Orquestador enfoque7.3 — réplica de enfoque7.1 sobre modelo Ollama local.

Mismo dispatch (`few_shot_rag_curriculum` SPA→MSLG, k=10) que enfoque7.1, pero
llamando a `ollama_client.translate(system, user)` en lugar del cliente
Anthropic. Mantiene la generación del archivo de submission oficial
`TeamName_SolutionName_SPA2MSLG.txt`.
"""

import csv
import json
import os
import time

import config
import data_loader
import evaluator
import ollama_client
import post_processor
import prompt_builder


def _build_prompt(exp_type: str, k: int, spa: str,
                  diverse_examples=None, emb_index=None):
    if exp_type == "zero_shot":
        return prompt_builder.build_zero_shot(spa)
    if exp_type == "zero_shot_cot":
        return prompt_builder.build_zero_shot_cot(spa)
    if exp_type == "zero_shot_glossary":
        return prompt_builder.build_zero_shot_glossary(spa)
    if exp_type == "zero_shot_full":
        return prompt_builder.build_zero_shot_full(spa)
    if exp_type == "few_shot":
        return prompt_builder.build_few_shot(spa, k=k)
    if exp_type == "few_shot_cot":
        return prompt_builder.build_few_shot_cot(spa, k=k)
    if exp_type == "few_shot_negative":
        return prompt_builder.build_few_shot_negative(spa, k=k)
    if exp_type == "few_shot_curriculum":
        return prompt_builder.build_few_shot_curriculum(spa, k=k)
    if exp_type == "few_shot_diverse":
        return prompt_builder.build_few_shot_diverse(spa, diverse_examples)
    if exp_type == "few_shot_full":
        return prompt_builder.build_few_shot_full(spa, k=k)
    if exp_type == "few_shot_rag":
        retrieved = emb_index.retrieve(spa, k=k)
        return prompt_builder.build_few_shot_rag(spa, retrieved)
    if exp_type == "few_shot_rag_curriculum":
        retrieved = emb_index.retrieve(spa, k=k)
        return prompt_builder.build_few_shot_rag_curriculum(spa, retrieved)
    raise ValueError(f"Tipo desconocido: {exp_type}")


def run_experiment(experiment: dict, pool: list, val: list, emb_index=None):
    exp_name = experiment["name"]
    exp_type = experiment["type"]
    k = experiment["k"]
    needs_diverse = exp_type == "few_shot_diverse"

    print(f"\n{'='*60}")
    print(f"  Experimento: {exp_name} (tipo={exp_type}, k={k})")
    print(f"  Modelo (Ollama): {config.OLLAMA_MODEL}")
    print(f"{'='*60}")

    diverse_examples = None
    if needs_diverse and emb_index is not None:
        diverse_examples = emb_index.select_diverse(k=k)

    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        spa = par["spa"]
        mslg_real = par["mslg"]

        system_prompt, user_prompt = _build_prompt(
            exp_type, k, spa,
            diverse_examples=diverse_examples,
            emb_index=emb_index,
        )
        raw_response = ollama_client.translate(system_prompt, user_prompt)
        mslg_pred = post_processor.clean(raw_response)

        results.append({
            "id": par["id"],
            "spa": spa,
            "mslg_real": mslg_real,
            "mslg_pred": mslg_pred,
            "raw_response": raw_response,
        })

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
    _save_submission_txt(results, exp_name)

    return results, metrics


def run_all(experiments=None):
    experiments = experiments or config.EXPERIMENTS

    pool, val = data_loader.split_dataset()

    need_embeddings = any(
        e["type"] in ("few_shot_diverse", "few_shot_rag",
                      "few_shot_rag_curriculum")
        for e in experiments
    )
    emb_index = None
    if need_embeddings:
        import embedding_index as emb_mod
        print("\nConstruyendo índice de embeddings (rag/rag-curriculum)...")
        emb_index = emb_mod.EmbeddingIndex(pool)

    summary = []
    for experiment in experiments:
        _, metrics = run_experiment(experiment, pool, val, emb_index)
        summary.append({"name": experiment["name"], **metrics})

    _print_summary(summary)
    _save_summary_csv(summary)

    return summary


def _save_results_csv(results: list, exp_name: str):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    fieldnames = ["id", "spa", "mslg_real", "mslg_pred", "raw_response"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Resultados guardados: {path}")


def _save_metrics_json(metrics: dict, exp_name: str, total_time: float):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json")
    data = {**metrics, "experiment": exp_name,
            "total_time_seconds": round(total_time, 1)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Métricas guardadas:   {path}")


def _save_submission_txt(results: list, exp_name: str):
    """Genera archivo .txt en formato oficial MSLG-SPA 2026 (subtask SPA2MSLG)."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fname = f"{config.TEAM_NAME}_{config.SOLUTION_NAME}_{config.SUBTASK}.txt"
    path = os.path.join(config.RESULTS_DIR, fname)

    with open(path, "wb") as f:
        for r in results:
            output = r["mslg_pred"].replace("\n", " ").strip()
            if config.SUBMISSION_INCLUDE_ID:
                line = f'"{r["id"]}"\t"{output}"\n'
            else:
                line = f'"{output}"\n'
            f.write(line.encode("utf-8"))
    print(f"  Submission generada:  {path}")


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
    print(f"  RESUMEN — ENFOQUE 7.3 (Ollama: {config.OLLAMA_MODEL})")
    print(f"{'='*60}")
    print(f"  {'Experimento':<32} {'BLEU':>8} {'METEOR':>8} {'chrF':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
    for row in summary:
        print(
            f"  {row['name']:<32} {row['bleu']:>8.4f} "
            f"{row['meteor']:>8.4f} {row['chrf']:>8.4f}"
        )
    print(f"{'='*60}")
