"""Orquestador enfoque7.2 — pipeline reverso MSLG -> SPA.

Clon estructural de enfoque7.1/experiment_runner.py, con los campos
intercambiados (query=MSLG, target=SPA) y soporte para:
  - métrica COMET (subtask MSLG2SPA),
  - generación del archivo de submission .txt en el formato oficial de la
    tarea MSLG-SPA 2026 (TeamName_SolutionName_MSLG2SPA.txt).
"""

import csv
import json
import os
import time

import anthropic_client
import config
import data_loader
import evaluator
import post_processor
import prompt_builder


def _build_prompt(exp_type: str, k: int, mslg: str, emb_index=None):
    if exp_type == "zero_shot_reverse":
        return prompt_builder.build_zero_shot(mslg)
    if exp_type == "few_shot_reverse":
        return prompt_builder.build_few_shot(mslg, k=k)
    if exp_type == "few_shot_rag_reverse":
        retrieved = emb_index.retrieve(mslg, k=k)
        return prompt_builder.build_few_shot_rag(mslg, retrieved)
    if exp_type == "few_shot_rag_curriculum_reverse":
        retrieved = emb_index.retrieve(mslg, k=k)
        return prompt_builder.build_few_shot_rag_curriculum(mslg, retrieved)
    raise ValueError(f"Tipo desconocido: {exp_type}")


def run_experiment(experiment: dict, pool: list, val: list, emb_index=None):
    exp_name = experiment["name"]
    exp_type = experiment["type"]
    k = experiment["k"]

    print(f"\n{'='*60}")
    print(f"  Experimento: {exp_name} (tipo={exp_type}, k={k})")
    print(f"  Modelo: {config.ANTHROPIC_MODEL} | cache={config.ENABLE_PROMPT_CACHE}")
    print(f"  Direccion: MSLG -> SPA")
    print(f"{'='*60}")

    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        mslg = par["mslg"]
        spa_real = par["spa"]

        system_prompt, user_prompt = _build_prompt(
            exp_type, k, mslg, emb_index=emb_index,
        )
        raw_response = anthropic_client.translate(system_prompt, user_prompt)
        spa_pred = post_processor.clean(raw_response)

        results.append({
            "id": par["id"],
            "mslg": mslg,
            "spa_real": spa_real,
            "spa_pred": spa_pred,
            "raw_response": raw_response,
        })

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(val) - i - 1)
        print(
            f"  [{exp_name}] {i+1}/{len(val)} "
            f"({elapsed:.0f}s, ~{remaining:.0f}s restantes) | "
            f"MSLG: {mslg[:45]}... | PRED: {spa_pred[:55]}"
        )

    metrics = evaluator.evaluate(results)
    total_time = time.time() - start_time

    print(f"\n  === Resultados {exp_name} ({total_time:.1f}s) ===")
    print(f"  BLEU:   {metrics['bleu']:.4f}")
    print(f"  METEOR: {metrics['meteor']:.4f}")
    print(f"  chrF:   {metrics['chrf']:.4f}")
    if metrics.get("comet") is not None:
        print(f"  COMET:  {metrics['comet']:.4f}")
    else:
        print(f"  COMET:  N/A (la actividad la calcula sobre el .txt enviado)")

    _save_results_csv(results, exp_name)
    _save_metrics_json(metrics, exp_name, total_time)
    _save_submission_txt(results, exp_name)

    return results, metrics


def run_all(experiments=None):
    experiments = experiments or config.EXPERIMENTS

    pool, val = data_loader.split_dataset()

    need_embeddings = any(
        e["type"] in ("few_shot_rag_reverse", "few_shot_rag_curriculum_reverse")
        for e in experiments
    )
    emb_index = None
    if need_embeddings:
        import embedding_index as emb_mod
        print("\nConstruyendo indice de embeddings sobre MSLG (rag/rag-curriculum)...")
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
    fieldnames = ["id", "mslg", "spa_real", "spa_pred", "raw_response"]
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
    print(f"  Metricas guardadas:   {path}")


def _save_submission_txt(results: list, exp_name: str):
    """Genera archivo .txt en el formato oficial MSLG-SPA 2026 (subtask MSLG2SPA).

    Formato (una línea por instancia, en el mismo orden que el archivo de test):
      "SystemOutput"\\n
    o, opcionalmente (con SUBMISSION_INCLUDE_ID=true):
      "InstanceIdentifier"\\t"SystemOutput"\\n

    Línea final con LF (\\n, formato Linux). Sin headers ni comentarios.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fname = f"{config.TEAM_NAME}_{config.SOLUTION_NAME}_{config.SUBTASK}.txt"
    path = os.path.join(config.RESULTS_DIR, fname)

    # write binary para forzar LF en Windows (sin convertir a CRLF)
    with open(path, "wb") as f:
        for r in results:
            output = r["spa_pred"].replace("\n", " ").strip()
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
        writer = csv.DictWriter(
            f, fieldnames=["name", "bleu", "meteor", "chrf", "comet"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nResumen guardado: {path}")


def _print_summary(summary: list):
    print(f"\n{'='*70}")
    print(f"  RESUMEN — ENFOQUE 7.2 reverso MSLG2SPA ({config.ANTHROPIC_MODEL})")
    print(f"{'='*70}")
    print(f"  {'Experimento':<40} {'BLEU':>7} {'METEOR':>7} {'chrF':>7} {'COMET':>7}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for row in summary:
        comet_str = (f"{row['comet']:>7.4f}" if row.get("comet") is not None
                     else f"{'N/A':>7}")
        print(
            f"  {row['name']:<40} {row['bleu']:>7.4f} "
            f"{row['meteor']:>7.4f} {row['chrf']:>7.4f} {comet_str}"
        )
    print(f"{'='*70}")
