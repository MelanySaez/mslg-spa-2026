"""Orquestador enfoque8.1 — Many-Shot MSLG → SPA."""

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


def run_experiment(experiment: dict, pool: list, val: list):
    exp_name = experiment["name"]
    print(f"\n{'='*60}\n  Experimento: {exp_name} (many-shot con {len(pool)} ejemplos)\n{'='*60}")

    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        mslg = par["mslg"]
        spa_real = par["spa"]

        system_prompt, user_prompt = prompt_builder.build_many_shot(mslg, pool)
        raw_response = anthropic_client.translate(system_prompt, user_prompt)
        spa_pred = post_processor.clean(raw_response)

        results.append({
            "id": par["id"], "mslg": mslg, "spa_real": spa_real,
            "spa_pred": spa_pred, "raw_response": raw_response,
        })

        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        remaining = avg * (len(val) - i - 1)
        print(
            f"  [{exp_name}] {i+1}/{len(val)} "
            f"({elapsed:.0f}s, ~{remaining:.0f}s restantes) | "
            f"MSLG: {mslg[:45]}... | PRED: {spa_pred[:55]}"
        )

    metrics = evaluator.evaluate(results)
    total_time = time.time() - start_time

    print(f"\n  === Resultados {exp_name} ({total_time:.1f}s) ===")
    for m in ["bleu", "meteor", "chrf"]:
        print(f"  {m.upper()}: {metrics[m]:.4f}")
    if metrics.get("comet") is not None:
        print(f"  COMET:  {metrics['comet']:.4f}")
    else:
        print(f"  COMET:  N/A (la actividad la calcula sobre el .txt enviado)")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    _save_results_csv(results, exp_name)
    _save_metrics_json(metrics, exp_name, total_time)

    return results, metrics


def run_submission():
    """Submission final con TODOS los pares del corpus (train+val) como pool."""
    full_data = data_loader.load_dataset()
    test_items = data_loader.load_test(config.TEST_PATH, config.TEST_SOURCE_COL)

    print(f"\n{'='*60}\n  SUBMISSION: Mega-Shot ({len(full_data)} ejemplos)\n{'='*60}")
    start = time.time()
    predictions = []

    for i, item in enumerate(test_items):
        mslg = item["source"]
        system_prompt, user_prompt = prompt_builder.build_many_shot(mslg, full_data)
        raw = anthropic_client.translate(system_prompt, user_prompt)
        spa_pred = post_processor.clean(raw)
        predictions.append({
            "id": item["id"], "mslg": mslg,
            "spa_pred": spa_pred, "raw_response": raw,
        })

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        remaining = avg * (len(test_items) - i - 1)
        print(
            f"  [TEST] {i+1}/{len(test_items)} "
            f"({elapsed:.0f}s, ~{remaining:.0f}s restantes) | "
            f"MSLG: {mslg[:40]}... → {spa_pred[:50]}"
        )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    _save_test_results_csv(predictions, "many-shot-all-mslg2spa")
    _save_submission_txt(predictions, "many-shot-all-mslg2spa")

    print(f"\n{'='*60}")
    print(f"  Submission lista — adjuntar al email a ansel@cicese.edu.mx")
    print(f"  Subject: 'MSLG-SPA 2026 Submission – {config.TEAM_NAME}'")
    print(f"{'='*60}")


def run_all():
    pool, val = data_loader.split_dataset()
    summary = []

    if config.RUN_VAL:
        for experiment in config.EXPERIMENTS:
            _, metrics = run_experiment(experiment, pool, val)
            summary.append({"name": experiment["name"], **metrics})

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "bleu", "meteor", "chrf", "comet"])
            writer.writeheader()
            writer.writerows(summary)

    if config.RUN_TEST:
        run_submission()


def _save_results_csv(results, exp_name):
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "mslg", "spa_real", "spa_pred", "raw_response"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Resultados guardados: {path}")


def _save_metrics_json(metrics, exp_name, total_time):
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {**metrics, "experiment": exp_name, "total_time_seconds": round(total_time, 1)},
            f, indent=2, ensure_ascii=False)
    print(f"  Métricas guardadas:   {path}")


def _save_test_results_csv(predictions, exp_name):
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_test_predictions.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "mslg", "spa_pred", "raw_response"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  Predicciones test:    {path}")


def _save_submission_txt(predictions, exp_name):
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    fname = f"{config.TEAM_NAME}_{config.SOLUTION_NAME}_{config.SUBTASK}.txt"
    path = os.path.join(config.SUBMISSIONS_DIR, fname)
    with open(path, "wb") as f:
        for r in predictions:
            output = r["spa_pred"].replace("\n", " ").strip()
            if config.SUBMISSION_INCLUDE_ID:
                line = f'"{r["id"]}"\t"{output}"\n'
            else:
                line = f'"{output}"\n'
            f.write(line.encode("utf-8"))
    print(f"  Submission generada:  {path}")
