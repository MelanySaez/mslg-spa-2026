"""Orquestador enfoque8 — Many-Shot sobre validación y submission oficial."""

import csv
import json
import os
import time

import config
import prompt_builder

# Importamos shims de enfoque7
import importlib.util
_e7_dir = os.path.join(os.path.dirname(__file__), "..", "enfoque7")
def _load_shim(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_e7_dir, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

anthropic_client = _load_shim("anthropic_client")
data_loader = _load_shim("data_loader")
evaluator = _load_shim("evaluator")
post_processor = _load_shim("post_processor")

def run_experiment(experiment: dict, pool: list, val: list):
    exp_name = experiment["name"]
    print(f"\n{'='*60}\n  Experimento: {exp_name} (many-shot con {len(pool)} ejemplos)\n{'='*60}")
    
    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        spa = par["spa"]
        mslg_real = par["mslg"]

        system_prompt, user_prompt = prompt_builder.build_many_shot(spa, pool)
        raw_response = anthropic_client.translate(system_prompt, user_prompt)
        mslg_pred = post_processor.clean(raw_response)

        results.append({
            "id": par["id"], "spa": spa, "mslg_real": mslg_real,
            "mslg_pred": mslg_pred, "raw_response": raw_response,
        })

        elapsed = time.time() - start_time
        print(f"  [{exp_name}] {i+1}/{len(val)} ({elapsed:.0f}s) | SPA: {spa[:45]}... | PRED: {mslg_pred[:55]}")

    metrics = evaluator.evaluate(results)
    total_time = time.time() - start_time

    print(f"\n  === Resultados {exp_name} ({total_time:.1f}s) ===")
    for m in ["bleu", "meteor", "chrf"]:
        print(f"  {m.upper()}: {metrics[m]:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv"), "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["id","spa","mslg_real","mslg_pred","raw_response"]).writeheader()
        csv.DictWriter(f, fieldnames=["id","spa","mslg_real","mslg_pred","raw_response"]).writerows(results)

    with open(os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**metrics, "experiment": exp_name}, f, indent=2, ensure_ascii=False)

    return results, metrics

def run_submission():
    """Ejecuta submission final con TODOS los ejemplos mezclados (train+val)."""
    full_data = data_loader.load_dataset()
    
    # Cargar test set (SPA2MSLG_test.txt)
    test_data = []
    with open(config.TEST_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            test_data.append({"id": row["ID"], "spa": row["SPA"].strip()})
    
    print(f"\n{'='*60}\n  SUBMISSION: Mega-Shot ({len(full_data)} ejemplos)\n{'='*60}")
    start = time.time()
    predictions = []
    
    for i, item in enumerate(test_data):
        spa = item["spa"]
        system_prompt, user_prompt = prompt_builder.build_many_shot(spa, full_data)
        raw = anthropic_client.translate(system_prompt, user_prompt)
        pred = post_processor.clean(raw)
        predictions.append(pred)

        elapsed = time.time() - start
        print(f"  [TEST] {i+1}/{len(test_data)} ({elapsed:.0f}s) | SPA: {spa[:40]}... → {pred[:50]}")

    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    filepath = os.path.join(config.SUBMISSIONS_DIR, "UTB_MANYSHOT_SPA2MSLG.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        for pred in predictions: f.write(f'"{pred}"\n')

    print(f"\nSubmission oficial guardado en: {filepath}")

def run_all():
    experiments = config.EXPERIMENTS
    pool, val = data_loader.split_dataset()
    summary = []
    
    if config.RUN_VAL:
        for experiment in experiments:
            _, metrics = run_experiment(experiment, pool, val)
            summary.append({"name": experiment["name"], **metrics})
            
        with open(os.path.join(config.RESULTS_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"]).writeheader()
            csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"]).writerows(summary)
            
    if config.RUN_TEST:
        run_submission()
