"""Orquestador enfoque 3.1"""
import csv
import json
import os
import time

import anthropic_client
import config
import data_loader
import embedding_index as emb_mod
import evaluator
import post_processor
import prompt_builder

def run_experiment(experiment, pool, val, emb_index=None):
    exp_name = experiment["name"]
    exp_type = experiment["type"]
    k = experiment["k"]

    print(f"\n{'='*60}\n  Experimento: {exp_name} (tipo={exp_type}, k={k})\n{'='*60}")
    results = []
    start_time = time.time()

    for i, par in enumerate(val):
        spa, mslg_real = par["spa"], par["mslg"]
        if exp_type == "rag":
            ejemplos = emb_index.retrieve(spa, k=k)
            system_prompt, user_prompt = prompt_builder.build_rag(spa, ejemplos)
        else:
            raise ValueError("Solo soportado RAG en este runner.")
            
        raw_response = anthropic_client.translate(system_prompt, user_prompt)
        mslg_pred = post_processor.clean(raw_response)

        results.append({
            "id": par["id"], "spa": spa, "mslg_real": mslg_real,
            "mslg_pred": mslg_pred, "raw_response": raw_response
        })
        
        elapsed = time.time() - start_time
        print(f"  [{exp_name}] {i+1}/{len(val)} ({elapsed:.0f}s) | SPA: {spa[:40]}... | PRED: {mslg_pred[:50]}")

    metrics = evaluator.evaluate(results)
    total_time = time.time() - start_time

    print(f"\n  === Resultados {exp_name} ({total_time:.1f}s) ===")
    for m in ["bleu", "meteor", "chrf"]: print(f"  {m.upper()}: {metrics[m]:.4f}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv"), "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["id","spa","mslg_real","mslg_pred","raw_response"]).writeheader()
        csv.DictWriter(f, fieldnames=["id","spa","mslg_real","mslg_pred","raw_response"]).writerows(results)
    
    with open(os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**metrics, "experiment": exp_name}, f, indent=2, ensure_ascii=False)

    return results, metrics

def run_all(experiments=None):
    experiments = experiments or config.EXPERIMENTS
    pool, val = data_loader.split_dataset()
    emb_index = emb_mod.EmbeddingIndex(pool)
    summary = []
    
    for experiment in experiments:
        _, metrics = run_experiment(experiment, pool, val, emb_index)
        summary.append({"name": experiment["name"], **metrics})
    
    with open(os.path.join(config.RESULTS_DIR, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"]).writeheader()
        csv.DictWriter(f, fieldnames=["name", "bleu", "meteor", "chrf"]).writerows(summary)
    return summary
