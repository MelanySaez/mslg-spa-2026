"""Re-evalúa BLEU/METEOR/chrF/COMET sobre un CSV de resultados ya generado.

Útil cuando un run terminó pero COMET falló: evita gastar la API otra vez.

Uso:
  uv run python enfoque7/enfoque7.2/recompute_metrics.py [exp_name]

Lee:
  results/<RESULTS_SUBDIR>/<exp_name>_results.csv

Sobrescribe:
  results/<RESULTS_SUBDIR>/<exp_name>_metrics.json
  results/<RESULTS_SUBDIR>/summary.csv (solo la fila del experimento)
  results/<RESULTS_SUBDIR>/<TeamName>_<SolutionName>_<SUBTASK>.txt
"""

import csv
import json
import os
import sys

import config
import evaluator


def _load_results(exp_name: str):
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Cargados {len(rows)} resultados de {path}")
    return rows


def _save_metrics(metrics, exp_name):
    path = os.path.join(config.RESULTS_DIR, f"{exp_name}_metrics.json")
    payload = {**metrics, "experiment": exp_name, "recomputed": True}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Métricas actualizadas: {path}")


def _save_summary(metrics, exp_name):
    path = os.path.join(config.RESULTS_DIR, "summary.csv")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["name"] != exp_name:
                    rows.append(r)
    rows.append({"name": exp_name, **{k: metrics[k]
                                       for k in ("bleu", "meteor", "chrf", "comet")}})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "bleu", "meteor", "chrf", "comet"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary actualizado: {path}")


def _save_submission(results, exp_name):
    fname = f"{config.TEAM_NAME}_{config.SOLUTION_NAME}_{config.SUBTASK}.txt"
    path = os.path.join(config.RESULTS_DIR, fname)
    with open(path, "wb") as f:
        for r in results:
            output = r["spa_pred"].replace("\n", " ").strip()
            if config.SUBMISSION_INCLUDE_ID:
                line = f'"{r["id"]}"\t"{output}"\n'
            else:
                line = f'"{output}"\n'
            f.write(line.encode("utf-8"))
    print(f"Submission regenerada: {path}")


def main():
    exp_name = (sys.argv[1] if len(sys.argv) > 1
                else config.EXPERIMENTS[0]["name"])
    results = _load_results(exp_name)

    metrics = evaluator.evaluate(results)

    print(f"\n=== Métricas {exp_name} ===")
    print(f"  BLEU:   {metrics['bleu']:.4f}")
    print(f"  METEOR: {metrics['meteor']:.4f}")
    print(f"  chrF:   {metrics['chrf']:.4f}")
    if metrics.get("comet") is not None:
        print(f"  COMET:  {metrics['comet']:.4f}")
    else:
        print(f"  COMET:  N/A")

    _save_metrics(metrics, exp_name)
    _save_summary(metrics, exp_name)
    _save_submission(results, exp_name)


if __name__ == "__main__":
    main()
