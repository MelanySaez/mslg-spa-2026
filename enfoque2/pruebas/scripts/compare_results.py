"""Compara métricas entre Ronda 1 y Ronda 2."""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRUEBAS_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PRUEBAS_DIR, "results")


def load_metrics(path):
    with open(path) as f:
        return json.load(f)["metrics"]


def main():
    r1_path = os.path.join(RESULTS_DIR, "round1_test_results.json")
    r2_path = os.path.join(RESULTS_DIR, "round2_test_results.json")

    if not os.path.exists(r1_path) or not os.path.exists(r2_path):
        print("Faltan archivos de resultados. Ejecuta evaluate.py en ambas rondas primero.")
        return

    r1 = load_metrics(r1_path)
    r2 = load_metrics(r2_path)

    print("\n── Comparación R1 vs R2 ──")
    print(f"{'Métrica':<15} {'R1':>10} {'R2':>10} {'Δ':>10}")
    print("─" * 50)
    for k in ["bleu", "rouge1", "rouge2", "rougeL"]:
        v1 = r1.get(k, 0)
        v2 = r2.get(k, 0)
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        print(f"{k:<15} {v1:>10.4f} {v2:>10.4f} {sign + str(round(delta, 4)):>10}")


if __name__ == "__main__":
    main()
