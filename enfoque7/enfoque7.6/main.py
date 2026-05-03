"""Entry point — Enfoque 7.6: pipeline reverso MSLG → SPA sobre NVIDIA NIM.

Misma arquitectura que enfoque7.2 (MSLG→SPA, few-shot-10-rag-curriculum) pero
ejecutándose contra el endpoint hospedado de NVIDIA Build con el modelo
`nvidia/riva-translate-4b-instruct-v1_1`.

Comparable directamente con 7.2 / 7.4 (mismo split, mismos prompts, misma
métrica incluyendo COMET vía CLI `comet-score`).
"""

import logging
import sys

import config
import experiment_runner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline_e7_6.log", encoding="utf-8"),
        ],
    )

    print("=" * 60)
    print("  Pipeline MSLG → SPA — Enfoque 7.6 (reverso, NVIDIA NIM)")
    print(f"  Modelo: {config.NVIDIA_MODEL}")
    print(f"  Endpoint: {config.NVIDIA_URL}")
    print(f"  Dataset: {config.DATASET_PATH}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print("  Experimento: few-shot-10-rag-curriculum-mslg2spa")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
