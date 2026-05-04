"""Entry point — Enfoque 7.4: pipeline reverso MSLG → SPA sobre Ollama (deepseek-r1:32b).

Misma arquitectura que enfoque7.2 (MSLG→SPA, few-shot-10-rag-curriculum) pero
ejecutándose contra un servidor Ollama local en lugar de la API de Anthropic.
Comparable directamente con 7.2 (mismo split, mismos prompts, misma métrica
incluyendo COMET vía CLI `comet-score`).
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
            logging.FileHandler("pipeline_e7_4.log", encoding="utf-8"),
        ],
    )

    print("=" * 60)
    print("  Pipeline MSLG → SPA — Enfoque 7.4 (reverso, Ollama local)")
    print(f"  Modelo: {config.OLLAMA_MODEL}")
    print(f"  Endpoint: {config.OLLAMA_URL}")
    print(f"  Dataset: {config.DATASET_PATH}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print(f"  RUN_TEST={config.RUN_TEST}  RUN_VAL={config.RUN_VAL}")
    print("  Experimento: few-shot-10-rag-curriculum-mslg2spa")
    print("=" * 60)

    if not (config.RUN_TEST or config.RUN_VAL):
        print("ERROR: ambos RUN_TEST y RUN_VAL están en false. Nada que correr.",
              file=sys.stderr)
        sys.exit(1)

    if config.RUN_VAL:
        experiment_runner.run_all()

    if config.RUN_TEST:
        experiment_runner.run_submission()


if __name__ == "__main__":
    main()
