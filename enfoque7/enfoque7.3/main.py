"""Entry point — Enfoque 7.3: Híbrido RAG + curriculum sobre Ollama (deepseek-r1:32b).

Misma arquitectura que enfoque7.1 (SPA→MSLG, few-shot-10-rag-curriculum) pero
ejecutándose contra un servidor Ollama local en lugar de la API de Anthropic.
Comparable directamente con 7.1 porque:
  - usa el mismo split (mismo seed, mismo pool/val),
  - usa los mismos prompts (shimeados desde enfoque7.1/prompt_builder.py),
  - usa la misma métrica (BLEU/METEOR/chrF de enfoque7/evaluator.py).
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
            logging.FileHandler("pipeline_e7_3.log", encoding="utf-8"),
        ],
    )

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 7.3 (Ollama local)")
    print(f"  Modelo: {config.OLLAMA_MODEL}")
    print(f"  Endpoint: {config.OLLAMA_URL}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print(f"  RUN_TEST={config.RUN_TEST}  RUN_VAL={config.RUN_VAL}")
    print("  Experimento: few-shot-10-rag-curriculum (SPA→MSLG)")
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
