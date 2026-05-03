"""Entry point — Enfoque 7.5: Híbrido RAG + curriculum sobre NVIDIA NIM.

Misma arquitectura que enfoque7.1 (SPA→MSLG, few-shot-10-rag-curriculum) pero
ejecutándose contra el endpoint hospedado de NVIDIA Build con el modelo
`nvidia/riva-translate-4b-instruct-v1_1` (4B, instruction-tuned para
traducción multilingüe).

Comparable directamente con 7.1 / 7.3:
  - mismo split (mismo seed, mismo pool/val),
  - mismos prompts (shimeados desde enfoque7.1/prompt_builder.py),
  - misma métrica (BLEU/METEOR/chrF de enfoque7/evaluator.py).
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
            logging.FileHandler("pipeline_e7_5.log", encoding="utf-8"),
        ],
    )

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 7.5 (NVIDIA NIM)")
    print(f"  Modelo: {config.NVIDIA_MODEL}")
    print(f"  Endpoint: {config.NVIDIA_URL}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print("  Experimento: few-shot-10-rag-curriculum (SPA→MSLG)")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
