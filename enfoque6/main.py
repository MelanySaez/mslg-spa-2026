"""Entry point — Enfoque 6: Reglas LSM + Híbrido (deepseek-r1:70b)."""

import logging
import sys

import experiment_runner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline_e6.log", encoding="utf-8"),
        ],
    )

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 6")
    print("  Reglas LSM (reglas_sintaxis_lsm.md) + FOL + deepseek-r1:70b")
    print("  Experimentos: zero-shot · few-shot · híbrido · RAG-híbrido")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
