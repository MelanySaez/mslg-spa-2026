"""Entry point: ejecuta todos los experimentos SPA→MSLG."""

import logging
import sys

import experiment_runner


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", encoding="utf-8"),
        ],
    )

    print("=" * 60)
    print("  Pipeline SPA → MSLG (IberLEF 2026)")
    print("  Estudio Zero-Shot → Few-Shot → RAG")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
