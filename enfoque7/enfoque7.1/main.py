"""Entry point — Enfoque 7.1: Híbrido RAG + curriculum (Claude Haiku 4.5)."""

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
            logging.FileHandler("pipeline_e7_1.log", encoding="utf-8"),
        ],
    )

    if not config.ANTHROPIC_API_KEY:
        print(
            "ERROR: ANTHROPIC_API_KEY no definida.\n"
            "Crea .env en enfoque7/ (heredado) o en la raíz del proyecto con:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 7.1")
    print(f"  Modelo: {config.ANTHROPIC_MODEL}")
    print(f"  Prompt caching: {config.ENABLE_PROMPT_CACHE}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print(f"  RUN_TEST={config.RUN_TEST}  RUN_VAL={config.RUN_VAL}")
    print("  Experimentos: few-shot-10-rag-curriculum (híbrido)")
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
