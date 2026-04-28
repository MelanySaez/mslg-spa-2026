"""Entry point — Enfoque 7.2: pipeline reverso MSLG -> SPA (Claude Haiku 4.5).

Replica el mejor experimento de 7.1 (few-shot-10-rag-curriculum) en sentido
inverso: dada una glosa MSLG, generar la oracion en espanol natural.
Usa el mismo split (mismo seed, mismo pool, mismo val) para que las metricas
sean directamente comparables a la tarea SPA -> MSLG de 7.1.
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
            logging.FileHandler("pipeline_e7_2.log", encoding="utf-8"),
        ],
    )

    if not config.ANTHROPIC_API_KEY:
        print(
            "ERROR: ANTHROPIC_API_KEY no definida.\n"
            "Crea .env en enfoque7/ (heredado) o en la raiz del proyecto con:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("  Pipeline MSLG -> SPA — Enfoque 7.2 (reverso)")
    print(f"  Modelo: {config.ANTHROPIC_MODEL}")
    print(f"  Prompt caching: {config.ENABLE_PROMPT_CACHE}")
    print(f"  Dataset: {config.DATASET_PATH}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print("  Experimento: few-shot-10-rag-curriculum-mslg2spa")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
