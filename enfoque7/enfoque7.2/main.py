"""Entry point — Enfoque 7.2: Plan de escalado en 3 pasos.

Paso 1 — sweep de k (8, 10, 12, 15) sobre few_shot_rag_curriculum.
Paso 2 — post_processor v2 con reglas determinísticas (aplicado a todos).
Paso 3 — Self-Consistency N=3 sobre la variante k=10 (ajustable tras ver paso 1).
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
            "Crea .env en enfoque7/ o en la raíz del proyecto con:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 7.2")
    print(f"  Modelo: {config.ANTHROPIC_MODEL}")
    print(f"  Prompt caching: {config.ENABLE_PROMPT_CACHE}")
    print(f"  Post-processor: v2 (preserva dm-, normaliza PORQUÉ, elimina cópula)")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print(f"  Experimentos: {len(config.EXPERIMENTS)} (sweep k + SC)")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
