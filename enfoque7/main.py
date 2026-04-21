"""Entry point — Enfoque 7: Zero-shot & Few-shot vía Anthropic API (Claude Haiku 4.5)."""

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
            logging.FileHandler("pipeline_e7.log", encoding="utf-8"),
        ],
    )

    if not config.ANTHROPIC_API_KEY:
        print(
            "ERROR: ANTHROPIC_API_KEY no definida.\n"
            "Crea un archivo .env en enfoque7/ (o en la raíz del proyecto) con:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "Ver enfoque7/.env.example para plantilla.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 7")
    print(f"  Modelo: {config.ANTHROPIC_MODEL}")
    print(f"  Prompt caching: {config.ENABLE_PROMPT_CACHE}")
    print("  Experimentos: zero-shot · few-shot (variantes)")
    print("=" * 60)

    experiment_runner.run_all()


if __name__ == "__main__":
    main()
