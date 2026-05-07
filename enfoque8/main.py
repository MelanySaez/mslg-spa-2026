"""Entry point — Enfoque 8: Many-Shot In-Context Learning."""

import logging
import sys
import config
import experiment_runner

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY no definida.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Pipeline SPA → MSLG — Enfoque 8 (Many-Shot)")
    print(f"  Modelo: {config.ANTHROPIC_MODEL}")
    print(f"  Prompt caching: {config.ENABLE_PROMPT_CACHE}")
    print(f"  RUN_VAL={config.RUN_VAL} | RUN_TEST={config.RUN_TEST}")
    print("=" * 60)

    if not (config.RUN_TEST or config.RUN_VAL):
        print("ERROR: RUN_TEST y RUN_VAL = false. Nada que correr.", file=sys.stderr)
        sys.exit(1)

    experiment_runner.run_all()

if __name__ == "__main__":
    main()
