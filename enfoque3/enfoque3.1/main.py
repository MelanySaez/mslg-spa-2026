"""Entry point Enfoque 3.1"""
import logging
import sys
import config
import experiment_runner

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY no definida.", file=sys.stderr)
        sys.exit(1)
        
    print("="*60 + "\n  Enfoque 3.1 (Anthropic RAG-10)\n" + "="*60)
    experiment_runner.run_all()

if __name__ == "__main__":
    main()
