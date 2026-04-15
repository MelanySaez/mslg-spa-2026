"""Entry point de enfoque4.

Uso desde la raíz del proyecto:
    uv run python -m enfoque4.main

Prerequisitos:
  - Ollama corriendo local: `ollama serve`
  - Modelo disponible:      `ollama pull qwen2.5:14b`
  - Dependencias:           `uv add spacy sentence-transformers scikit-learn nltk requests numpy`
  - Modelo spaCy:           `uv run python -m spacy download es_core_news_lg`
"""

from .experiment_runner import run_all


def main():
    run_all()


if __name__ == "__main__":
    main()
