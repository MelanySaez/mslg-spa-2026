"""Enfoque4: FOL-enhanced RAG.

Al importar el paquete se agrega enfoque3/ al sys.path para poder reusar
sus módulos (data_loader, embedding_index, ollama_client, post_processor,
evaluator, prompt_builder PROMPT_BASE) bajo sus nombres originales.

Los módulos locales de enfoque4 se importan como `enfoque4.xxx` (paquete),
por lo que NO colisionan con los nombres de módulos de enfoque3.
"""

import os
import sys

_E3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enfoque3"))
if _E3_DIR not in sys.path:
    sys.path.insert(0, _E3_DIR)
