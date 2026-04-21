"""Construcción de prompts para Enfoque 7 (Anthropic API).

Reutiliza las constantes (PROMPT_BASE, FIXED_EXAMPLES, LSM_GLOSSARY,
NEGATIVE_EXAMPLES, COT_INSTRUCTIONS) del prompt_builder de enfoque6.

A diferencia de enfoque6 (prompt monolítico para Ollama), aquí cada builder
retorna una tupla (system, user):
  - system: contenido estático, cacheable por la API de Anthropic.
  - user:   la oración variable a traducir.
"""

import importlib.util
import os

# ── Importar constantes del prompt_builder de enfoque6 ────────────────────────
_e6_path = os.path.join(os.path.dirname(__file__), "..", "enfoque6", "prompt_builder.py")
_spec = importlib.util.spec_from_file_location("enfoque6_prompt_builder", _e6_path)
_e6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e6)

PROMPT_BASE = _e6.PROMPT_BASE
FIXED_EXAMPLES = _e6.FIXED_EXAMPLES
LSM_GLOSSARY = _e6.LSM_GLOSSARY
NEGATIVE_EXAMPLES = _e6.NEGATIVE_EXAMPLES
COT_INSTRUCTIONS = _e6.COT_INSTRUCTIONS
_format_negative_examples = _e6._format_negative_examples
_format_examples = _e6._format_examples


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user(sentence: str) -> str:
    """Mensaje de usuario variable: solo la oración a traducir.

    Formato minimalista para que el modelo emita la glosa sin preámbulo.
    """
    return f'Traduce:\nSPA: "{sentence}"\nMSLG:'


def _compose_system(*blocks: str) -> str:
    return "\n\n".join(b for b in blocks if b)


# ── Zero-shot ─────────────────────────────────────────────────────────────────

def build_zero_shot(sentence: str):
    system = PROMPT_BASE
    return system, _user(sentence)


def build_zero_shot_cot(sentence: str):
    system = _compose_system(PROMPT_BASE, COT_INSTRUCTIONS)
    return system, _user(sentence)


def build_zero_shot_glossary(sentence: str):
    system = _compose_system(PROMPT_BASE, LSM_GLOSSARY)
    return system, _user(sentence)


def build_zero_shot_full(sentence: str):
    system = _compose_system(
        PROMPT_BASE,
        LSM_GLOSSARY,
        _format_negative_examples(),
        COT_INSTRUCTIONS,
    )
    return system, _user(sentence)


# ── Few-shot ──────────────────────────────────────────────────────────────────

def _examples_block(examples, header="EJEMPLOS:"):
    return f"{header}\n{_format_examples(examples)}"


def build_few_shot(sentence: str, k: int = 5):
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(FIXED_EXAMPLES[:k]),
    )
    return system, _user(sentence)


def build_few_shot_cot(sentence: str, k: int = 10):
    system = _compose_system(
        PROMPT_BASE,
        COT_INSTRUCTIONS,
        _examples_block(FIXED_EXAMPLES[:k]),
    )
    return system, _user(sentence)


def build_few_shot_negative(sentence: str, k: int = 10):
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(FIXED_EXAMPLES[:k], header="EJEMPLOS CORRECTOS:"),
        _format_negative_examples(),
    )
    return system, _user(sentence)


def build_few_shot_curriculum(sentence: str, k: int = 10):
    ordered = sorted(FIXED_EXAMPLES[:k], key=lambda e: len(e["spa"]))
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(ordered, header="EJEMPLOS (de simple a complejo):"),
    )
    return system, _user(sentence)


def build_few_shot_diverse(sentence: str, diverse_examples: list):
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(diverse_examples,
                        header="EJEMPLOS (cobertura diversa de patrones LSM):"),
    )
    return system, _user(sentence)


def build_few_shot_full(sentence: str, k: int = 10):
    system = _compose_system(
        PROMPT_BASE,
        LSM_GLOSSARY,
        COT_INSTRUCTIONS,
        _examples_block(FIXED_EXAMPLES[:k], header="EJEMPLOS CORRECTOS:"),
        _format_negative_examples(),
    )
    return system, _user(sentence)
