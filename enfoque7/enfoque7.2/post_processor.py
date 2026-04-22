"""Post-processor v2 — reglas determinísticas sobre la salida cruda del LLM.

Mejoras sobre enfoque3/post_processor.py:
  1. Preserva el prefijo dm- en minúscula (el .upper() del v1 lo destruía).
  2. Normaliza POR-QUÉ (con guión) → PORQUÉ (junto), alineado al corpus.
  3. Elimina tokens residuales de cópula (SER, ESTAR, SOY, ERES, ES, ESTÁ, FUE,
     ERA) que no deben aparecer en glosas MSLG.
  4. Conserva el resto del comportamiento v1: extracción de primera línea,
     eliminación de prefijos, comillas y artículos sueltos.
"""

import re


_ARTICLES = {"EL", "LA", "LOS", "LAS", "UN", "UNA", "UNOS", "UNAS"}

# Tokens de cópula que residualmente puede emitir el modelo y que el corpus no usa.
# Se eliminan como tokens sueltos (no dentro de compuestos o locuciones con guión).
_COPULA_TOKENS = {
    "SER", "ESTAR", "SOY", "ERES", "ES", "ESTÁ", "ESTA", "SON",
    "FUE", "ERA", "ESTUVE", "ESTUVO", "FUI", "FUISTE",
}

# Reemplazos de normalización ortográfica (alineados al corpus).
_REPLACEMENTS = [
    (re.compile(r"\bPOR-QUÉ\b"),  "PORQUÉ"),
    (re.compile(r"\bPOR QUE\b"),  "PORQUE"),
]

# Prefijos de respuesta a recortar.
_PREFIX_PATTERNS = [
    re.compile(r"^MSLG\s*:\s*", re.IGNORECASE),
    re.compile(r"^Traducción\s*:\s*", re.IGNORECASE),
    re.compile(r"^Respuesta\s*:\s*", re.IGNORECASE),
    re.compile(r"^Glosa\s*:\s*", re.IGNORECASE),
    re.compile(r"^LSM\s*:\s*", re.IGNORECASE),
]


def _smart_upper(text: str) -> str:
    """Uppercase inteligente que preserva el prefijo 'dm-' en minúsculas."""
    tokens = text.split()
    out = []
    for tok in tokens:
        low = tok.lower()
        if low.startswith("dm-"):
            # Conserva 'dm-' y pasa a mayúsculas lo que sigue.
            out.append("dm-" + tok[3:].upper())
        else:
            out.append(tok.upper())
    return " ".join(out)


def clean(raw_response: str) -> str:
    text = raw_response.strip()

    # 1. Primera línea no vacía.
    for line in text.split("\n"):
        line = line.strip()
        if line:
            text = line
            break

    # 2. Eliminar prefijos de respuesta.
    for pat in _PREFIX_PATTERNS:
        text = pat.sub("", text)

    # 3. Eliminar comillas.
    text = (
        text.replace('"', "")
            .replace("'", "")
            .replace("“", "")
            .replace("”", "")
    )

    # 4. Uppercase preservando 'dm-'.
    text = _smart_upper(text)

    # 5. Eliminar puntuación excepto ¿ ? ¡ ! - + (y el guión usado en compuestos).
    text = re.sub(r"[^\w\s¿?¡!\-+]", "", text)

    # 6. Normalizaciones ortográficas alineadas al corpus.
    for pat, repl in _REPLACEMENTS:
        text = pat.sub(repl, text)

    # 7. Eliminar tokens sueltos de cópula y artículos (no tocar compuestos).
    cleaned_tokens = []
    for token in text.split():
        # Compuestos con guión o +MUJER → se conservan íntegros.
        if "-" in token or "+" in token:
            cleaned_tokens.append(token)
            continue
        if token in _ARTICLES:
            continue
        if token in _COPULA_TOKENS:
            continue
        cleaned_tokens.append(token)

    text = " ".join(cleaned_tokens)

    # 8. Normalizar espacios.
    text = re.sub(r"\s+", " ", text).strip()

    return text
