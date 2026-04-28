"""Post-procesamiento para salidas SPA del modelo (sentido reverso).

A diferencia del post-procesamiento de glosa MSLG (que pasa todo a mayúsculas
y elimina puntuación), aquí queremos preservar el español natural:
  - capitalización propia (nombres propios, inicio de oración),
  - signos de puntuación,
  - acentos.

Limpieza:
  1. Tomar la primera línea no vacía (descartar explicaciones extra).
  2. Quitar prefijos comunes ("Español:", "Traducción:", "SPA:", "Spanish:").
  3. Quitar comillas envolventes y markdown (``, **, *).
  4. Normalizar espacios.
  5. Capitalizar la primera letra si está en minúscula.
  6. Asegurar puntuación final (., ?, !).
  7. Para preguntas: si empieza con palabra interrogativa común y no tiene ¿
     al inicio, añadirlo.
"""

import re


_PREFIX_PATTERNS = [
    re.compile(r"^\s*(?:Español|Spanish|Traducción|Translation|SPA|ES)\s*:\s*",
               re.IGNORECASE),
]

_QUOTE_CHARS = ('"', "'", "“", "”", "«", "»", "‘",
                "’")

_INTERROG_STARTERS = (
    "qué ", "quién", "quiénes", "cuál", "cuáles", "cuándo", "cómo",
    "dónde", "adónde", "por qué", "cuánto", "cuánta", "cuántos", "cuántas",
)


def clean(raw_response: str) -> str:
    """Limpia la respuesta cruda para obtener la oración SPA final."""
    if not raw_response:
        return ""

    text = raw_response.strip()

    # 1. Primera línea no vacía
    for line in text.split("\n"):
        line = line.strip()
        if line:
            text = line
            break

    # 2. Quitar prefijos comunes
    for pat in _PREFIX_PATTERNS:
        text = pat.sub("", text)

    # 3. Quitar markdown
    text = text.replace("**", "").replace("`", "")
    # asterisco suelto solo si rodea texto (énfasis)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # 4. Quitar comillas envolventes (una sola capa)
    text = text.strip()
    if len(text) >= 2 and text[0] in _QUOTE_CHARS and text[-1] in _QUOTE_CHARS:
        text = text[1:-1].strip()

    # 5. Normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # 6. Capitalización inicial
    # Si empieza con ¿ o ¡, capitalizar la siguiente letra
    if text[0] in "¿¡" and len(text) > 1 and text[1].islower():
        text = text[0] + text[1].upper() + text[2:]
    elif text[0].islower():
        text = text[0].upper() + text[1:]

    # 7. Puntuación final
    if text[-1] not in ".!?…":
        # heurística: ¿ al inicio → ?, ¡ al inicio → !
        if text.startswith("¿"):
            text += "?"
        elif text.startswith("¡"):
            text += "!"
        else:
            # Si empieza por interrogativa común, asumir pregunta
            lower_start = text.lower()
            if any(lower_start.startswith(s) for s in _INTERROG_STARTERS):
                text = "¿" + text + "?"
                # Re-capitalizar letra tras ¿
                if len(text) > 1 and text[1].islower():
                    text = text[0] + text[1].upper() + text[2:]
            else:
                text += "."

    return text
