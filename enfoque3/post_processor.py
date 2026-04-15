"""Post-procesamiento de las salidas del LLM."""

import re


# Artículos que deben eliminarse (solo como tokens sueltos, no dentro de compuestos)
_ARTICLES = {"EL", "LA", "LOS", "LAS", "UN", "UNA", "UNOS", "UNAS"}


def clean(raw_response):
    """
    Limpia la respuesta cruda del LLM para obtener una glosa válida.

    1. Extrae la primera línea significativa.
    2. Elimina prefijos como "MSLG:", "Traducción:", etc.
    3. Elimina comillas.
    4. Convierte a mayúsculas.
    5. Elimina puntuación excepto ¿ ? ¡ ! - +
    6. Elimina artículos residuales sueltos.
    7. Normaliza espacios.
    """
    text = raw_response.strip()

    # Tomar la primera línea no vacía
    for line in text.split("\n"):
        line = line.strip()
        if line:
            text = line
            break

    # Eliminar prefijos comunes
    prefixes = [
        r"^MSLG\s*:\s*",
        r"^Traducción\s*:\s*",
        r"^Respuesta\s*:\s*",
        r"^Glosa\s*:\s*",
        r"^LSM\s*:\s*",
    ]
    for pat in prefixes:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Eliminar comillas
    text = text.replace('"', "").replace("'", "").replace("\u201c", "").replace("\u201d", "")

    # Convertir a mayúsculas
    text = text.upper()

    # Eliminar puntuación excepto ¿ ? ¡ ! - + y el prefijo dm-
    # Primero proteger dm- y tokens compuestos
    text = re.sub(r"[^\w\s¿?¡!\-+]", "", text)

    # Eliminar artículos sueltos (no dentro de tokens con guión)
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        # Si el token contiene guión, es un compuesto — no filtrar
        if "-" in token or "+" in token:
            cleaned_tokens.append(token)
        elif token in _ARTICLES:
            continue  # eliminar artículo suelto
        else:
            cleaned_tokens.append(token)

    text = " ".join(cleaned_tokens)

    # Normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()

    return text
