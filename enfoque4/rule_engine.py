"""Motor de reglas FOL (port de enfoque1/codigo.py).

Genera un candidato de glosa MSLG a partir de una oración en español usando
reglas lingüísticas deterministas sobre spaCy. El resultado se inyecta en
el prompt RAG como hint auxiliar para el LLM.

Sin fallback por similitud: la recuperación semántica del RAG cubre ese rol.
"""

import spacy


MASCULINOS_IRREGULARES = {
    "madre": "padre", "mamá": "papá", "mujer": "hombre",
    "reina": "rey", "actriz": "actor", "emperatriz": "emperador",
    "yerna": "yerno", "nuera": "nuero",
}

NO_GENERO = {
    "sopa", "moneda", "acta", "mesa", "casa", "agua", "hora", "forma",
    "cama", "cara", "mano", "foto", "nota", "lista", "carta", "bolsa",
    "crema", "planta", "trampa", "copa", "firma", "cuenta", "vista",
    "cinta", "llama", "palma", "zona", "manga", "carga", "plaza",
    "pasta", "taza", "falta", "marca", "masa", "pista", "selva",
    "playa", "orilla", "rama", "rueda", "ruta", "sangre", "sombra",
    "tienda", "tierra", "vuelta", "palabra", "persona",
    "cosa", "parte", "vez", "vida", "noche", "tarde", "mañana",
    "semana", "pregunta", "respuesta", "clase", "lengua", "señal",
    "credencial", "delegación", "antena", "corona",
    "huella", "escuela", "iglesia", "cocina", "medicina", "historia",
    "música", "física", "química", "biología", "economía",
    "tecnología", "matemáticas", "literatura", "cultura",
    "mezcla", "regla", "tabla", "silla", "ventana", "puerta", "pared",
    "calle", "ciudad", "aldea", "villa", "región", "nación",
}

LUGARES_CONOCIDOS = {
    "villahermosa", "guadalajara", "monterrey", "puebla", "oaxaca",
    "tijuana", "mérida", "cancún", "veracruz", "acapulco", "toluca",
    "mexico", "méxico", "tepito", "tlatelolco", "iztapalapa",
    "san francisco", "new york", "nueva york", "estados unidos",
    "california", "texas", "florida", "chicago",
}

MARCADORES_TEMPORALES = {
    "ayer", "mañana", "hoy", "siempre", "nunca", "después",
    "antes", "ahora", "tarde", "temprano", "pronto",
}

STOPWORDS_MSLG = {"el", "la", "los", "las", "un", "una", "unos", "unas",
                  "de", "del", "al", "a", "en", "con", "por", "para",
                  "que", "se", "lo", "le", "les",
                  "su", "sus"}

INTENSIFICADORES = {"muy", "mucho", "muchos", "muchas", "demasiado",
                    "bastante", "súper", "super", "tan"}

COPULAS = {"ser", "estar", "parecer"}

SIGLAS_ORG = {"ine", "imss", "issste", "sep", "unam", "ipn", "cfe", "pemex",
              "sat", "inegi", "conacyt", "ssa", "cdmx", "shcp", "onu", "oms"}

NOMBRES_PERSONAS = {
    "santiago", "isabel", "diego", "rivera", "cristóbal", "colón",
    "guadalupe", "victoria", "trinidad", "virginia", "loreto",
    "juan", "maria", "maría", "pedro", "josé", "ana", "carlos",
    "luis", "jorge", "rosa", "elena", "miguel", "alejandro",
}


def cargar_nlp(model_name="es_core_news_lg"):
    """Carga modelo spaCy. Descarga explícita si falta para dar error claro."""
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"Modelo spaCy '{model_name}' no instalado. "
            f"Instálalo con: python -m spacy download {model_name}"
        ) from e


def construir_dicc_compuestos(pool):
    """Extrae tokens con guión de las glosas MSLG del pool y los mapea
    a su frase equivalente en español (solo si todas las partes aparecen).
    """
    dicc = {}
    for par in pool:
        tokens_mslg = par["mslg"].split()
        for token in tokens_mslg:
            if "-" in token and not token.startswith("dm-"):
                partes = token.replace("-", " ").lower()
                spa = par["spa"].lower()
                if all(p in spa for p in partes.split()):
                    dicc[partes] = token
    return dicc


def construir_dicc_nombres(pool):
    """Extrae nombres propios marcados con dm- en las glosas del pool."""
    nombres = set()
    for par in pool:
        tokens_mslg = par["mslg"].split()
        spa_lower = par["spa"].lower()
        for token in tokens_mslg:
            if token.startswith("dm-"):
                nombre_mslg = token[3:].lower()
                if nombre_mslg in spa_lower:
                    nombres.add(nombre_mslg)
    return nombres


def generar_gloss_fol(oracion_spa, nlp, dicc_compuestos, nombres_personas):
    """Aplica el pipeline FOL: genera una glosa MSLG candidata.

    Reglas aplicadas en orden:
      1. Compuestos multipalabra del diccionario (consumen tokens).
      2. Artículos y preposiciones → eliminar.
      3. Intensificadores → flag MUCHO al final.
      4. Marcadores temporales → anteponer al resultado.
      5. Nombres propios PROPN/PER → dm-NOMBRE (excepto siglas y lugares).
      6. Cópulas (ser/estar/parecer) → eliminar.
      7. HACER impersonal en expresiones de clima → eliminar.
      8. Verbos → lemma en MAYÚSCULAS (infinitivo).
      9. Femenino NOUN → MASCULINO+MUJER (excepto NO_GENERO).
     10. Default → lemma en MAYÚSCULAS.
    """
    doc = nlp(oracion_spa)
    tokens_salida = []
    marcadores_inicio = []
    hay_intensificador = False

    texto_lower = oracion_spa.lower()
    compuestos_encontrados = {}
    for frase_spa, token_mslg in dicc_compuestos.items():
        if frase_spa in texto_lower:
            compuestos_encontrados[frase_spa] = token_mslg

    tokens_spacy = list(doc)
    i = 0
    while i < len(tokens_spacy):
        token = tokens_spacy[i]

        if token.is_punct:
            i += 1
            continue

        texto_tok = token.text.lower()

        if token.pos_ == "DET" and texto_tok in STOPWORDS_MSLG:
            i += 1
            continue

        if texto_tok in STOPWORDS_MSLG:
            i += 1
            continue

        if texto_tok in INTENSIFICADORES:
            hay_intensificador = True
            i += 1
            continue

        if texto_tok in MARCADORES_TEMPORALES and token.pos_ == "ADV":
            marcadores_inicio.append(token.lemma_.upper())
            i += 1
            continue

        if token.pos_ == "PROPN" or token.ent_type_ in ("PER", "ORG", "LOC"):
            if texto_tok in SIGLAS_ORG or token.ent_type_ == "ORG":
                tokens_salida.append(token.text.upper())
            elif texto_tok in nombres_personas or texto_tok in NOMBRES_PERSONAS:
                tokens_salida.append("dm-" + token.text.upper())
            elif token.ent_type_ == "LOC" or texto_tok in LUGARES_CONOCIDOS:
                tokens_salida.append(token.text.upper())
            elif token.ent_type_ == "PER":
                tokens_salida.append("dm-" + token.text.upper())
            else:
                tokens_salida.append("dm-" + token.text.upper())
            i += 1
            continue

        if token.pos_ in ("VERB", "AUX") and token.lemma_.lower() in COPULAS:
            i += 1
            continue

        if token.lemma_.lower() == "hacer" and token.pos_ in ("VERB", "AUX"):
            CLIMA = {"calor", "frío", "frio", "sol", "viento", "fresco", "humedad"}
            resto = [t.lemma_.lower() for t in tokens_spacy[i + 1:] if not t.is_punct]
            if any(c in resto for c in CLIMA):
                i += 1
                continue

        if token.pos_ in ("VERB", "AUX"):
            tokens_salida.append(token.lemma_.upper())
            i += 1
            continue

        if token.pos_ == "NOUN" and token.morph.get("Gender") == ["Fem"]:
            lema = token.lemma_.lower()
            if lema in NO_GENERO:
                tokens_salida.append(lema.upper())
                i += 1
                continue
            if lema in MASCULINOS_IRREGULARES:
                base = MASCULINOS_IRREGULARES[lema].upper()
            else:
                if lema.endswith("a"):
                    base = lema[:-1] + "o"
                else:
                    base = lema
                base = base.upper()
            tokens_salida.append(base + "+MUJER")
            i += 1
            continue

        tokens_salida.append(token.lemma_.upper())
        i += 1

    resultado = marcadores_inicio + tokens_salida

    if hay_intensificador and len(resultado) >= 1:
        resultado.insert(-1, "MUCHO")

    return " ".join(resultado) if resultado else oracion_spa.upper()


_MARCADORES_UP = {m.upper() for m in MARCADORES_TEMPORALES}


def es_fol_degenerado(gloss_fol, oracion_spa):
    """True si el candidato FOL no aportó señal útil y solo replicó la entrada.

    Un FOL útil activa al menos una de estas señales:
      - prefijo dm- (nombres propios)
      - sufijo +MUJER (género femenino)
      - token MUCHO (intensificador)
      - token compuesto con guión (no dm-)
      - marcador temporal movido al inicio
      - reducción sensible de tokens (artículos/preposiciones/cópulas eliminados)

    Si ninguna señal se activó, el candidato es pura mayúscula del SPA y
    conviene hacer fallback a RAG puro para evitar sesgar al LLM.
    """
    tokens_fol = gloss_fol.split()
    if not tokens_fol:
        return True

    tiene_dm = any(t.startswith("dm-") for t in tokens_fol)
    tiene_mujer = any("+MUJER" in t for t in tokens_fol)
    tiene_mucho = "MUCHO" in tokens_fol
    tiene_compuesto = any("-" in t and not t.startswith("dm-") for t in tokens_fol)
    tiene_temporal_inicio = tokens_fol[0] in _MARCADORES_UP

    palabras_spa = [w for w in oracion_spa.split() if any(c.isalpha() for c in w)]
    ratio_reduccion = 1 - (len(tokens_fol) / max(len(palabras_spa), 1))
    tiene_reduccion = ratio_reduccion > 0.15

    senales = (
        tiene_dm or tiene_mujer or tiene_mucho or tiene_compuesto
        or tiene_temporal_inicio or tiene_reduccion
    )
    return not senales
