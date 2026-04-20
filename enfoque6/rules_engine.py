"""Motor de reglas FOL SPA→MSLG.

Adapta las reglas de enfoque1 para producir:
  - draft: glosa determinística (puede tener errores)
  - annotations: metadatos estructurados para enriquecer el prompt híbrido
"""

# ── Constantes léxicas ────────────────────────────────────────────────────────

MASCULINOS_IRREGULARES = {
    "madre": "padre", "mamá": "papá", "mujer": "hombre",
    "reina": "rey", "actriz": "actor", "emperatriz": "emperador",
    "abuela": "abuelo", "tía": "tío", "nuera": "nuero",
}

NO_GENERO = {
    "sopa", "moneda", "acta", "mesa", "casa", "agua", "hora", "forma",
    "cama", "cara", "mano", "foto", "nota", "lista", "carta", "bolsa",
    "crema", "planta", "trampa", "copa", "firma", "cuenta", "vista",
    "cinta", "llama", "palma", "zona", "manga", "carga", "plaza",
    "pasta", "taza", "falta", "marca", "masa", "pista", "selva",
    "playa", "orilla", "rama", "rueda", "ruta", "sangre", "sombra",
    "tienda", "tierra", "vuelta", "palabra", "persona", "cosa",
    "parte", "vez", "vida", "noche", "tarde", "mañana", "semana",
    "pregunta", "respuesta", "clase", "lengua", "señal", "escuela",
    "iglesia", "cocina", "medicina", "historia", "música", "mezcla",
    "regla", "tabla", "silla", "ventana", "puerta", "pared", "calle",
    "ciudad", "aldea", "villa", "región", "nación",
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

STOPWORDS_MSLG = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "con", "por", "para",
    "que", "se", "lo", "le", "les", "su", "sus",
}

INTENSIFICADORES = {
    "muy", "mucho", "muchos", "muchas", "demasiado",
    "bastante", "súper", "super", "tan",
}

COPULAS = {"ser", "estar", "parecer"}

SIGLAS_ORG = {
    "ine", "imss", "issste", "sep", "unam", "ipn", "cfe", "pemex",
    "sat", "inegi", "conacyt", "ssa", "cdmx", "shcp", "onu", "oms",
}

NOMBRES_PERSONAS_FIJOS = {
    "santiago", "isabel", "diego", "rivera", "cristóbal", "colón",
    "guadalupe", "victoria", "trinidad", "virginia", "loreto",
    "juan", "maria", "maría", "pedro", "josé", "ana", "carlos",
    "luis", "jorge", "rosa", "elena", "miguel", "alejandro",
}

NEGACIONES_IRREGULARES = {
    "poder": "NO-PODER",
    "haber": "NO-HABER",
    "saber": "NO-SABER",
    "conocer": "NO-CONOCER",
    "gustar": "NO-GUSTAR",
    "querer": "NO-QUERER",
    "hacer": "NO-HACER",
    "servir": "NO-SERVIR",
    "ver": "NO-VER",
}


# ── Motor de reglas ───────────────────────────────────────────────────────────

class RulesEngine:
    """Carga spaCy una sola vez y expone analyze()."""

    def __init__(self, corpus_pairs=None):
        import spacy
        self._nlp = spacy.load("es_core_news_lg")
        self._dicc_compuestos = {}
        self._nombres_corpus = set()

        if corpus_pairs:
            self._dicc_compuestos = _build_compound_dict(corpus_pairs)
            self._nombres_corpus = _build_name_dict(corpus_pairs)

    def analyze(self, sentence: str) -> dict:
        """
        Analiza `sentence` con reglas FOL y devuelve:
          {
            "draft":            str,   # glosa determinística
            "proper_nouns":     list,  # nombres propios detectados
            "temporal_markers": list,  # marcadores de tiempo detectados
            "intensifiers":     list,  # intensificadores detectados
            "is_question":      bool,
            "has_negation":     bool,
            "compounds":        list,  # compuestos multipalabra detectados
          }
        """
        doc = self._nlp(sentence)

        proper_nouns = []
        temporal_markers = []
        intensifiers_found = []
        compounds_found = []
        is_question = "?" in sentence
        has_negation = False

        tokens_out = []
        markers_prepend = []
        negation_next = False

        tokens_spacy = list(doc)
        i = 0

        # Detectar compuestos multipalabra
        texto_lower = sentence.lower()
        for frase, token_mslg in self._dicc_compuestos.items():
            if frase in texto_lower:
                compounds_found.append(f"{frase} → {token_mslg}")

        # Detectar negación
        for tok in tokens_spacy:
            if tok.lemma_.lower() == "no" and tok.pos_ in ("ADV", "PART"):
                has_negation = True
                break

        while i < len(tokens_spacy):
            token = tokens_spacy[i]

            if token.is_punct:
                i += 1
                continue

            texto_tok = token.text.lower()

            # Artículos y stopwords → eliminar
            if token.pos_ == "DET" and texto_tok in STOPWORDS_MSLG:
                i += 1
                continue
            if texto_tok in STOPWORDS_MSLG:
                i += 1
                continue

            # Negación "no" → registrar y usar forma irregular si aplica
            if texto_tok == "no" and token.pos_ in ("ADV", "PART"):
                negation_next = True
                i += 1
                continue

            # Intensificadores → MUCHO
            if texto_tok in INTENSIFICADORES:
                intensifiers_found.append(texto_tok)
                tokens_out.append("MUCHO")
                i += 1
                continue

            # Marcadores temporales → inicio
            if texto_tok in MARCADORES_TEMPORALES and token.pos_ == "ADV":
                marcador = token.lemma_.upper()
                temporal_markers.append(marcador)
                markers_prepend.append(marcador)
                i += 1
                continue

            # Nombres propios
            if token.pos_ == "PROPN" or token.ent_type_ in ("PER", "ORG", "LOC"):
                if texto_tok in SIGLAS_ORG or token.ent_type_ == "ORG":
                    tokens_out.append(token.text.upper())
                elif (texto_tok in self._nombres_corpus
                      or texto_tok in NOMBRES_PERSONAS_FIJOS
                      or token.ent_type_ == "PER"):
                    glosa = "dm-" + token.text.upper()
                    tokens_out.append(glosa)
                    proper_nouns.append(f"{token.text} → {glosa}")
                elif token.ent_type_ == "LOC" or texto_tok in LUGARES_CONOCIDOS:
                    tokens_out.append(token.text.upper())
                else:
                    glosa = "dm-" + token.text.upper()
                    tokens_out.append(glosa)
                    proper_nouns.append(f"{token.text} → {glosa}")
                i += 1
                continue

            # Verbos cópula → eliminar
            if token.pos_ in ("VERB", "AUX") and token.lemma_.lower() in COPULAS:
                i += 1
                continue

            # HACER impersonal en expresiones climáticas
            if token.lemma_.lower() == "hacer" and token.pos_ in ("VERB", "AUX"):
                CLIMA = {"calor", "frío", "frio", "sol", "viento", "fresco", "humedad"}
                resto = [t.lemma_.lower() for t in tokens_spacy[i + 1:] if not t.is_punct]
                if any(c in resto for c in CLIMA):
                    i += 1
                    continue

            # Verbos → infinitivo
            if token.pos_ in ("VERB", "AUX"):
                lema = token.lemma_.lower()
                if negation_next and lema in NEGACIONES_IRREGULARES:
                    tokens_out.append(NEGACIONES_IRREGULARES[lema])
                    negation_next = False
                else:
                    if negation_next:
                        tokens_out.append("NO")
                        negation_next = False
                    tokens_out.append(lema.upper())
                i += 1
                continue

            # Género femenino de persona → MASCULINO+MUJER
            if token.pos_ == "NOUN" and token.morph.get("Gender") == ["Fem"]:
                lema = token.lemma_.lower()
                if lema in NO_GENERO:
                    tokens_out.append(lema.upper())
                elif lema in MASCULINOS_IRREGULARES:
                    base = MASCULINOS_IRREGULARES[lema].upper()
                    tokens_out.append(base + "+MUJER")
                else:
                    base = (lema[:-1] + "o" if lema.endswith("a") else lema).upper()
                    tokens_out.append(base + "+MUJER")
                i += 1
                continue

            # Default
            tokens_out.append(token.lemma_.upper())
            i += 1

        # Negación pendiente al final (sin verbo siguiente)
        if negation_next:
            tokens_out.append("NO")

        draft_tokens = markers_prepend + tokens_out
        draft = " ".join(draft_tokens) if draft_tokens else sentence.upper()

        return {
            "draft": draft,
            "proper_nouns": proper_nouns,
            "temporal_markers": temporal_markers,
            "intensifiers": intensifiers_found,
            "is_question": is_question,
            "has_negation": has_negation,
            "compounds": compounds_found,
        }


# ── Helpers para construir diccionarios desde corpus ─────────────────────────

def _build_compound_dict(pairs):
    dicc = {}
    for par in pairs:
        for token in par["mslg"].split():
            if "-" in token and not token.startswith("dm-"):
                partes = token.replace("-", " ").lower()
                spa = par["spa"].lower()
                if all(p in spa for p in partes.split()):
                    dicc[partes] = token
    return dicc


def _build_name_dict(pairs):
    nombres = set()
    for par in pairs:
        spa_lower = par["spa"].lower()
        for token in par["mslg"].split():
            if token.startswith("dm-"):
                nombre = token[3:].lower()
                if nombre in spa_lower:
                    nombres.add(nombre)
    return nombres
