import csv
import re
import unicodedata
from collections import defaultdict

import spacy
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# 1. CARGAR DATASET

def cargar_dataset(ruta_tsv: str):
    """
    Carga el TSV con columnas: ID, MSLG, SPA
    Devuelve lista de dicts.
    """
    pares = []
    with open(ruta_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fila in reader:
            pares.append({
                "id":   fila["ID"].strip(),
                "mslg": fila["MSLG"].strip(),
                "spa":  fila["SPA"].strip(),
            })
    return pares


def split_train_val(pares, val_ratio=0.18, seed=42):
    """400 train / 90 val aproximadamente."""
    import random
    random.seed(seed)
    datos = pares[:]
    random.shuffle(datos)
    corte = int(len(datos) * (1 - val_ratio))
    return datos[:corte], datos[corte:]


# 2. CONSTRUIR DICCIONARIO DE COMPUESTOS (extrae automáticamente del corpus)

def construir_diccionario_compuestos(pares_train):
    """
    Busca tokens con guión en las glosas MSLG y trata de mapearlos a su equivalente en español mirando el par alineado.
    Devuelve dict: frase_spa -> TOKEN-CON-GUIÓN
    """
    dicc = {}
    for par in pares_train:
        tokens_mslg = par["mslg"].split()
        for token in tokens_mslg:
            if "-" in token and not token.startswith("dm-"):
                # Convertir el token a palabras candidatas en español
                partes = token.replace("-", " ").lower()
                spa = par["spa"].lower()
                # Solo guardar si todas las partes aparecen en el SPA
                if all(p in spa for p in partes.split()):
                    dicc[partes] = token
    return dicc


# 3. ÍNDICE DE SIMILITUD PARA FALLBACK

def construir_indice_similitud(pares_train):
    """
    Índice simple basado en overlap de bigramas (sin GPU).
    Para cada oración de test, busca el par más similar del corpus.
    """
    def bigramas(texto):
        tokens = texto.lower().split()
        return set(zip(tokens, tokens[1:])) | set(tokens)  # unigramas + bigramas

    indice = [(par, bigramas(par["spa"])) for par in pares_train]
    return indice


def buscar_mas_similar(oracion_spa, indice, top_k=3):
    """Devuelve los top_k pares más similares por overlap de bigramas."""
    bg_query = set(oracion_spa.lower().split())
    # incluir bigramas
    tokens = oracion_spa.lower().split()
    bg_query |= set(zip(tokens, tokens[1:]))

    scores = []
    for par, bg_corpus in indice:
        overlap = len(bg_query & bg_corpus) / (len(bg_query | bg_corpus) + 1e-9)
        scores.append((overlap, par))

    scores.sort(reverse=True)
    return [par for _, par in scores[:top_k]]


# 4. REGLAS DE TRANSFORMACIÓN SPA → MSLG

# Géneros femeninos irregulares conocidos (femenino → masculino)
MASCULINOS_IRREGULARES = {
    "madre": "padre", "mamá": "papá", "mujer": "hombre",
    "reina": "rey", "actriz": "actor", "emperatriz": "emperador",
    "yerna": "yerno", "nuera": "nuero",
}

# Sustantivos femeninos que NO tienen par masculino (no aplicar +MUJER)
# Son cosas/conceptos, no personas con género gramatical relevante en LSM
NO_GENERO = {
    "sopa", "moneda", "acta", "mesa", "casa", "agua", "hora", "forma",
    "cama", "cara", "mano", "foto", "nota", "lista", "carta", "bolsa",
    "crema", "planta", "trampa", "copa", "firma", "cuenta", "vista",
    "cinta", "llama", "palma", "zona", "manga", "carga", "plaza",
    "pasta", "taza", "falta", "marca", "masa", "pista", "selva",
    "playa", "orilla", "rama", "rueda", "ruta", "sangre", "sombra",
    "tienda", "tierra", "trampa", "vuelta", "palabra", "persona",
    "cosa", "parte", "vez", "vida", "noche", "tarde", "mañana",
    "semana", "pregunta", "respuesta", "clase", "lengua", "señal",
    "credencial", "delegación", "moneda", "antena", "corona",
    "huella", "escuela", "iglesia", "cocina", "medicina", "historia",
    "música", "física", "química", "biología", "economía",
    "tecnología", "matemáticas", "literatura", "cultura",
    "mezcla", "regla", "tabla", "silla", "ventana", "puerta", "pared",
    "calle", "ciudad", "aldea", "villa", "región", "nación",
}

# Lugares conocidos que NO deben llevar dm- (no son personas)
LUGARES_CONOCIDOS = {
    "villahermosa", "guadalajara", "monterrey", "puebla", "oaxaca",
    "tijuana", "mérida", "cancún", "veracruz", "acapulco", "toluca",
    "mexico", "méxico", "tepito", "tlatelolco", "iztapalapa",
    "san francisco", "new york", "nueva york", "estados unidos",
    "california", "texas", "florida", "chicago",
}

# Palabras que ya son glosas MSLG válidas (no necesitan transformación)
MARCADORES_TEMPORALES = {
    "ayer", "mañana", "hoy", "siempre", "nunca", "después",
    "antes", "ahora", "tarde", "temprano", "pronto",
}

STOPWORDS_MSLG = {"el", "la", "los", "las", "un", "una", "unos", "unas",
                  "de", "del", "al", "a", "en", "con", "por", "para",
                  "que", "se", "lo", "le", "les",
                  "su", "sus"}  # posesivos de tercera persona → eliminar

INTENSIFICADORES = {"muy", "mucho", "muchos", "muchas", "demasiado",
                    "bastante", "súper", "super", "tan"}

# Verbos cópula que se eliminan en LSM (el estado se infiere del adjetivo)
COPULAS = {"ser", "estar", "parecer"}

# Siglas y organizaciones que NO deben llevar dm-
SIGLAS_ORG = {"ine", "imss", "issste", "sep", "unam", "ipn", "cfe", "pemex",
              "sat", "inegi", "conacyt", "ssa", "cdmx", "shcp", "onu", "oms"}

# Nombres propios de personas como respaldo (para los que spaCy confunde con lugares)
# El diccionario dinámico del corpus tiene prioridad sobre esta lista
NOMBRES_PERSONAS = {
    "santiago", "isabel", "diego", "rivera", "cristóbal", "colón",
    "guadalupe", "victoria", "trinidad", "virginia", "loreto",
    "juan", "maria", "maría", "pedro", "josé", "ana", "carlos",
    "luis", "jorge", "rosa", "elena", "miguel", "alejandro",
}


def construir_diccionario_nombres(pares_train):
    """
    Extrae automáticamente del corpus todos los nombres que llevan dm-
    en las glosas MSLG y los mapea a su forma en español.
    Devuelve set de nombres en minúsculas que deben llevar dm-.
    """
    nombres = set()
    for par in pares_train:
        tokens_mslg = par["mslg"].split()
        spa_lower = par["spa"].lower()
        for token in tokens_mslg:
            if token.startswith("dm-"):
                # Extraer el nombre sin el prefijo dm-
                nombre_mslg = token[3:].lower()  # ej: "dm-SANTIAGO" → "santiago"
                # Verificar que aparece en el español del par
                if nombre_mslg in spa_lower:
                    nombres.add(nombre_mslg)
    return nombres





def aplicar_reglas_spa_a_mslg(oracion_spa: str, nlp, dicc_compuestos: dict, nombres_personas: set) -> str:
    """
    Aplica el pipeline completo de reglas FOL para convertir
    una oración en español a glosas MSLG.
    """
    doc = nlp(oracion_spa)

    tokens_salida = []
    marcadores_inicio = []
    hay_intensificador = False

    # ── Paso 1: Detectar compuestos (buscar frases multipalabra del diccionario) ──
    texto_lower = oracion_spa.lower()
    compuestos_encontrados = {}
    for frase_spa, token_mslg in dicc_compuestos.items():
        if frase_spa in texto_lower:
            compuestos_encontrados[frase_spa] = token_mslg

    # ── Paso 2: Recorrer tokens con spaCy ──
    i = 0
    tokens_spacy = list(doc)
    usados = set()  # índices de tokens ya consumidos por compuestos

    while i < len(tokens_spacy):
        token = tokens_spacy[i]

        # Saltar puntuación
        if token.is_punct:
            i += 1
            continue

        texto_tok = token.text.lower()

        # ── Regla: Artículos → eliminar ──
        if token.pos_ == "DET" and texto_tok in STOPWORDS_MSLG:
            i += 1
            continue

        # ── Regla: Stopwords ──
        if texto_tok in STOPWORDS_MSLG:
            i += 1
            continue

        # ── Regla: Intensificadores → MUCHO ──
        if texto_tok in INTENSIFICADORES:
            hay_intensificador = True
            i += 1
            continue

        # ── Regla: Marcadores temporales → anteponer ──
        if texto_tok in MARCADORES_TEMPORALES and token.pos_ == "ADV":
            marcadores_inicio.append(token.lemma_.upper())
            i += 1
            continue

        # ── Regla: Nombres propios → dm- solo para PERSONAS ──
        if token.pos_ == "PROPN" or token.ent_type_ in ("PER", "ORG", "LOC"):
            # Siglas y organizaciones → sin dm-
            if texto_tok in SIGLAS_ORG or token.ent_type_ == "ORG":
                tokens_salida.append(token.text.upper())
            # Nombre extraído del corpus O en lista fija → siempre con dm-
            elif texto_tok in nombres_personas or texto_tok in NOMBRES_PERSONAS:
                tokens_salida.append("dm-" + token.text.upper())
            # Lugares → sin dm-
            elif token.ent_type_ == "LOC" or texto_tok in LUGARES_CONOCIDOS:
                tokens_salida.append(token.text.upper())
            # Nombres conocidos de personas → con dm-
            elif token.ent_type_ == "PER":
                tokens_salida.append("dm-" + token.text.upper())
            # Cualquier otro PROPN ambiguo → con dm-
            else:
                tokens_salida.append("dm-" + token.text.upper())
            i += 1
            continue

        # ── Regla: Verbos cópula (ser/estar) → eliminar (VERB y AUX) ──
        if token.pos_ in ("VERB", "AUX") and token.lemma_.lower() in COPULAS:
            i += 1
            continue

        # ── Regla: HACER en expresiones de clima/impersonal → eliminar ──
        # Ej: "hace calor", "hace frío", "hace mucho calor"
        if token.lemma_.lower() == "hacer" and token.pos_ in ("VERB", "AUX"):
            CLIMA = {"calor", "frío", "frio", "sol", "viento", "fresco", "humedad"}
            resto = [t.lemma_.lower() for t in tokens_spacy[i+1:] if not t.is_punct]
            if any(c in resto for c in CLIMA):
                i += 1
                continue

        # ── Regla: Verbos → lema (infinitivo) ──
        if token.pos_ in ("VERB", "AUX"):
            tokens_salida.append(token.lemma_.upper())
            i += 1
            continue

        # ── Regla: Género femenino → MASCULINO+MUJER ──
        # Solo aplica a sustantivos que refieren PERSONAS, no a cosas
        if token.pos_ == "NOUN" and token.morph.get("Gender") == ["Fem"]:
            lema = token.lemma_.lower()
            # Si es una cosa/concepto, no aplicar +MUJER
            if lema in NO_GENERO:
                tokens_salida.append(lema.upper())
                i += 1
                continue
            if lema in MASCULINOS_IRREGULARES:
                base = MASCULINOS_IRREGULARES[lema].upper()
            else:
                # Heurística: quitar -a final y agregar -o
                if lema.endswith("a"):
                    base = lema[:-1] + "o"
                else:
                    base = lema
                base = base.upper()
            tokens_salida.append(base + "+MUJER")
            i += 1
            continue

        # ── Default: usar lema en mayúsculas ──
        tokens_salida.append(token.lemma_.upper())
        i += 1

    # ── Paso 3: Ensamblar ──
    resultado = marcadores_inicio + tokens_salida

    # ── Regla: Intensificadores → MUCHO antes del último token ──
    # En LSM el intensificador generalmente precede al concepto que modifica
    if hay_intensificador and len(resultado) >= 1:
        resultado.insert(-1, "MUCHO")

    return " ".join(resultado) if resultado else oracion_spa.upper()


# 5. TRADUCCIÓN CON FALLBACK

def traducir_con_fallback(oracion_spa: str, nlp, dicc_compuestos: dict,
                          indice_similitud, nombres_personas: set, umbral_confianza=0.15):
    """
    Intenta traducir con reglas. Si la oración parece fuera de cobertura
    (muy corta o sin tokens reconocibles), usa fallback por similitud.
    """
    resultado = aplicar_reglas_spa_a_mslg(oracion_spa, nlp, dicc_compuestos, nombres_personas)

    # Fallback: si el resultado es idéntico al input (nada cambió), usar similitud
    if resultado.strip() == oracion_spa.strip().upper() or len(resultado.split()) <= 1:
        similares = buscar_mas_similar(oracion_spa, indice_similitud, top_k=1)
        if similares:
            return similares[0]["mslg"]  # devolver la glosa del par más similar

    return resultado


# 6. EVALUACIÓN

def evaluar(predicciones, referencias):
    """
    Calcula BLEU, chrF y METEOR.
    predicciones: lista de strings
    referencias:  lista de strings
    """
    # BLEU (sacrebleu espera lista de hipótesis y lista de listas de referencias)
    bleu = sacrebleu.corpus_bleu(predicciones, [referencias])

    # chrF
    chrf = sacrebleu.corpus_chrf(predicciones, [referencias])

    # METEOR (promedio por oración)
    meteor_scores = []
    for pred, ref in zip(predicciones, referencias):
        score = meteor_score([ref.split()], pred.split())
        meteor_scores.append(score)
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    return {
        "BLEU":   round(bleu.score, 4),
        "chrF":   round(chrf.score, 4),
        "METEOR": round(meteor_avg * 100, 4),
    }


# 7. MAIN

def main():
    RUTA_DATASET = "enfoque1/data.txt"  

    print("Cargando dataset...")
    pares = cargar_dataset(RUTA_DATASET)
    train, val = split_train_val(pares)
    print(f"  Train: {len(train)} pares | Val: {len(val)} pares")

    print("Construyendo diccionario de compuestos...")
    dicc_compuestos = construir_diccionario_compuestos(train)
    print(f"  {len(dicc_compuestos)} compuestos encontrados")
    for k, v in list(dicc_compuestos.items())[:5]:
        print(f"    '{k}' → {v}")

    print("Construyendo diccionario de nombres con dm-...")
    nombres_personas = construir_diccionario_nombres(train)
    print(f"  {len(nombres_personas)} nombres extraídos del corpus")
    print(f"    Ejemplos: {sorted(nombres_personas)[:8]}")

    print("Construyendo índice de similitud...")
    indice = construir_indice_similitud(train)

    print("Cargando modelo spaCy...")
    nlp = spacy.load("es_core_news_lg")

    # ── Traducir conjunto de validación ──
    print("\nTraduciendo conjunto de validación...")
    predicciones = []
    referencias  = []

    for par in val:
        pred = traducir_con_fallback(
            par["spa"], nlp, dicc_compuestos, indice, nombres_personas
        )
        predicciones.append(pred)
        referencias.append(par["mslg"])

    # ── Mostrar ejemplos ──
    print("\n── Ejemplos de traducción ──")
    for par, pred in zip(val[:8], predicciones[:8]):
        print(f"  SPA:  {par['spa']}")
        print(f"  REF:  {par['mslg']}")
        print(f"  PRED: {pred}")
        print()

    # ── Métricas ──
    print("── Métricas en validación (SPA→MSLG) ──")
    metricas = evaluar(predicciones, referencias)
    for k, v in metricas.items():
        print(f"  {k}: {v}")

    # ── Guardar predicciones ──
    with open("predicciones_e1_val.tsv", "w", encoding="utf-8") as f:
        f.write("ID\tSPA\tREF_MSLG\tPRED_MSLG\n")
        for par, pred in zip(val, predicciones):
            f.write(f"{par['id']}\t{par['spa']}\t{par['mslg']}\t{pred}\n")
    print("\nPredicciones guardadas en predicciones_e1_val.tsv")


if __name__ == "__main__":
    main()