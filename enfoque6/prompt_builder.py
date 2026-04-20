"""Construcción de prompts para Enfoque 6.

Extiende el sistema de prompts de enfoque3 con:
  1. Reglas gramaticales completas de la LSM (desde reglas_sintaxis_lsm.md).
  2. Prompts híbridos que incluyen el borrador del motor de reglas FOL.
"""

# ── Sistema de reglas extendido ───────────────────────────────────────────────

PROMPT_BASE = """\
Eres un traductor experto de español a glosas de Lengua de Señas Mexicana (LSM/MSLG).
Tu tarea es traducir la oración en español a su equivalente en glosas LSM.
Responde ÚNICAMENTE con la glosa traducida, sin explicaciones ni texto adicional.

═══ REGLAS GRAMATICALES DE LA LSM ═══

ORDEN DE CONSTITUYENTES:
- Orden básico: Sujeto-Verbo-Objeto (SVO).
- Con verbos demostrativos o complemento focalizado: Sujeto-Objeto-Verbo (SOV).
- Topicalización: el elemento más importante va AL INICIO de la oración.
- Marcadores de TIEMPO (ayer, hoy, mañana, antes, después, siempre) van SIEMPRE al INICIO.
- Adjetivos van DESPUÉS del sustantivo que modifican: "perro negro" → PERRO NEGRO.
- Adverbios de modo van DESPUÉS del verbo.
- Circunstanciales de tiempo específico (años, meses, fechas) van DESPUÉS del verbo.

ELIMINACIONES:
- Eliminar todos los artículos: el, la, los, las, un, una, unos, unas.
- Eliminar preposiciones: de, en, a, por, para — salvo dentro de conceptos compuestos.
- Eliminar verbos cópula (ser, estar, parecer): la predicación se expresa por yuxtaposición.
  Ejemplo: "María es maestra" → MARÍA MAESTRO+MUJER
- Eliminar posesivos de tercera persona (su, sus) cuando el contexto los hace obvios.

TRANSFORMACIONES OBLIGATORIAS:
- Verbos: SIEMPRE en forma infinitiva en MAYÚSCULAS (llegó → LLEGAR, cenaste → CENAR).
- Nombres propios de personas: prefijo dm- (Isabel → dm-ISABEL, Diego Rivera → dm-DIEGO-RIVERA).
  No aplica a organizaciones (UNAM, INE) ni lugares (VERACRUZ, GUADALAJARA).
- Sustantivos femeninos que designan personas: forma masculina + MUJER.
  hermana → HERMANO+MUJER | maestra → MAESTRO+MUJER | tía → TÍO+MUJER | amiga → AMIGO+MUJER.
  Excepciones: sustantivos que no designan personas (mesa, casa, hora) → no aplica +MUJER.
- Intensificadores (muy, demasiado, bastante, súper): MUCHO después del adjetivo.
  Alternativa: VERDAD al final para "muy" + adjetivo evaluativo.
- Conceptos compuestos: unir con guión (cinturón de seguridad → CINTURÓN-DE-SEGURIDAD).
- Todo en MAYÚSCULAS.

NEGACIÓN:
- Forma general: NO antes o después del verbo + rotación lateral de cabeza (RNM).
- Nueve verbos tienen forma negativa irregular (preferir estas formas):
  no poder → NO-PODER | no haber/no hay → NO-HAY o NO-HABER | no saber → NO-SABER
  no querer → NO-QUERER | no hacer → NO-HACER | no gustar → NO-GUSTAR
  no conocer → NO-CONOCER | no ver → NO-VER | no poder → NO-PODER
- Negación existencial: NO-HAY o NADA.

INTERROGACIÓN:
- Conservar ¿...?.
- CÓMO: antes del verbo (o duplicado: inicio + final).
- CUÁNDO: inicio, antes del verbo, o duplicado (inicio + final).
- DÓNDE: inicio de la oración o antes del verbo.
- CUÁNTO: al FINAL de la oración.
- POR-QUÉ: al FINAL de la oración.
- QUIÉN: al INICIO.
- QUÉ: al inicio o antes del sustantivo.
- Las palabras interrogativas pueden duplicarse (inicio Y final) para énfasis.

POSESIVOS Y PRONOMBRES:
- Posesivos (mi, tu, nuestro): SÍ conservar → MI, TU, NUESTRO.
- Pronombres personales: YO, TÚ, ÉL, ELLA, NOSOTROS, USTEDES, ELLOS.
- Pronombres de 1ª y 2ª persona como objeto suelen ir ANTES del verbo (SOV).

COORDINACIÓN Y SUBORDINACIÓN:
- Coordinación copulativa: usar Y o simplemente yuxtaponer oraciones.
- Adversativas: PERO entre oraciones.
- Condicionales: IMAGINAR [condición] [consecuencia].
- Oraciones subordinadas: generalmente por yuxtaposición sin conjunción explícita.\
"""

# 25 ejemplos fijos curados para cubrir categorías principales:
# afirmativa, negativa (con/sin forma irregular), interrogativa, pasado, futuro,
# nombres propios, lugares, compuestos, intensificadores, posesivos, plurales,
# números, coordinación, condicional, cópula, femenino+MUJER, topicalización.
FIXED_EXAMPLES = [
    {"spa": "Los peces azules son mis favoritos.",
     "mslg": "PEZ COLOR AZUL MI FAVORITO"},
    {"spa": "Isabel tiene una corona de oro.",
     "mslg": "dm-ISABEL TENER SU CORONA ORO"},
    {"spa": "Mi hermana está embarazada.",
     "mslg": "MI HERMANO+MUJER YA EMBARAZADA"},
    {"spa": "¿Ya cenaste?",
     "mslg": "¿TÚ YA CENAR?"},
    {"spa": "Ayer llegó mi tío de San Francisco.",
     "mslg": "AYER MI TIO SAN FRANCISCO YA LLEGAR"},
    {"spa": "Esa antena es muy cara.",
     "mslg": "ESA ANTENA CARA VERDAD"},
    {"spa": "Debes pagarme a fuerza.",
     "mslg": "A-FUERZA TÚ DEBER TÚ PAGAR MÍ"},
    {"spa": "La hija de mi vecina tiene autismo.",
     "mslg": "MI VECINO+MUJER SU HIJO+MUJER TENER AUTISMO"},
    {"spa": "Visité el mural de Diego Rivera.",
     "mslg": "YO YA VISITAR MURAL dm-DIEGO-RIVERA"},
    {"spa": "Olvidé mi contraseña.",
     "mslg": "YO OLVIDAR CONTRASEÑA"},
    {"spa": "El cumpleaños de mi tía es en abril y le haremos fiesta.",
     "mslg": "ABRIL CUMPLEAÑOS MI TÍO+MUJER NOSOTROS HACER FIESTA"},
    {"spa": "No traigo efectivo.",
     "mslg": "EFECTIVO NINGÚN YO TRAER"},
    {"spa": "Mi auto no tiene cinturón de seguridad.",
     "mslg": "CINTURÓN-DE-SEGURIDAD MI AUTO NO-HAY"},
    {"spa": "Mi amiga y su novio van los sábados al casino.",
     "mslg": "TODOS-SÁBADOS MI AMIGO+MUJER SUYO NOVIO CASINO IR"},
    {"spa": "Mi tía cocina delicioso.",
     "mslg": "MI TÍO+MUJER DELICIOSO COCINAR"},
    {"spa": "Hoy es mi cumpleaños.",
     "mslg": "HOY MI CUMPLEAÑOS"},
    {"spa": "No sé dónde está mi libro.",
     "mslg": "MI LIBRO DÓNDE YO NO-SABER"},
    {"spa": "¿Cuánto cuesta el pan?",
     "mslg": "PAN COSTAR CUÁNTO"},
    {"spa": "Mañana iremos al cine con mis primos.",
     "mslg": "MAÑANA NOSOTROS MI PRIMO CINE IR"},
    {"spa": "Si llueve, me quedo en casa.",
     "mslg": "IMAGINAR LLOVER YO CASA QUEDAR"},
    {"spa": "Tengo tres hermanos y una hermana.",
     "mslg": "YO HERMANO 3 HERMANO+MUJER 1 TENER"},
    {"spa": "¿Por qué no viniste ayer?",
     "mslg": "AYER TÚ NO-VENIR POR-QUÉ"},
    {"spa": "El perro negro ladra mucho.",
     "mslg": "PERRO NEGRO LADRAR MUCHO"},
    {"spa": "Nunca he viajado a Europa.",
     "mslg": "EUROPA YO NUNCA VIAJAR"},
    {"spa": "María trabaja en el hospital.",
     "mslg": "dm-MARÍA HOSPITAL TRABAJAR"},
]

# Glosario inline: términos SPA → glosa LSM frecuente.
# Ayuda al modelo en zero-shot a no alucinar el léxico.
LSM_GLOSSARY = """\
GLOSARIO DE REFERENCIA (SPA → LSM):
Pronombres: yo→YO, tú→TÚ, él/ella→ÉL/ELLA, nosotros→NOSOTROS, ustedes→USTEDES, ellos→ELLOS.
Posesivos: mi→MI, tu→TU, su→SU, nuestro→NUESTRO, mío→MÍO, suyo→SUYO.
Tiempo: ayer→AYER, hoy→HOY, mañana→MAÑANA, antes→ANTES, después→DESPUÉS, siempre→SIEMPRE, nunca→NUNCA, ahora→AHORA, ya→YA.
Interrogativas: qué→QUÉ, quién→QUIÉN, dónde→DÓNDE, cuándo→CUÁNDO, cómo→CÓMO, cuánto→CUÁNTO, por qué→POR-QUÉ.
Cantidad: mucho→MUCHO, poco→POCO, todo→TODO, nada→NADA, algo→ALGO, más→MÁS, menos→MENOS.
Verbos comunes: ir→IR, venir→VENIR, tener→TENER, querer→QUERER, poder→PODER, saber→SABER, hacer→HACER, ver→VER, comer→COMER, beber→BEBER, trabajar→TRABAJAR, estudiar→ESTUDIAR, vivir→VIVIR, dormir→DORMIR.
Negativos irregulares: no poder→NO-PODER, no haber→NO-HAY, no saber→NO-SABER, no querer→NO-QUERER, no hacer→NO-HACER, no gustar→NO-GUSTAR, no conocer→NO-CONOCER, no ver→NO-VER.
Conectores: y→Y, pero→PERO, si→IMAGINAR (condicional).
Familia (forma masculina + MUJER para femenino): hermana→HERMANO+MUJER, tía→TÍO+MUJER, hija→HIJO+MUJER, amiga→AMIGO+MUJER, madre→MAMÁ, padre→PAPÁ.\
"""

# Ejemplos negativos: muestran errores frecuentes que el modelo debe evitar.
NEGATIVE_EXAMPLES = [
    {"spa": "Mi madre cocina arroz.",
     "mal":  "LA MADRE COCINA EL ARROZ",
     "bien": "MI MAMÁ ARROZ COCINAR",
     "error": "no eliminó artículos y no convirtió verbo a infinitivo"},
    {"spa": "Ayer fui al mercado.",
     "mal":  "YO IR MERCADO AYER",
     "bien": "AYER YO MERCADO IR",
     "error": "marcador temporal no va al INICIO"},
    {"spa": "Mi hermana es alta.",
     "mal":  "MI HERMANA SER ALTA",
     "bien": "MI HERMANO+MUJER ALTA",
     "error": "no aplicó +MUJER y conservó cópula SER"},
]

# Instrucciones de chain-of-thought estructurado.
COT_INSTRUCTIONS = """\
RAZONA ANTES DE RESPONDER (no muestres los pasos, solo la glosa final):
  1. Identifica: tiempo, sujeto, objeto, verbo principal, negación, interrogativa.
  2. Elimina: artículos, preposiciones (salvo en compuestos), verbos cópula.
  3. Transforma: verbos a infinitivo MAYÚSCULAS, femeninos a MASCULINO+MUJER, nombres propios con dm-.
  4. Reordena: tiempo al inicio; interrogativas según regla (CUÁNTO/POR-QUÉ al final, QUIÉN al inicio).
  5. Emite SOLO la glosa final en una línea, sin explicar.\
"""


# ── Constructores de prompts ──────────────────────────────────────────────────

def build_zero_shot(sentence: str) -> str:
    return f"""{PROMPT_BASE}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot(sentence: str, k: int = 5) -> str:
    examples = FIXED_EXAMPLES[:k]
    examples_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in examples
    )
    return f"""{PROMPT_BASE}

EJEMPLOS:
{examples_text}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_rag(sentence: str, retrieved_examples: list) -> str:
    examples_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in retrieved_examples
    )
    return f"""{PROMPT_BASE}

EJEMPLOS (similares a la oración a traducir):
{examples_text}

Traduce:
SPA: "{sentence}"
MSLG:"""


def _format_analysis(analysis: dict) -> str:
    """Formatea las anotaciones del motor de reglas como texto para el prompt."""
    lines = []
    if analysis["temporal_markers"]:
        lines.append(f"- Marcadores temporales (van al INICIO): {', '.join(analysis['temporal_markers'])}")
    if analysis["proper_nouns"]:
        lines.append(f"- Nombres propios detectados: {', '.join(analysis['proper_nouns'])}")
    if analysis["has_negation"]:
        lines.append("- Negación detectada: verificar si aplica forma irregular (NO-PODER, NO-HABER, etc.)")
    if analysis["intensifiers"]:
        lines.append(f"- Intensificadores detectados: {', '.join(analysis['intensifiers'])} → usar MUCHO o VERDAD")
    if analysis["compounds"]:
        lines.append(f"- Posibles compuestos: {', '.join(analysis['compounds'])}")
    if analysis["is_question"]:
        lines.append("- Es pregunta: conservar ¿...? y aplicar posición correcta de la palabra interrogativa")
    lines.append(f"- Borrador automático (reglas FOL, puede tener errores): {analysis['draft']}")
    return "\n".join(lines) if lines else "- Sin anotaciones especiales."


def build_hybrid_zero(sentence: str, analysis: dict) -> str:
    analysis_text = _format_analysis(analysis)
    return f"""{PROMPT_BASE}

ANÁLISIS PREVIO (usa como guía para mejorar el borrador, no es definitivo):
{analysis_text}

Aplica las reglas LSM para corregir y mejorar el borrador. Responde ÚNICAMENTE con la glosa final.

SPA: "{sentence}"
MSLG:"""


def build_hybrid_few(sentence: str, analysis: dict, k: int = 5) -> str:
    examples = FIXED_EXAMPLES[:k]
    examples_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in examples
    )
    analysis_text = _format_analysis(analysis)
    return f"""{PROMPT_BASE}

EJEMPLOS:
{examples_text}

ANÁLISIS PREVIO (usa como guía para mejorar el borrador, no es definitivo):
{analysis_text}

Aplica las reglas LSM para corregir y mejorar el borrador. Responde ÚNICAMENTE con la glosa final.

SPA: "{sentence}"
MSLG:"""


def build_rag_hybrid(sentence: str, analysis: dict, retrieved_examples: list) -> str:
    examples_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in retrieved_examples
    )
    analysis_text = _format_analysis(analysis)
    return f"""{PROMPT_BASE}

EJEMPLOS (similares a la oración a traducir):
{examples_text}

ANÁLISIS PREVIO (usa como guía para mejorar el borrador, no es definitivo):
{analysis_text}

Aplica las reglas LSM para corregir y mejorar el borrador. Responde ÚNICAMENTE con la glosa final.

SPA: "{sentence}"
MSLG:"""


# ── Nuevos constructores: mejoras zero-shot y few-shot ─────────────────────────

def _format_negative_examples() -> str:
    lines = ["EJEMPLOS DE ERRORES FRECUENTES (NO los repitas):"]
    for ex in NEGATIVE_EXAMPLES:
        lines.append(f'  SPA:  "{ex["spa"]}"')
        lines.append(f'  MAL:  "{ex["mal"]}"   ← {ex["error"]}')
        lines.append(f'  BIEN: "{ex["bien"]}"')
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_examples(examples: list) -> str:
    return "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in examples
    )


def build_zero_shot_cot(sentence: str) -> str:
    """Zero-shot con chain-of-thought estructurado."""
    return f"""{PROMPT_BASE}

{COT_INSTRUCTIONS}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_zero_shot_glossary(sentence: str) -> str:
    """Zero-shot con glosario inline de términos frecuentes."""
    return f"""{PROMPT_BASE}

{LSM_GLOSSARY}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_zero_shot_full(sentence: str) -> str:
    """Zero-shot con CoT + glosario + negativos (combinación máxima)."""
    return f"""{PROMPT_BASE}

{LSM_GLOSSARY}

{_format_negative_examples()}

{COT_INSTRUCTIONS}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot_cot(sentence: str, k: int = 10) -> str:
    """Few-shot + chain-of-thought."""
    examples = FIXED_EXAMPLES[:k]
    return f"""{PROMPT_BASE}

{COT_INSTRUCTIONS}

EJEMPLOS:
{_format_examples(examples)}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot_negative(sentence: str, k: int = 10) -> str:
    """Few-shot + ejemplos negativos al final para reforzar errores a evitar."""
    examples = FIXED_EXAMPLES[:k]
    return f"""{PROMPT_BASE}

EJEMPLOS CORRECTOS:
{_format_examples(examples)}

{_format_negative_examples()}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot_curriculum(sentence: str, k: int = 10) -> str:
    """Few-shot con ejemplos ordenados por dificultad creciente (proxy: longitud)."""
    examples = sorted(FIXED_EXAMPLES[:k], key=lambda e: len(e["spa"]))
    return f"""{PROMPT_BASE}

EJEMPLOS (de simple a complejo):
{_format_examples(examples)}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot_diverse(sentence: str, diverse_examples: list) -> str:
    """Few-shot con ejemplos seleccionados por diversidad (clustering sobre pool).

    La selección de diverse_examples se hace fuera (embedding_index).
    """
    return f"""{PROMPT_BASE}

EJEMPLOS (cobertura diversa de patrones LSM):
{_format_examples(diverse_examples)}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot_full(sentence: str, k: int = 10) -> str:
    """Few-shot + CoT + glosario + negativos (combinación máxima)."""
    examples = FIXED_EXAMPLES[:k]
    return f"""{PROMPT_BASE}

{LSM_GLOSSARY}

{COT_INSTRUCTIONS}

EJEMPLOS CORRECTOS:
{_format_examples(examples)}

{_format_negative_examples()}

Traduce:
SPA: "{sentence}"
MSLG:"""
