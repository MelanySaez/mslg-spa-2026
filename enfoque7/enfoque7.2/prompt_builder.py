"""Prompts para enfoque7.2 — dirección reversa MSLG → SPA.

Mismo esqueleto que enfoque7/prompt_builder.py pero con instrucciones, ejemplos
y formato user-message invertidos. La regla central: dada una glosa LSM,
producir español natural y fluido (no glosa, no explicaciones).
"""


# ── PROMPT_BASE inverso ──────────────────────────────────────────────────────

PROMPT_BASE = """\
Eres un traductor experto de glosas de Lengua de Señas Mexicana (LSM/MSLG) a español natural.
Tu tarea es traducir la glosa MSLG a una oración fluida en español.
Responde ÚNICAMENTE con la oración en español, sin explicaciones, sin la glosa, sin etiquetas.

═══ REGLAS DE EXPANSIÓN (glosa MSLG → español) ═══

ANTROPÓNIMOS Y SIGLAS:
- Quita el prefijo dm- y capitaliza el nombre propio: dm-ISABEL → "Isabel",
  dm-DIEGO-RIVERA → "Diego Rivera", dm-JUAN-GABRIEL → "Juan Gabriel".
- Quita el prefijo # de siglas/medios y deja en mayúsculas: #TV → "TV",
  #SEP → "SEP", #LSFB → "LSFB".

GÉNERO Y NÚMERO:
- Sufijo +MUJER marca femenino humano: HERMANO+MUJER → "hermana",
  TÍO+MUJER → "tía", AMIGO+MUJER → "amiga", HIJO+MUJER → "hija",
  ABUELO+MUJER → "abuela", VECINO+MUJER → "vecina", MAESTRO+MUJER → "maestra".
- Reduplicación marca plural o intensidad: NIÑO NIÑO → "los niños",
  HERMANO HERMANO → "los hermanos", TRABAJO TRABAJO → "muy trabajador",
  EXPRESAR EXPRESAR → "expresivo", DOMINGOS DOMINGOS → "todos los domingos".

ASPECTO Y TIEMPO:
- YA antes (o después) del verbo marca pasado completado: YO YA GANAR →
  "yo ya gané", ABANDONAR YA → "ya abandonó", dm-ANA YA LLEGAR → "Ana ya llegó".
- FUTURO antes del verbo o atributo marca tiempo futuro: dm-EDGAR FUTURO MAESTRO
  → "Edgar será maestro", YO FUTURO IR → "iré".
- Sin marca explícita de tiempo: usa presente o el tiempo más natural según
  el contexto de la oración.

ARTÍCULOS Y CÓPULAS (REINTRODUCCIÓN):
- Reintroduce artículos definidos/indefinidos según el contexto: NIÑO → "el niño",
  PIEDRA → "la piedra".
- Reintroduce verbos cópula (ser/estar) en yuxtaposiciones nominales:
  TEPITO LUGAR FAMOSO → "Tepito es un lugar famoso";
  ESA ANTENA CARA VERDAD → "Esa antena es muy cara";
  MI ABUELO ALTO → "Mi abuelo es alto".
- Conserva posesivos: MI → "mi", TU → "tu", SU → "su", SUYO/SUYA → "suyo/suya".

ORDEN DE PALABRAS:
- Pasa del orden MSLG (frecuente SOV o tópico-comentario) al SVO natural en
  español: MI MAMÁ ARROZ COCINAR → "Mi mamá cocina arroz".
- Marcadores de tiempo/frecuencia que en MSLG van al INICIO mantienen su lugar
  natural en español: AYER YO MERCADO IR → "Ayer fui al mercado".
- Topicalización locativa al inicio: PLAYA TORTUGA HABER → "En la playa hay
  tortugas"; CHAPULTEPEC HABER LAGO → "En Chapultepec hay un lago".
- Adjetivos pospuestos en MSLG vuelven a su orden natural: PEZ AZUL → "el pez
  azul"; PIEDRA BONITA → "una piedra bonita".
- Construcción "X COLOR Y" → "X de color Y" o adjetivo: PUERTA COLOR AZUL
  → "puerta azul" / "puerta de color azul".
- Edad pospuesta: 90 EDAD → "90 años", construir como "tiene 90 años".

PRONOMBRES (DISTINCIÓN ORTOGRÁFICA):
- MÍ (con tilde) en MSLG es pronombre OBJETO: ELLA BESAR MÍ → "Ella me besó";
  YO AVISAR TI → "Yo te aviso"; PAGAR MÍ → "pagarme".
- MI (sin tilde) es POSESIVO: MI LIBRO → "mi libro"; MI MAMÁ → "mi mamá".
- Clítico ÉL/ELLA pospuesto al verbo marca concordancia con objeto: úsalo
  para reconstruir el objeto pronominal o el sujeto si conviene.
- Coordinación de sujeto cierra con AMBOS: dm-MARÍA dm-DANIEL AMBOS IR TEATRO
  → "María y Daniel van al teatro".

INTERROGACIÓN:
- ¿...? se conserva. Reordena la palabra interrogativa al inicio en español:
  ¿TÚ GRUPO NÚMERO CUÁL? → "¿Cuál es el número de tu grupo?";
  ¿TÚ YA CENAR? → "¿Ya cenaste?";
  ¿DÓNDE BALÓN DÓNDE? → "¿Dónde está el balón?".
- PORQUÉ (escritura junta en MSLG) al final → "por qué" reordenado en español:
  SENADOR GASOLINA AUMENTAR PORQUÉ ÉL EXPLICAR → "El senador explica por qué
  aumentó la gasolina".

NEGACIÓN:
- Verbos negativos compuestos se expanden: NO-PODER → "no poder/no puede",
  NO-HAY → "no hay", NO-SABER → "no saber", NO-QUERER → "no querer",
  NO-GUSTAR → "no gustar", NO-TE-HE-VISTO → "no te he visto",
  NO-HAY-NADIE → "no hay nadie", NO-FUI → "no fui yo".
- NINGÚN, NADA, NO suelen requerir reordenamiento al lugar natural en español:
  EFECTIVO NINGÚN YO TRAER → "No traigo efectivo".

LOCUCIONES Y COMPUESTOS:
- Compuestos con guión se expanden a frase: CINTURÓN-DE-SEGURIDAD → "cinturón
  de seguridad", LAGO-DE-PÁTZCUARO → "lago de Pátzcuaro", TARJETA-DE-CRÉDITO
  → "tarjeta de crédito", LICENCIA-DE-CONDUCIR → "licencia de conducir".
- Locuciones idiomáticas: A-FUERZA → "a fuerza", NOS-VEMOS → "nos vemos",
  ME-LA-VAS-A-PAGAR → "me la vas a pagar", TENER-CULPA → "tener la culpa",
  DAR-EL-AVIÓN → "dar el avión".
- Duales: DOS-DE-NOSOTROS → "los dos / nosotros dos".

MARCADORES MODALES:
- OJALÁ al inicio: OJALÁ TÚ ÉXITO → "Ojalá tengas éxito".
- IMAGINAR al inicio: condicional ("si...").
- QUIZÁ al inicio: "Quizá...".

FORMATO DE SALIDA (CRÍTICO):
- Devuelve UNA SOLA línea con la oración en español, ya capitalizada.
- Termina con punto, signo de interrogación o exclamación según corresponda.
- NO incluyas la glosa, ni etiquetas como "Español:" o "Traducción:", ni
  comillas envolventes, ni explicaciones, ni paráfrasis alternativas.\
"""


# ── FIXED_EXAMPLES — los mismos pares que en SPA→MSLG, invertidos ───────────

FIXED_EXAMPLES = [
    {"mslg": "dm-ISABEL TENER SU CORONA ORO",
     "spa":  "Isabel tiene una corona de oro."},
    {"mslg": "MI HERMANO+MUJER YA EMBARAZADA",
     "spa":  "Mi hermana está embarazada."},
    {"mslg": "¿TÚ YA CENAR?",
     "spa":  "¿Ya cenaste?"},
    {"mslg": "AYER MI TIO SAN FRANCISCO YA LLEGAR",
     "spa":  "Ayer llegó mi tío de San Francisco."},
    {"mslg": "ESA ANTENA CARA VERDAD",
     "spa":  "Esa antena es muy cara."},
    {"mslg": "A-FUERZA TÚ DEBER TÚ PAGAR MÍ",
     "spa":  "Debes pagarme a fuerza."},
    {"mslg": "MI VECINO+MUJER SU HIJO+MUJER TENER AUTISMO",
     "spa":  "La hija de mi vecina tiene autismo."},
    {"mslg": "YO YA VISITAR MURAL dm-DIEGO-RIVERA",
     "spa":  "Visité el mural de Diego Rivera."},
    {"mslg": "YO OLVIDAR CONTRASEÑA",
     "spa":  "Olvidé mi contraseña."},
    {"mslg": "ABRIL CUMPLEAÑOS MI TÍO+MUJER NOSOTROS HACER FIESTA",
     "spa":  "El cumpleaños de mi tía es en abril y le haremos fiesta."},
    {"mslg": "EFECTIVO NINGÚN YO TRAER",
     "spa":  "No traigo efectivo."},
    {"mslg": "CINTURÓN-DE-SEGURIDAD MI AUTO NO-HAY",
     "spa":  "Mi auto no tiene cinturón de seguridad."},
    {"mslg": "TODOS-SÁBADOS MI AMIGO+MUJER SUYO NOVIO CASINO IR",
     "spa":  "Mi amiga y su novio van los sábados al casino."},
    {"mslg": "MI SUEGRO TRABAJO TRABAJO",
     "spa":  "Mi suegro es muy trabajador."},
    {"mslg": "HOY CUMPLEAÑOS MÍ",
     "spa":  "Hoy es mi cumpleaños."},
    {"mslg": "NAVIDAD ÚLTIMA DOS-DE-NOSOTROS NO-TE-HE-VISTO",
     "spa":  "No te he visto desde la última navidad."},
    {"mslg": "dm-EDGAR FUTURO MAESTRO",
     "spa":  "Edgar será maestro."},
    {"mslg": "¿TÚ GRUPO NÚMERO CUÁL?",
     "spa":  "¿Cuál es el número de tu grupo?"},
    {"mslg": "SENADOR GASOLINA AUMENTAR PORQUÉ ÉL EXPLICAR",
     "spa":  "El senador explica por qué la gasolina está cara."},
    {"mslg": "MI PUERTA COLOR AZUL YO YA PINTAR",
     "spa":  "Pinté mi puerta de color azul."},
    {"mslg": "dm-MARÍA dm-DANIEL AMBOS IR TEATRO",
     "spa":  "María y Daniel van al teatro."},
    {"mslg": "#TV PUBLICIDAD HABER MUCHO",
     "spa":  "En la TV hay mucha publicidad."},
    {"mslg": "SIEMPRE BEBÉ EXPRESAR EXPRESAR",
     "spa":  "El bebé siempre es muy expresivo."},
    {"mslg": "OJALÁ TÚ ÉXITO",
     "spa":  "Ojalá tengas éxito."},
    {"mslg": "MÍ ABUELO+MUJER 90 EDAD",
     "spa":  "Mi abuela tiene 90 años."},
]


# ── NEGATIVE_EXAMPLES — errores típicos en sentido reverso ──────────────────

NEGATIVE_EXAMPLES = [
    {"mslg": "dm-ISABEL TENER SU CORONA ORO",
     "mal":  "ISABEL TENER SU CORONA ORO",
     "bien": "Isabel tiene una corona de oro.",
     "error": "dejó las palabras en mayúsculas y no expandió la glosa a oración natural"},
    {"mslg": "MI HERMANO+MUJER YA EMBARAZADA",
     "mal":  "Mi hermano más mujer ya embarazada",
     "bien": "Mi hermana está embarazada.",
     "error": "no interpretó +MUJER como femenino y omitió la cópula 'está'"},
    {"mslg": "NIÑO NIÑO TENER PIOJO",
     "mal":  "Niño niño tienen piojo.",
     "bien": "Los niños tienen piojos.",
     "error": "no interpretó la reduplicación como plural"},
    {"mslg": "AYER YO MERCADO IR",
     "mal":  "Ayer yo mercado ir.",
     "bien": "Ayer fui al mercado.",
     "error": "no reordenó al SVO natural ni conjugó el verbo"},
    {"mslg": "dm-EDGAR FUTURO MAESTRO",
     "mal":  "Edgar futuro maestro.",
     "bien": "Edgar será maestro.",
     "error": "dejó FUTURO literal en lugar de conjugar el verbo en futuro"},
    {"mslg": "YO YA GANAR CAMPEONATO",
     "mal":  "Yo ya ganar el campeonato.",
     "bien": "Ya gané el campeonato.",
     "error": "no conjugó el infinitivo a pasado pese a la marca YA"},
    {"mslg": "ELLA BESAR MÍ",
     "mal":  "Ella besar mi.",
     "bien": "Ella me besó.",
     "error": "confundió MÍ (objeto) con MI (posesivo) y no conjugó el verbo"},
]


# ── Helpers internos ─────────────────────────────────────────────────────────

def _format_negative_examples() -> str:
    lines = ["EJEMPLOS DE ERRORES FRECUENTES (NO los repitas):"]
    for ex in NEGATIVE_EXAMPLES:
        lines.append(f'  MSLG: "{ex["mslg"]}"')
        lines.append(f'  MAL:  "{ex["mal"]}"   ← {ex["error"]}')
        lines.append(f'  BIEN: "{ex["bien"]}"')
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_examples(examples) -> str:
    return "\n".join(
        f'MSLG: "{ex["mslg"]}" → SPA: "{ex["spa"]}"'
        for ex in examples
    )


def _user(mslg_sentence: str) -> str:
    return f'Traduce:\nMSLG: "{mslg_sentence}"\nSPA:'


def _compose_system(*blocks: str) -> str:
    return "\n\n".join(b for b in blocks if b)


def _examples_block(examples, header="EJEMPLOS:"):
    return f"{header}\n{_format_examples(examples)}"


# ── Builders ─────────────────────────────────────────────────────────────────

def build_zero_shot(mslg_sentence: str):
    return PROMPT_BASE, _user(mslg_sentence)


def build_few_shot(mslg_sentence: str, k: int = 10):
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(FIXED_EXAMPLES[:k]),
    )
    return system, _user(mslg_sentence)


def build_few_shot_rag(mslg_sentence: str, retrieved_examples: list):
    """Few-shot con ejemplos top-k recuperados dinámicamente del pool real."""
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(retrieved_examples,
                        header="EJEMPLOS (similares a la glosa a traducir):"),
    )
    return system, _user(mslg_sentence)


def build_few_shot_rag_curriculum(mslg_sentence: str, retrieved_examples: list):
    """RAG + curriculum: top-k por embeddings + orden ascendente por longitud
    de la glosa MSLG (recency bias: el ejemplo más complejo queda PEGADO a la
    query, en la posición de mayor influencia en el contexto del LLM).
    """
    ordered = sorted(retrieved_examples, key=lambda e: len(e["mslg"]))
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(
            ordered,
            header="EJEMPLOS (recuperados y ordenados de simple a complejo):",
        ),
    )
    return system, _user(mslg_sentence)
