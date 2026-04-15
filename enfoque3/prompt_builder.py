"""Construcción de prompts para los diferentes niveles de experimento."""

PROMPT_BASE = """\
Eres un traductor de español a glosas de Lengua de Señas Mexicana (LSM).

Tu tarea es traducir la oración en español a su equivalente en glosas LSM.
Responde ÚNICAMENTE con la glosa traducida, sin explicaciones ni texto adicional.

REGLAS DE TRADUCCIÓN:
- Eliminar todos los artículos (el, la, los, las, un, una, unos, unas).
- Los verbos NO se conjugan: usar siempre la forma infinitiva (llegó → LLEGAR, cenaste → CENAR).
- Nombres propios llevan prefijo dm-: Isabel → dm-ISABEL, Diego Rivera → dm-DIEGO-RIVERA.
- Para femenino se agrega +MUJER al sustantivo masculino: hermana → HERMANO+MUJER, hija → HIJO+MUJER, tía → TÍO+MUJER, amiga → AMIGO+MUJER, vecina → VECINO+MUJER, maestra → MAESTRO+MUJER.
- Conceptos compuestos se unen con guión: licencia de conducir → LICENCIA-DE-CONDUCIR.
- Marcadores de tiempo van AL INICIO: "Ayer llegó mi tío" → AYER MI TIO ... LLEGAR.
- "muy" se traduce como VERDAD al final o MUCHO: "muy cara" → CARA VERDAD, "muy alto" → ALTO.
- Las glosas van SIEMPRE en MAYÚSCULAS.
- Las preguntas conservan ¿...?
- No usar preposiciones (de, en, a, por, para) a menos que sean parte de un concepto compuesto.
- Posesivos (mi, tu, su) SÍ se conservan: MI, TU, SU.
- Pronombres personales se conservan: YO, TÚ, ÉL, ELLA, NOSOTROS, USTEDES, ELLOS, ELLAS.
- Las oraciones en LSM suelen ser MÁS CORTAS que en español."""

# 15 ejemplos fijos seleccionados manualmente para cubrir las convenciones principales.
# Orden: simple, dm-, +MUJER, pregunta, temporal, VERDAD, compuesto con guión,
#         +MUJER doble, dm- compuesto, negación, +MUJER+temporal, negación2,
#         pregunta+compuesto, temporal+compuesto, posesivo+simple.
FIXED_EXAMPLES = [
    # 1. Simple: eliminación de artículos, plural→singular
    {"spa": "Los peces azules son mis favoritos.",
     "mslg": "PEZ COLOR AZUL MI FAVORITO"},
    # 2. dm- para nombre propio, verbo infinitivo
    {"spa": "Isabel tiene una corona de oro.",
     "mslg": "dm-ISABEL TENER SU CORONA ORO"},
    # 3. +MUJER para femenino
    {"spa": "Mi hermana está embarazada.",
     "mslg": "MI HERMANO+MUJER YA EMBARAZADA"},
    # 4. Pregunta con ¿...?, verbo infinitivo
    {"spa": "¿Ya cenaste?",
     "mslg": "¿TÚ YA CENAR?"},
    # 5. Marcador temporal al inicio
    {"spa": "Ayer llegó mi tío de San Francisco.",
     "mslg": "AYER MI TIO SAN FRANCISCO YA LLEGAR"},
    # 6. VERDAD como intensificador de "muy"
    {"spa": "Esa antena es muy cara.",
     "mslg": "ESA ANTENA CARA VERDAD"},
    # 7. Concepto compuesto con guión
    {"spa": "Debes pagarme a fuerza.",
     "mslg": "A-FUERZA TÚ DEBER TÚ PAGAR MÍ"},
    # 8. +MUJER doble (vecina, hija)
    {"spa": "La hija de mi vecina tiene autismo.",
     "mslg": "MI VECINO+MUJER SU HIJO+MUJER TENER AUTISMO"},
    # 9. dm- con nombre compuesto
    {"spa": "Visité el mural de Diego Rivera.",
     "mslg": "YO YA VISITAR MURAL dm-DIEGO-RIVERA"},
    # 10. Oración simple corta, verbo infinitivo
    {"spa": "Olvidé mi contraseña.",
     "mslg": "YO OLVIDAR CONTRASEÑA"},
    # 11. +MUJER (tía), temporal, fiesta
    {"spa": "El cumpleaños de mi tía es en abril y le haremos fiesta.",
     "mslg": "ABRIL CUMPLEAÑOS MI TÍO+MUJER NOSOTROS HACER FIESTA"},
    # 12. Negación
    {"spa": "No traigo efectivo.",
     "mslg": "EFECTIVO NINGÚN YO TRAER"},
    # 13. Cinturón de seguridad (compuesto con guión), negación
    {"spa": "Mi auto no tiene cinturón de seguridad.",
     "mslg": "CINTURÓN-DE-SEGURIDAD MI AUTO NO-HAY"},
    # 14. Mi amiga (+MUJER), temporal con TODOS-SÁBADOS
    {"spa": "Mi amiga y su novio van los sábados al casino.",
     "mslg": "TODOS-SÁBADOS MI AMIGO+MUJER SUYO NOVIO CASINO IR"},
    # 15. Mi tía cocina (+MUJER), oración corta
    {"spa": "Mi tía cocina delicioso.",
     "mslg": "MI TÍO+MUJER DELICIOSO COCINAR"},
]


def build_zero_shot(sentence):
    """Prompt zero-shot: solo reglas + oración."""
    return f"""{PROMPT_BASE}

Traduce:
SPA: "{sentence}"
MSLG:"""


def build_few_shot(sentence, k=5):
    """Prompt few-shot: reglas + K ejemplos fijos + oración."""
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


def build_rag(sentence, retrieved_examples):
    """Prompt RAG: reglas + K ejemplos recuperados dinámicamente + oración."""
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
