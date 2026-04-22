"""Construcción de prompts para Enfoque 7 (Anthropic API).

Versión autónoma — NO importa desde enfoque6. Reglas, ejemplos y glosario
alineados con el corpus MSLG_SPA_train.txt (no con reglas académicas genéricas
de Cruz Aldrete 2008, que divergen en convenciones ortográficas).

Cada builder retorna (system, user):
  - system: contenido estático, cacheable por la API de Anthropic.
  - user:   oración variable a traducir.
"""


# ── PROMPT_BASE alineado al corpus ────────────────────────────────────────────

PROMPT_BASE = """\
Eres un traductor experto de español a glosas de Lengua de Señas Mexicana (LSM/MSLG).
Tu tarea es traducir la oración en español a su equivalente en glosas LSM.
Responde ÚNICAMENTE con la glosa traducida, sin explicaciones ni texto adicional.

═══ REGLAS GRAMATICALES (alineadas al corpus MSLG-SPA 2026) ═══

ORDEN DE CONSTITUYENTES:
- Orden básico: Sujeto-Verbo-Objeto (SVO).
- Con objeto focalizado o verbos demostrativos: Sujeto-Objeto-Verbo (SOV).
- Topicalización frecuente: lo más informativo va AL INICIO (objeto, locativo o tiempo).
- Marcadores de TIEMPO al INICIO: AYER, HOY, MAÑANA, ANTES, AHORA, ANTIER, ANOCHE,
  HACE-TIEMPO, HACE-PASADO, SEMANA PASADA, AÑO-PASADO, PRÓXIMO, ÚLTIMA.
- Marcadores de FRECUENCIA/MODO al INICIO: SIEMPRE, DIARIO, A-VECES, NUNCA,
  TODOS-SÁBADOS, TODO-SÁBADO, DOMINGOS DOMINGOS.
- Topicalización locativa: el lugar va al inicio (PLAYA TORTUGA HABER;
  CHAPULTEPEC HABER LAGO; CINE MI AMIGO ÉL BESAR MÍ).
- Adjetivos van DESPUÉS del sustantivo (PEZ AZUL, PIEDRA BONITA, CARRO LUJOSO).
- COLOR pospuesto: "puerta azul" → PUERTA COLOR AZUL; "zapato amarillo" → ZAPATO COLOR AMARILLO.
- Edades pospuestas: "90 años" → 90 EDAD.
- Patrón tópico-comentario sin cópula: "Tepito es famoso" → TEPITO LUGAR FAMOSO;
  "vivir en EE.UU. es difícil" → ESTADOS UNIDOS DIFÍCIL VIVIR.

ASPECTO Y TIEMPO:
- YA marca aspecto perfectivo/completado; va ANTES del verbo principal (masivamente
  frecuente en el corpus): YO YA GANAR, ÉL YA COMPRAR, YO YA PERDER.
  También puede ir al FINAL: EMBARAZADA YA, ABANDONAR YA.
- FUTURO como partícula pre-atributo o pre-verbo: "Edgar será maestro" →
  dm-EDGAR FUTURO MAESTRO.
- PRÓXIMO + marcador temporal: PRÓXIMO AÑO, PRÓXIMO-DOMINGO, PRÓXIMO VACACIONES.

ELIMINACIONES:
- Eliminar artículos: el, la, los, las, un, una, unos, unas.
- Eliminar preposiciones: de, en, a, por, para — SALVO dentro de compuestos o
  en finalidad explícita (PARA AGUA ABSORBER).
- Eliminar verbos cópula (ser, estar, parecer): usar yuxtaposición.
- CONSERVAR posesivos: MI, TU, SU, NUESTRO, MÍO, SUYO/SUYA, TUYO. El corpus los
  conserva de forma sistemática (dm-ISABEL TENER SU CORONA ORO).

TRANSFORMACIONES OBLIGATORIAS:
- Verbos: SIEMPRE infinitivo en MAYÚSCULAS (llegó→LLEGAR, cenaste→CENAR, compré→COMPRAR).
- Antropónimos: prefijo dm- + MAYÚSCULAS (Isabel→dm-ISABEL, Juan Gabriel→dm-JUAN-GABRIEL,
  Diego Rivera→dm-DIEGO-RIVERA). NO aplica a organizaciones ni lugares.
- Organizaciones/siglas/medios con prefijo #: #TV, #SEP, #LSFB, #VGT.
- Femenino humano con +MUJER: hermana→HERMANO+MUJER, maestra→MAESTRO+MUJER,
  tía→TÍO+MUJER, amiga→AMIGO+MUJER, hija→HIJO+MUJER, abuela→ABUELO+MUJER,
  novia→NOVIO+MUJER, cantante (f)→CANTAR+MUJER, vecina→VECINO+MUJER.
  NO aplica a objetos/conceptos (mesa, casa).
- Intensificadores (muy, demasiado, bastante): posposición MUCHO tras el adjetivo,
  o VERDAD al final para "muy" + adjetivo evaluativo (ESA ANTENA CARA VERDAD).
- Compuestos con guión: CINTURÓN-DE-SEGURIDAD, LAGO-DE-PÁTZCUARO, TARJETA-DE-CRÉDITO,
  LICENCIA-DE-CONDUCIR, CARLOS-SALINAS-DE-GORTARI.
- Locuciones y frases idiomáticas con guión: NO-TE-HE-VISTO, NOS-VEMOS, NO-HAY-NADIE,
  NO-FUI, NO-SIRVE, NO-ATREVERSE, TENER-CULPA, DAR-EL-AVIÓN, TERMINAR-COMO-AMIGOS,
  ENTREGAR-ENGANCHE, A-FUERZA, A-SALVO, A-VECES, DEJARLO-ASÍ, ME-LA-VAS-A-PAGAR.
- Duales con guión: DOS-DE-NOSOTROS, NOSOTROS-DE-DOS, USTEDES-DEDOS.
- AMBOS al final del sujeto compuesto coordinado: "María y Daniel van al teatro" →
  dm-MARÍA dm-DANIEL AMBOS IR TEATRO.
- Todo en MAYÚSCULAS, excepto el prefijo dm- y sufijos/clasificadores incorporados
  en minúscula (AZOTEAarriba, HOSPITALacá, ENCENDERfuego, SEXTO-Primaria).

REDUPLICACIÓN (mecanismo productivo):
- Plural: NIÑO NIÑO = "los niños", HERMANO HERMANO, DIENTE DIENTE.
- Intensidad/estado: TRABAJO TRABAJO = "trabajador", EXPRESAR EXPRESAR = "expresivo",
  COLOCAR COLOCAR = "colocó mucha/por todos lados".
- Habitualidad temporal: TODO-SÁBADO TODO-SÁBADO, DOMINGOS DOMINGOS.
- Énfasis predicativo con sujeto duplicado al final: YO CLAVADO HÁBIL YO;
  ÉL BUENO ALUMNO Él; TÚ HURAÑO TÚ.

PRONOMBRES:
- Personales sujeto: YO, TÚ, ÉL, ELLA, NOSOTROS, USTEDES, ELLOS.
- Pronombres objeto (distinción ortográfica crítica):
    MÍ (con tilde) = pronombre objeto ("ella me besó" → ELLA BESAR MÍ).
    MI (sin tilde) = posesivo ("mi libro" → MI LIBRO).
    TI = pronombre objeto ("te aviso" → YO AVISAR TI).
- Clítico ÉL/ELLA pospuesto al verbo marca concordancia de objeto:
  "conocí al sacerdote" → HOY NUEVO SACERDOTE YO YA CONOCER ÉL;
  "admiran a Madonna" → CANTAR+MUJER dm-MADONNA ELLOS ADMIRAR ELLA.
- SUYO/SUYA pospuesto como alternativa a SU antepuesto:
  dm-ANA SUYO PERRO ABANDONAR YA; MI HERMANO SUYA NOVIO+MUJER.

NEGACIÓN:
- Forma general: NO antes o después del verbo.
- Negación implícita por contexto (frecuente): a veces NO se omite y la negación
  queda inferida ("no me dio pastel" → AYER MI HERMANO PASTEL ÉL DAR MÍ).
- Verbos con forma negativa irregular (prefieren estas formas):
    no poder → NO-PODER | no haber/no hay → NO-HAY o NO-HABER
    no saber → NO-SABER | no querer → NO-QUERER | no hacer → NO-HACER
    no gustar → NO-GUSTAR | no conocer → NO-CONOCER | no ver → NO-VER
    no servir → NO-SIRVE | no venir → NO-VENIR | no atreverse → NO-ATREVERSE
    no ser yo → NO-FUI | no hay nadie → NO-HAY-NADIE
    no te he visto → NO-TE-HE-VISTO
- Negación existencial: NO-HAY, NADA, o NINGÚN.

INTERROGACIÓN:
- Conservar ¿...?.
- QUIÉN: al INICIO.
- QUÉ: al inicio o antes del sustantivo.
- CÓMO: antes del verbo (puede duplicarse inicio + final).
- CUÁNDO: inicio o antes del verbo, duplicable.
- DÓNDE: inicio o antes del verbo, frecuente duplicación (¿DÓNDE BALÓN DÓNDE?).
- CUÁNTO: al FINAL.
- CUÁL: al FINAL (¿TÚ GRUPO NÚMERO CUÁL?).
- PORQUÉ (escritura JUNTA, sin guión): al FINAL.
  Ejemplo corpus: SENADOR GASOLINA AUMENTAR PORQUÉ ÉL EXPLICAR.
  NO usar "POR-QUÉ" con guión: el corpus usa PORQUÉ junto.
- PARA-QUÉ: preverbal, duplicable.

COORDINACIÓN Y SUBORDINACIÓN:
- Copulativa: Y o yuxtaposición. Sujeto compuesto cierra con AMBOS.
- Adversativa: PERO entre oraciones.
- Condicional: seña IMAGINAR al inicio (opcional; muchas veces por yuxtaposición).
- Concesiva: NI-MODO o AUNQUE.
- Desiderativa: OJALÁ al inicio ("Ojalá tengas éxito" → OJALÁ TÚ ÉXITO).
- Dubitativa: QUIZÁ al inicio.
- Subordinación general: por yuxtaposición.\
"""


# ── FIXED_EXAMPLES — todos literales del train (corpus-faithful) ─────────────

FIXED_EXAMPLES = [
    # Afirmativa simple con SU conservado
    {"spa": "Isabel tiene una corona de oro.",
     "mslg": "dm-ISABEL TENER SU CORONA ORO"},
    # YA aspectual + femenino +MUJER
    {"spa": "Mi hermana está embarazada.",
     "mslg": "MI HERMANO+MUJER YA EMBARAZADA"},
    # Interrogativa YA + TÚ
    {"spa": "¿Ya cenaste?",
     "mslg": "¿TÚ YA CENAR?"},
    # Tiempo + compuesto + YA
    {"spa": "Ayer llegó mi tío de San Francisco.",
     "mslg": "AYER MI TIO SAN FRANCISCO YA LLEGAR"},
    # Intensificador VERDAD final
    {"spa": "Esa antena es muy cara.",
     "mslg": "ESA ANTENA CARA VERDAD"},
    # Locución A-FUERZA + MÍ objeto
    {"spa": "Debes pagarme a fuerza.",
     "mslg": "A-FUERZA TÚ DEBER TÚ PAGAR MÍ"},
    # Doble +MUJER
    {"spa": "La hija de mi vecina tiene autismo.",
     "mslg": "MI VECINO+MUJER SU HIJO+MUJER TENER AUTISMO"},
    # dm- con nombre compuesto
    {"spa": "Visité el mural de Diego Rivera.",
     "mslg": "YO YA VISITAR MURAL dm-DIEGO-RIVERA"},
    # Infinitivo sin cópula
    {"spa": "Olvidé mi contraseña.",
     "mslg": "YO OLVIDAR CONTRASEÑA"},
    # Coordinación con tiempo al inicio
    {"spa": "El cumpleaños de mi tía es en abril y le haremos fiesta.",
     "mslg": "ABRIL CUMPLEAÑOS MI TÍO+MUJER NOSOTROS HACER FIESTA"},
    # Negación con NINGÚN
    {"spa": "No traigo efectivo.",
     "mslg": "EFECTIVO NINGÚN YO TRAER"},
    # Compuesto + NO-HAY topicalizado
    {"spa": "Mi auto no tiene cinturón de seguridad.",
     "mslg": "CINTURÓN-DE-SEGURIDAD MI AUTO NO-HAY"},
    # Frecuencia al inicio + SUYO
    {"spa": "Mi amiga y su novio van los sábados al casino.",
     "mslg": "TODOS-SÁBADOS MI AMIGO+MUJER SUYO NOVIO CASINO IR"},
    # Reduplicación de estado (TRABAJO TRABAJO)
    {"spa": "Mi suegro es muy trabajador.",
     "mslg": "MI SUEGRO TRABAJO TRABAJO"},
    # Orden OBJ+V con MÍ como pronombre objeto
    {"spa": "Hoy es mi cumpleaños.",
     "mslg": "HOY CUMPLEAÑOS MÍ"},
    # Locución NO-TE-HE-VISTO + dual
    {"spa": "No te he visto desde la última navidad.",
     "mslg": "NAVIDAD ÚLTIMA DOS-DE-NOSOTROS NO-TE-HE-VISTO"},
    # FUTURO partícula
    {"spa": "Edgar será maestro.",
     "mslg": "dm-EDGAR FUTURO MAESTRO"},
    # CUÁL al final
    {"spa": "¿Cuál es el número de tu grupo?",
     "mslg": "¿TÚ GRUPO NÚMERO CUÁL?"},
    # PORQUÉ junto al final
    {"spa": "El senador explica por qué la gasolina está cara.",
     "mslg": "SENADOR GASOLINA AUMENTAR PORQUÉ ÉL EXPLICAR"},
    # COLOR pospuesto + tópico
    {"spa": "Pinté mi puerta de color azul.",
     "mslg": "MI PUERTA COLOR AZUL YO YA PINTAR"},
    # AMBOS al final del sujeto
    {"spa": "María y Daniel van al teatro.",
     "mslg": "dm-MARÍA dm-DANIEL AMBOS IR TEATRO"},
    # # prefijo siglas
    {"spa": "En la TV hay mucha publicidad.",
     "mslg": "#TV PUBLICIDAD HABER MUCHO"},
    # Frecuencia SIEMPRE + reduplicación
    {"spa": "El bebé siempre es muy expresivo.",
     "mslg": "SIEMPRE BEBÉ EXPRESAR EXPRESAR"},
    # Desiderativa OJALÁ al inicio
    {"spa": "Ojalá tengas éxito.",
     "mslg": "OJALÁ TÚ ÉXITO"},
    # Edad pospuesta
    {"spa": "Mi abuela tiene 90 años.",
     "mslg": "MÍ ABUELO+MUJER 90 EDAD"},
]


# ── LSM_GLOSSARY corregido ────────────────────────────────────────────────────

LSM_GLOSSARY = """\
GLOSARIO DE REFERENCIA (SPA → LSM):
Pronombres sujeto: yo→YO, tú→TÚ, él→ÉL, ella→ELLA, nosotros→NOSOTROS, ustedes→USTEDES, ellos→ELLOS.
Pronombres OBJETO (con tilde): me→MÍ, te→TI. Clítico objeto posverbal: él→ÉL, ella→ELLA.
Posesivos (SIN tilde, conservar): mi→MI, tu→TU, su→SU, nuestro→NUESTRO. Pospuestos: mío→MÍO, tuyo→TUYO, suyo/suya→SUYO/SUYA.
Tiempo (al INICIO): ayer→AYER, hoy→HOY, mañana→MAÑANA, antes→ANTES, después→DESPUÉS, ahora→AHORA, antier→ANTIER, anoche→ANOCHE.
Frecuencia (al INICIO): siempre→SIEMPRE, diario→DIARIO, a veces→A-VECES, nunca→NUNCA, los sábados→TODOS-SÁBADOS.
Aspecto: ya/completado→YA (antes del verbo), será→FUTURO (pre-atributo), próximo→PRÓXIMO.
Interrogativas: qué→QUÉ, quién→QUIÉN, dónde→DÓNDE, cuándo→CUÁNDO, cómo→CÓMO.
Interrogativas al FINAL: cuánto→CUÁNTO, cuál→CUÁL, por qué→PORQUÉ (JUNTO, sin guión).
Cantidad: mucho→MUCHO, poco→POCO, todo→TODO, nada→NADA, algo→ALGO, ningún→NINGÚN.
Colores: "X color Y"→X COLOR Y (color pospuesto).
Siglas/medios con #: TV→#TV, SEP→#SEP, LSFB→#LSFB, VGT→#VGT.
Verbos frecuentes: ir→IR, venir→VENIR, tener→TENER, querer→QUERER, poder→PODER, saber→SABER, hacer→HACER, ver→VER, comer→COMER, trabajar→TRABAJAR, estudiar→ESTUDIAR, vivir→VIVIR, dar→DAR.
Negativos irregulares: no poder→NO-PODER, no haber/no hay→NO-HAY/NO-HABER, no saber→NO-SABER, no querer→NO-QUERER, no hacer→NO-HACER, no gustar→NO-GUSTAR, no conocer→NO-CONOCER, no ver→NO-VER, no servir→NO-SIRVE, no venir→NO-VENIR, no atreverse→NO-ATREVERSE, no fui (yo)→NO-FUI, no hay nadie→NO-HAY-NADIE, no te he visto→NO-TE-HE-VISTO.
Duales: ambos/los dos→DOS-DE-NOSOTROS o NOSOTROS-DE-DOS (dual); coordinación con AMBOS al final.
Locuciones con guión: a fuerza→A-FUERZA, a salvo→A-SALVO, tener culpa→TENER-CULPA, dar el avión→DAR-EL-AVIÓN, nos vemos→NOS-VEMOS, terminar como amigos→TERMINAR-COMO-AMIGOS.
Familia (masc + MUJER para fem): hermana→HERMANO+MUJER, tía→TÍO+MUJER, hija→HIJO+MUJER, amiga→AMIGO+MUJER, abuela→ABUELO+MUJER, madre→MAMÁ, padre→PAPÁ.
Edad: "X años"→X EDAD (pospuesto).\
"""


# ── NEGATIVE_EXAMPLES ampliados con errores reales ────────────────────────────

NEGATIVE_EXAMPLES = [
    {"spa":  "Mi madre cocina arroz.",
     "mal":  "LA MADRE COCINA EL ARROZ",
     "bien": "MI MAMÁ ARROZ COCINAR",
     "error": "no eliminó artículos y no convirtió verbo a infinitivo"},
    {"spa":  "Ayer fui al mercado.",
     "mal":  "YO IR MERCADO AYER",
     "bien": "AYER YO MERCADO IR",
     "error": "marcador temporal no va al INICIO"},
    {"spa":  "Mi hermana es alta.",
     "mal":  "MI HERMANA SER ALTA",
     "bien": "MI HERMANO+MUJER ALTA",
     "error": "no aplicó +MUJER y conservó cópula SER"},
    {"spa":  "¿Por qué llegaste tarde?",
     "mal":  "TARDE TÚ LLEGAR POR-QUÉ",
     "bien": "TARDE ¿POR QUÉ?",
     "error": "usó POR-QUÉ con guión; el corpus usa PORQUÉ junto o POR QUÉ separado (NO con guión)"},
    {"spa":  "Yo ya gané el campeonato.",
     "mal":  "YO GANAR CAMPEONATO",
     "bien": "YO YA GANAR CAMPEONATO",
     "error": "omitió YA (aspecto perfectivo) antes del verbo"},
    {"spa":  "Los niños tienen piojos.",
     "mal":  "NIÑOS TENER PIOJOS",
     "bien": "NIÑO NIÑO TENER PIOJO",
     "error": "no aplicó reduplicación para marcar plural"},
    {"spa":  "Isabel tiene una corona de oro.",
     "mal":  "dm-ISABEL TENER CORONA ORO",
     "bien": "dm-ISABEL TENER SU CORONA ORO",
     "error": "eliminó el posesivo SU; el corpus lo conserva sistemáticamente"},
    {"spa":  "Edgar será maestro.",
     "mal":  "dm-EDGAR SERÁ MAESTRO",
     "bien": "dm-EDGAR FUTURO MAESTRO",
     "error": "conservó verbo SER conjugado; debe usar partícula FUTURO"},
]


# ── COT_INSTRUCTIONS ajustadas ────────────────────────────────────────────────

COT_INSTRUCTIONS = """\
RAZONA ANTES DE RESPONDER (no muestres los pasos, solo la glosa final):
  1. Identifica: tiempo, frecuencia, sujeto, objeto, verbo, negación, interrogativa, aspecto.
  2. Elimina: artículos, preposiciones (salvo en compuestos/finalidad), cópulas.
     CONSERVA posesivos (MI, TU, SU, SUYO, TUYO).
  3. Transforma: verbos a infinitivo MAYÚSCULAS; femeninos humanos a MASCULINO+MUJER;
     antropónimos con dm-; siglas con #; compuestos y locuciones con guión.
  4. Aspecto: si la acción está completada añade YA antes del verbo.
     Si es futuro añade FUTURO antes del atributo/verbo.
  5. Plural/intensidad: considera reduplicación (NIÑO NIÑO, TRABAJO TRABAJO).
  6. Reordena: tiempo/frecuencia/locativo al INICIO; interrogativas según regla
     (CUÁNTO/CUÁL/PORQUÉ al final; QUIÉN al inicio). Usa ¿...?.
  7. Pronombres: MÍ (objeto, con tilde) vs MI (posesivo, sin tilde). Clítico
     ÉL/ELLA posverbal para concordancia de objeto.
  8. Emite SOLO la glosa final en una línea, sin explicar.\
"""


# ── Helpers internos ──────────────────────────────────────────────────────────

def _format_negative_examples() -> str:
    lines = ["EJEMPLOS DE ERRORES FRECUENTES (NO los repitas):"]
    for ex in NEGATIVE_EXAMPLES:
        lines.append(f'  SPA:  "{ex["spa"]}"')
        lines.append(f'  MAL:  "{ex["mal"]}"   ← {ex["error"]}')
        lines.append(f'  BIEN: "{ex["bien"]}"')
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_examples(examples) -> str:
    return "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in examples
    )


def _user(sentence: str) -> str:
    return f'Traduce:\nSPA: "{sentence}"\nMSLG:'


def _compose_system(*blocks: str) -> str:
    return "\n\n".join(b for b in blocks if b)


def _examples_block(examples, header="EJEMPLOS:"):
    return f"{header}\n{_format_examples(examples)}"


# ── Zero-shot ─────────────────────────────────────────────────────────────────

def build_zero_shot(sentence: str):
    return PROMPT_BASE, _user(sentence)


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


def build_few_shot_rag(sentence: str, retrieved_examples: list):
    """Few-shot con ejemplos top-k recuperados dinámicamente del pool real.

    retrieved_examples viene de EmbeddingIndex.retrieve(sentence, k).
    NOTA: el system aquí varía por oración → NO se cachea eficientemente
    (cache_control se ignora a nivel de ejemplos). Trade-off aceptable:
    los ejemplos del corpus real dominan sobre reglas escritas.
    """
    system = _compose_system(
        PROMPT_BASE,
        _examples_block(retrieved_examples,
                        header="EJEMPLOS (similares a la oración a traducir):"),
    )
    return system, _user(sentence)
