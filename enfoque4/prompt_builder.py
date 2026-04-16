"""Construcción del prompt FOL-RAG y RAG enriquecido.

Reusa PROMPT_BASE de enfoque3/prompt_builder.py y lo extiende con un
bloque condensado de sintaxis LSM (tomado de la gramática de Cruz
Aldrete, 2008 — ver reglas_sintaxis_lsm.md). El prompt enriquecido se
usa tanto por build_fol_rag como por build_rag_enriched (fallback
cuando el candidato FOL es degenerado).

Estructura de build_fol_rag:
  1. PROMPT_BASE_ENRICHED (reglas de enfoque3 + sintaxis LSM).
  2. EJEMPLOS recuperados por el índice híbrido (referencia principal).
  3. PISTA OPCIONAL: candidato FOL al final, descartable si contradice
     los ejemplos (evita el sesgo del prompt previo donde el LLM copiaba
     ciegamente el candidato).
"""

import prompt_builder as _e3_pb  # enfoque3/prompt_builder.py (vía sys.path)

PROMPT_BASE = _e3_pb.PROMPT_BASE

PROMPT_SYNTAX_LSM = """\

REGLAS ADICIONALES DE SINTAXIS LSM (gramática de Cruz Aldrete, 2008):
- Orden básico: Sujeto-Verbo-Objeto (SVO). Con pronombres 1ª/2ª persona como objeto directo, Sujeto-Objeto-Verbo (SOV): "él me robó" → ÍNDICE YO ROBAR.
- Topicalización: el elemento tópico (sujeto, objeto o circunstancial) va al INICIO; el resto sigue el orden SVO.
- Complementos temporales (AYER, MAÑANA, ANTES, FUTURO): posición INICIAL obligatoria.
- Complemento de MODO (rápido, despacio, bien, mal): posterior al verbo.
- Complemento de INSTRUMENTO: anterior al verbo.
- Cuantificador MUCHO/POCO: posterior al sustantivo ("muchos zapatos" → ZAPATO MUCHO).
- Adjetivos calificativos: posteriores al sustantivo ("perro negro" → PERRO NEGRO).
- Numerales: preferentemente prenominales ("tres perros" → TRES PERRO). Unidades de medida, posnominales ("tres metros" → TRES METRO).
- No existen cópulas (ser/estar): se eliminan. "Edgar es maestro" → dm-EDGAR MAESTRO.
- Predicados existenciales/locativos: la locación se antepone, con HABER / NO-HABER. "Hay libros en la mesa" → MESA LIBRO HABER.
- Preguntas qu-: la palabra interrogativa puede ir al inicio, anterior al verbo, o DUPLICARSE al final. Tendencias: CUÁNDO/DÓNDE/QUIÉN/QUÉ → inicio; CUÁNTO y POR-QUÉ → posición FINAL.
- Negación canónica: misma estructura afirmativa + seña NO (opcional, posterior al verbo).
- Verbos con forma negativa IRREGULAR (fusionan "no" + verbo en una sola seña):
    no poder → NO-PODER;  no haber → NO-HABER;  no saber → NO-SABER;
    no conocer → NO-CONOCER;  no gustar → NO-GUSTAR;  no querer → NO-QUERER;
    no hacer → NO-HACER;  no servir/usar → NO-SERVIR/NO-USAR;  no ver → NO-VER.
- Oraciones condicionales: anteponer IMAGINAR al inicio (opcional).
- Adversativas entre oraciones: usar PERO.
- La glosa final suele ser MÁS BREVE que el español; eliminar relleno."""

PROMPT_BASE_ENRICHED = PROMPT_BASE + PROMPT_SYNTAX_LSM


def _format_examples(retrieved_examples):
    return "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in retrieved_examples
    )


def build_fol_rag(sentence, retrieved_examples, gloss_fol):
    ejemplos_text = _format_examples(retrieved_examples)
    return f"""{PROMPT_BASE_ENRICHED}

EJEMPLOS (similares a la oración a traducir — son la referencia principal de orden y vocabulario LSM):
{ejemplos_text}

Traduce siguiendo EN PRIMER LUGAR el patrón de los EJEMPLOS anteriores.

SPA: "{sentence}"

PISTA OPCIONAL (candidato de reglas — úsalo SOLO si coincide con el estilo de los ejemplos, IGNÓRALO si contradice su orden o vocabulario): {gloss_fol}

MSLG:"""


def build_rag_enriched(sentence, retrieved_examples):
    """RAG puro usando el prompt enriquecido con sintaxis LSM.
    Se usa cuando el candidato FOL es degenerado.
    """
    ejemplos_text = _format_examples(retrieved_examples)
    return f"""{PROMPT_BASE_ENRICHED}

EJEMPLOS (similares a la oración a traducir):
{ejemplos_text}

Traduce:
SPA: "{sentence}"
MSLG:"""
