"""Construcción del prompt FOL-RAG.

Reusa PROMPT_BASE de enfoque3/prompt_builder.py y añade dos bloques:
  1. EJEMPLOS recuperados semánticamente por el EmbeddingIndex (KB).
  2. CANDIDATO AUTOMÁTICO: glosa generada por las reglas FOL de enfoque1.

El candidato se marca como falible para que el LLM no lo copie ciegamente
y se instruye explícitamente a respetar el orden LSM de los ejemplos.
"""

import prompt_builder as _e3_pb  # enfoque3/prompt_builder.py (vía sys.path)

PROMPT_BASE = _e3_pb.PROMPT_BASE


def build_fol_rag(sentence, retrieved_examples, gloss_fol):
    ejemplos_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in retrieved_examples
    )
    return f"""{PROMPT_BASE}

EJEMPLOS (similares a la oración a traducir):
{ejemplos_text}

CANDIDATO AUTOMÁTICO (generado por reglas lingüísticas; puede tener errores de orden o vocabulario — úsalo como referencia, corrígelo si es necesario):
{gloss_fol}

Traduce (devuelve SOLO la glosa final, respetando el orden LSM visto en los ejemplos):
SPA: "{sentence}"
MSLG:"""
