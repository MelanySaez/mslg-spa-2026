"""Construcción del prompt FOL-RAG.

Reusa PROMPT_BASE de enfoque3/prompt_builder.py y añade dos bloques:
  1. EJEMPLOS recuperados semánticamente por el EmbeddingIndex (KB).
  2. PISTA OPCIONAL: candidato generado por reglas FOL (al final, marcado
     como descartable si contradice los ejemplos).

El orden importa: ejemplos RAG van primero para que dominen el estilo y
el orden LSM; la pista FOL aparece al final con instrucción explícita de
ignorarla si entra en conflicto. Esto evita el sesgo observado en la
versión previa, donde el LLM copiaba el candidato incluso cuando era malo.
"""

import prompt_builder as _e3_pb  # enfoque3/prompt_builder.py (vía sys.path)

PROMPT_BASE = _e3_pb.PROMPT_BASE


def build_fol_rag(sentence, retrieved_examples, gloss_fol):
    ejemplos_text = "\n".join(
        f'SPA: "{ex["spa"]}" → MSLG: "{ex["mslg"]}"'
        for ex in retrieved_examples
    )
    return f"""{PROMPT_BASE}

EJEMPLOS (similares a la oración a traducir — son la referencia principal de orden y vocabulario LSM):
{ejemplos_text}

Traduce siguiendo EN PRIMER LUGAR el patrón de los EJEMPLOS anteriores.

SPA: "{sentence}"

PISTA OPCIONAL (candidato de reglas — úsalo SOLO si coincide con el estilo de los ejemplos, IGNÓRALO si contradice su orden o vocabulario): {gloss_fol}

MSLG:"""
