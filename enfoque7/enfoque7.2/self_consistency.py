"""Self-Consistency: genera N candidatos con temperature>0 y elige el centroide
por chrF pairwise (candidato más cercano al consenso del grupo).

Ref: Wang et al. 2022 — "Self-Consistency Improves Chain of Thought Reasoning
in Language Models" (arXiv 2203.11171).
"""

import anthropic_client
import config
import post_processor

try:
    from sacrebleu.metrics import CHRF
    _chrf = CHRF()
except ImportError as exc:
    raise ImportError(
        "Falta la dependencia 'sacrebleu'. Instala con: uv add sacrebleu"
    ) from exc


def translate_with_sc(system_prompt: str, user_prompt: str, n: int = 3,
                      temperature: float = None) -> tuple:
    """Genera N candidatos, limpia cada uno y devuelve (mejor_crudo, mejor_limpio).

    Si n=1 → una sola llamada (sin SC), equivalente al flujo normal.
    Si n>=2 → centroid picking por chrF sobre los candidatos limpios.
    """
    if temperature is None:
        temperature = config.SC_TEMPERATURE

    raws = []
    cleans = []
    for _ in range(n):
        raw = anthropic_client.translate(system_prompt, user_prompt, temperature=temperature)
        raws.append(raw)
        cleans.append(post_processor.clean(raw))

    if n == 1:
        return raws[0], cleans[0]

    # Centroid picking: candidato con mayor chrF promedio contra los otros.
    best_idx = _centroid_pick(cleans)
    return raws[best_idx], cleans[best_idx]


def _centroid_pick(candidates: list) -> int:
    """Retorna el índice del candidato con mayor similitud promedio al resto."""
    n = len(candidates)
    if n <= 1:
        return 0
    scores = []
    for i, cand in enumerate(candidates):
        pair = []
        for j, other in enumerate(candidates):
            if i == j:
                continue
            # chrF.sentence_score(hypothesis, [reference])
            pair.append(_chrf.sentence_score(cand, [other]).score)
        scores.append(sum(pair) / len(pair) if pair else 0.0)
    return scores.index(max(scores))
