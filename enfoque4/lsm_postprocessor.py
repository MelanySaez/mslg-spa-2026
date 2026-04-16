"""Post-procesado específico LSM sobre la salida ya limpia del LLM.

Se ejecuta DESPUÉS de enfoque3/post_processor.clean() y aplica
correcciones determinísticas basadas en la gramática de Cruz Aldrete:

  1. dm- prefix case fix (DM-ISABEL → dm-ISABEL).
  2. Strip leaked prepositions (standalone DE/EN/A/POR/PARA/CON).
  3. Move temporal markers to front (AYER/HOY/MAÑANA/ANTES/...).
  4. SOV reorder when 1st/2nd person pronoun is direct object.
  5. Locative fronting for HABER existential patterns.
  6. Remove consecutive duplicate tokens.
"""

import re

_TEMPORALES = {
    "AYER", "MAÑANA", "HOY", "SIEMPRE", "NUNCA", "DESPUÉS",
    "ANTES", "AHORA", "TEMPRANO", "PRONTO", "FUTURO",
}

_PREPS_LEAK = {"DE", "EN", "A", "POR", "PARA", "CON", "AL", "DEL"}

_PRON_12_MAP = {
    "me": "YO", "mí": "YO", "yo": "YO",
    "te": "TÚ", "ti": "TÚ", "tú": "TÚ",
    "nos": "NOSOTROS", "nosotros": "NOSOTROS",
    "os": "USTEDES",
}

_LUGARES = {
    "VILLAHERMOSA", "GUADALAJARA", "MONTERREY", "PUEBLA", "OAXACA",
    "TIJUANA", "MÉRIDA", "CANCÚN", "VERACRUZ", "ACAPULCO", "TOLUCA",
    "MEXICO", "MÉXICO", "TEPITO", "TLATELOLCO", "IZTAPALAPA",
    "SAN-FRANCISCO", "CALIFORNIA", "TEXAS", "FLORIDA", "CHICAGO",
    "COLOMBIA", "ESTADOS-UNIDOS",
}


def _fix_dm_prefix(tokens):
    """DM-NOMBRE → dm-NOMBRE (post_processor.clean uppercases everything)."""
    return [
        re.sub(r"^DM-", "dm-", t) if t.startswith("DM-") else t
        for t in tokens
    ]


def _strip_leaked_preps(tokens):
    """Remove standalone prepositions not part of compound tokens."""
    return [t for t in tokens if "-" in t or "+" in t or t not in _PREPS_LEAK]


def _temporal_to_front(tokens):
    """Move temporal markers to position 0 if not already there."""
    temp = [t for t in tokens if t in _TEMPORALES]
    rest = [t for t in tokens if t not in _TEMPORALES]
    return temp + rest


def _deduplicate_consecutive(tokens):
    """Remove consecutive repeated tokens."""
    if not tokens:
        return tokens
    result = [tokens[0]]
    for t in tokens[1:]:
        if t != result[-1]:
            result.append(t)
    return result


def _reorder_sov(tokens, spa_sentence, nlp):
    """Move 1st/2nd person pronoun BEFORE verb when pronoun is direct object.

    LSM uses SOV when OD is 1st/2nd person: "él me robó" → ÉL YO ROBAR
    (not ÉL ROBAR YO). Detects the pattern from the Spanish dependency
    parse, finds corresponding tokens in the gloss, swaps if needed.
    """
    if nlp is None:
        return tokens

    doc = nlp(spa_sentence)

    root_verb = None
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
            root_verb = tok
            break
    if root_verb is None:
        return tokens

    obj_pron_gloss = None
    for tok in doc:
        if tok.dep_ in ("obj", "iobj", "expl:pass") and tok.text.lower() in _PRON_12_MAP:
            obj_pron_gloss = _PRON_12_MAP[tok.text.lower()]
            break
    if obj_pron_gloss is None:
        return tokens

    verb_gloss = root_verb.lemma_.upper()

    verb_idx = None
    pron_idx = None
    for i, t in enumerate(tokens):
        if t == verb_gloss and verb_idx is None:
            verb_idx = i
        if t == obj_pron_gloss and pron_idx is None:
            pron_idx = i

    if verb_idx is not None and pron_idx is not None and verb_idx < pron_idx:
        pron = tokens.pop(pron_idx)
        tokens.insert(verb_idx, pron)

    return tokens


def _front_locative_haber(tokens):
    """Locative fronting for HABER existential patterns.

    LSM: LOC + NOM + HABER. If HABER present and a known place appears
    after it, move place before HABER.
    """
    if "HABER" not in tokens and "NO-HABER" not in tokens:
        return tokens

    haber_tok = "HABER" if "HABER" in tokens else "NO-HABER"
    haber_idx = tokens.index(haber_tok)

    for i in range(haber_idx + 1, len(tokens)):
        if tokens[i] in _LUGARES or tokens[i].startswith("dm-"):
            loc = tokens.pop(i)
            tokens.insert(0, loc)
            break

    return tokens


def postprocess(gloss, spa_sentence=None, nlp=None):
    """Apply all LSM post-processing rules sequentially.

    Args:
        gloss: output of enfoque3 post_processor.clean() (uppercase, stripped).
        spa_sentence: original Spanish sentence (for spaCy-based SOV reorder).
        nlp: loaded spaCy model (optional — SOV reorder skipped if None).

    Returns:
        Post-processed gloss string.
    """
    tokens = gloss.split()
    if not tokens:
        return gloss

    tokens = _fix_dm_prefix(tokens)
    tokens = _strip_leaked_preps(tokens)
    tokens = _temporal_to_front(tokens)
    tokens = _reorder_sov(tokens, spa_sentence, nlp) if spa_sentence else tokens
    tokens = _front_locative_haber(tokens)
    tokens = _deduplicate_consecutive(tokens)

    return " ".join(tokens)
