"""Evaluación de métricas: BLEU, METEOR, chrF."""

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import corpus_chrf


def _ensure_nltk_resources():
    """Descarga recursos NLTK necesarios si no están disponibles."""
    for resource in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource != "punkt" and resource != "punkt_tab"
                          else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def evaluate(results):
    """
    Calcula BLEU, METEOR y chrF sobre los resultados.

    Args:
        results: lista de dicts con claves 'mslg_real' y 'mslg_pred'.

    Returns:
        dict con 'bleu', 'meteor', 'chrf'.
    """
    _ensure_nltk_resources()

    references = []
    hypotheses = []

    for r in results:
        ref_tokens = r["mslg_real"].split()
        hyp_tokens = r["mslg_pred"].split()
        references.append([ref_tokens])
        hypotheses.append(hyp_tokens)

    # BLEU (corpus-level, smoothing method 1)
    smoother = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoother)

    # METEOR (promedio de sentence-level)
    meteor_scores = []
    for r in results:
        ref = r["mslg_real"]
        hyp = r["mslg_pred"]
        score = meteor_score([ref.split()], hyp.split())
        meteor_scores.append(score)
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # chrF (corpus-level)
    ref_strings = [[r["mslg_real"]] for r in results]
    hyp_strings = [r["mslg_pred"] for r in results]
    chrf = corpus_chrf(ref_strings, hyp_strings)

    return {
        "bleu": bleu,
        "meteor": meteor_avg,
        "chrf": chrf,
    }
