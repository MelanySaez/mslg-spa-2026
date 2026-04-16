"""Retrieval híbrido para enfoque4.

Combina tres señales para recuperar ejemplos del pool:
  1. Similitud léxica BM25 — captura solapamiento de palabras exactas.
  2. Similitud semántica densa (SentenceTransformer) — captura paráfrasis.
  3. Reranking con cross-encoder multilingüe — ordena los candidatos más
     promisorios con un juicio par-a-par más preciso.

Pipeline:
  query → (BM25 top-N) ∪ (denso top-N)
        → fusion Reciprocal Rank Fusion (RRF)
        → filtro por longitud de tokens (oraciones de tamaño similar)
        → rerank cross-encoder sobre top-M supervivientes
        → top-k final

Comparado con `enfoque3/embedding_index.py`, añade robustez léxica,
reduce ruido de longitud y aplica rerank semántico par-a-par.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config as _e3_config  # enfoque3/config.py vía sys.path

from . import config as e4_config


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    """Índice híbrido BM25 + denso + cross-encoder rerank."""

    def __init__(
        self,
        pool,
        use_reranker=None,
        candidates_fusion=None,
        rerank_pool=None,
        length_tolerance=None,
    ):
        self.pool = pool
        self.sentences = [item["spa"] for item in pool]
        self.tokenized = [_tokenize(s) for s in self.sentences]

        self.use_reranker = (
            e4_config.RERANKER_ENABLED if use_reranker is None else use_reranker
        )
        self.candidates_fusion = (
            e4_config.RETRIEVAL_CANDIDATES if candidates_fusion is None else candidates_fusion
        )
        self.rerank_pool = (
            e4_config.RERANK_POOL if rerank_pool is None else rerank_pool
        )
        self.length_tolerance = (
            e4_config.LENGTH_TOLERANCE if length_tolerance is None else length_tolerance
        )

        print(f"Cargando modelo denso: {_e3_config.EMBEDDING_MODEL}...")
        self.dense_model = SentenceTransformer(_e3_config.EMBEDDING_MODEL)

        print(f"Indexando {len(self.sentences)} oraciones (denso + BM25)...")
        self.embeddings = np.array(
            self.dense_model.encode(self.sentences, show_progress_bar=True)
        )
        self.bm25 = BM25Okapi(self.tokenized)

        if self.use_reranker:
            print(f"Cargando cross-encoder: {e4_config.RERANKER_MODEL}...")
            self.reranker = CrossEncoder(e4_config.RERANKER_MODEL)
        else:
            self.reranker = None

        print("Índice híbrido listo.")

    def _dense_order(self, query):
        q_emb = self.dense_model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        return list(np.argsort(-sims))

    def _bm25_order(self, query):
        scores = self.bm25.get_scores(_tokenize(query))
        return list(np.argsort(-scores))

    @staticmethod
    def _rrf(dense_order, bm25_order, kfactor=60, top_n=None):
        """Reciprocal Rank Fusion: combina dos rankings en uno único."""
        scores = {}
        for rank, idx in enumerate(dense_order):
            idx = int(idx)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (kfactor + rank)
        for rank, idx in enumerate(bm25_order):
            idx = int(idx)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (kfactor + rank)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        if top_n is not None:
            ordered = ordered[:top_n]
        return [idx for idx, _ in ordered]

    def _length_filter(self, indices, query):
        """Descarta candidatos con longitud muy diferente a la query.
        Si el filtro vacía el pool, se devuelve el pool original como salvaguarda.
        """
        q_len = max(len(_tokenize(query)), 1)
        kept = []
        for idx in indices:
            cand_len = max(len(self.tokenized[idx]), 1)
            ratio = abs(cand_len - q_len) / q_len
            if ratio <= self.length_tolerance:
                kept.append(idx)
        return kept if kept else indices

    def _rerank(self, query, indices):
        pairs = [(query, self.sentences[idx]) for idx in indices]
        scores = self.reranker.predict(pairs)
        order = sorted(range(len(indices)), key=lambda i: -scores[i])
        return [indices[i] for i in order]

    def retrieve(self, query_sentence, k=10):
        dense_order = self._dense_order(query_sentence)
        bm25_order = self._bm25_order(query_sentence)
        fused = self._rrf(dense_order, bm25_order, top_n=self.candidates_fusion)

        fused = [idx for idx in fused if self.sentences[idx] != query_sentence]
        filtered = self._length_filter(fused, query_sentence)

        if self.reranker is not None:
            pool = filtered[: self.rerank_pool]
            ordered = self._rerank(query_sentence, pool)
            tail = [idx for idx in filtered if idx not in pool]
            ordered = ordered + tail
        else:
            ordered = filtered

        top_k = ordered[:k]
        return [
            {"spa": self.pool[idx]["spa"], "mslg": self.pool[idx]["mslg"]}
            for idx in top_k
        ]
