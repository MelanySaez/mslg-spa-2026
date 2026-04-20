"""Índice de embeddings semánticos para recuperación RAG."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config


class EmbeddingIndex:
    """Índice de embeddings sobre el pool de ejemplos."""

    def __init__(self, pool: list):
        self.pool = pool
        self.sentences = [item["spa"] for item in pool]

        print(f"Cargando modelo de embeddings: {config.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        print(f"Generando embeddings para {len(self.sentences)} oraciones...")
        self.embeddings = np.array(
            self.model.encode(self.sentences, show_progress_bar=True)
        )
        print("Índice listo.")

    def retrieve(self, query_sentence: str, k: int = 5) -> list:
        """Recupera los top-k ejemplos más similares semánticamente."""
        query_emb = self.model.encode([query_sentence])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            if self.sentences[idx] == query_sentence:
                continue
            results.append({
                "spa": self.pool[idx]["spa"],
                "mslg": self.pool[idx]["mslg"],
            })
            if len(results) >= k:
                break

        return results
