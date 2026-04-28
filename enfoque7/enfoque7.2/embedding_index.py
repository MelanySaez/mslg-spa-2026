"""Índice de embeddings para retrieval RAG sobre el pool indexado por MSLG.

Variante de enfoque6/embedding_index.py adaptada al sentido reverso: la query
es una glosa MSLG y los vecinos top-k se buscan por similitud MSLG-MSLG en el
pool, devolviendo pares (mslg, spa) para usarse como ejemplos few-shot.

Nota: el modelo Sentence-BERT multilingual no fue entrenado sobre glosa LSM,
por lo que la similitud puede ser ruidosa. Aun así produce señal útil porque
muchos tokens de la glosa son palabras españolas en mayúsculas (TRABAJAR,
HERMANO, AYER, etc.) que el embedding sí reconoce.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config


class EmbeddingIndex:
    """Índice de embeddings sobre el pool, indexado por el campo MSLG."""

    def __init__(self, pool: list):
        self.pool = pool
        self.sentences = [item["mslg"] for item in pool]

        print(f"Cargando modelo de embeddings: {config.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        print(f"Generando embeddings MSLG para {len(self.sentences)} oraciones...")
        self.embeddings = np.array(
            self.model.encode(self.sentences, show_progress_bar=True)
        )
        print("Índice listo.")

    def retrieve(self, query_mslg: str, k: int = 5) -> list:
        """Recupera los top-k pares más similares por MSLG."""
        query_emb = self.model.encode([query_mslg])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            if self.sentences[idx] == query_mslg:
                continue
            results.append({
                "spa": self.pool[idx]["spa"],
                "mslg": self.pool[idx]["mslg"],
            })
            if len(results) >= k:
                break

        return results
