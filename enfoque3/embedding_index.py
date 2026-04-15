"""Índice de embeddings semánticos para recuperación RAG."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config


class EmbeddingIndex:
    """Construye un índice de embeddings sobre el pool de ejemplos."""

    def __init__(self, pool):
        """
        Args:
            pool: lista de dicts {id, spa, mslg} del pool de entrenamiento.
        """
        self.pool = pool
        self.sentences = [item["spa"] for item in pool]

        print(f"Cargando modelo de embeddings: {config.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        print(f"Generando embeddings para {len(self.sentences)} oraciones...")
        self.embeddings = self.model.encode(self.sentences, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings)
        print("Índice de embeddings listo.")

    def retrieve(self, query_sentence, k=5):
        """
        Recupera los top-k ejemplos más similares a la query.

        Args:
            query_sentence: oración en español a buscar.
            k: número de ejemplos a retornar.

        Returns:
            Lista de dicts {spa, mslg} de los k ejemplos más similares.
        """
        query_emb = self.model.encode([query_sentence])
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        # Excluir la propia oración si aparece (similarity ~1.0)
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
