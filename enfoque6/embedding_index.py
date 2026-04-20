"""Índice de embeddings semánticos para recuperación RAG."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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

    def select_diverse(self, k: int = 10, random_state: int = 42) -> list:
        """Selecciona k ejemplos diversos vía k-means sobre embeddings del pool.

        Toma el centroide de cada cluster y devuelve el ejemplo más cercano.
        Garantiza cobertura de patrones en lugar de similitud a una query.
        """
        k = min(k, len(self.pool))
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(self.embeddings)
        selected = []
        for cluster_id in range(k):
            members = np.where(km.labels_ == cluster_id)[0]
            if len(members) == 0:
                continue
            centroid = km.cluster_centers_[cluster_id].reshape(1, -1)
            member_embs = self.embeddings[members]
            dists = np.linalg.norm(member_embs - centroid, axis=1)
            best_idx = members[np.argmin(dists)]
            selected.append({
                "spa": self.pool[best_idx]["spa"],
                "mslg": self.pool[best_idx]["mslg"],
            })
        return selected
