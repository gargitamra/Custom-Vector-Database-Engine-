import numpy as np


class BruteForceIndex:
    def __init__(self, embeddings_file="vectors.npy"):
        self.vectors = self._load_vectors(embeddings_file)
        self.is_deleted = np.zeros(len(self.vectors), dtype=bool)

    @staticmethod
    def _load_vectors(embeddings_file):
        vectors = np.asarray(np.load(embeddings_file), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embeddings must not contain zero vectors")
        return vectors / norms

    def _normalize(self, vector):
        value = np.asarray(vector, dtype=np.float32)
        if value.ndim != 1 or value.shape[0] != self.vectors.shape[1]:
            raise ValueError(f"vector must have {self.vectors.shape[1]} dimensions")
        norm = np.linalg.norm(value)
        if norm == 0:
            raise ValueError("vector must not be zero")
        return value / norm

    def search(self, query_vector, k=10):
        if k < 1:
            raise ValueError("k must be at least 1")
        q = self._normalize(query_vector)
        scores = np.dot(self.vectors, q)
        active_indices = np.flatnonzero(~self.is_deleted)
        if len(active_indices) == 0:
            return [], []
        actual_k = min(k, len(active_indices))
        active_scores = scores[active_indices]
        top_k_relative = np.argsort(active_scores)[-actual_k:][::-1]
        top_k_indices = active_indices[top_k_relative]
        top_k_scores = scores[top_k_indices]
        return top_k_indices.tolist(), top_k_scores.tolist()

    def insert(self, vector):
        v = self._normalize(vector)
        self.vectors = np.vstack([self.vectors, v])
        self.is_deleted = np.append(self.is_deleted, False)
        return len(self.vectors) - 1

    def delete(self, idx):
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.vectors):
            if self.is_deleted[idx]:
                return False
            self.is_deleted[idx] = True
            return True
        return False