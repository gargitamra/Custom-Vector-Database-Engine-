import numpy as np


class IVFFlatIndex:
    def __init__(self, embeddings_file="vectors.npy", num_clusters=100, n_probe=5):
        self.vectors = self._load_vectors(embeddings_file)
        self.num_clusters = min(num_clusters, len(self.vectors))
        if self.num_clusters < 1:
            raise ValueError("num_clusters must be at least 1")
        self.n_probe = max(1, min(n_probe, self.num_clusters))
        self.is_deleted = np.zeros(len(self.vectors), dtype=bool)
        self.centroids = None
        self.inverted_lists = {i: [] for i in range(self.num_clusters)}
        self._train()

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

    def _train(self, num_iterations=15):
        rng = np.random.default_rng(42)
        random_indices = rng.choice(len(self.vectors), self.num_clusters, replace=False)
        self.centroids = self.vectors[random_indices].copy()

        for _ in range(num_iterations):
            similarities = np.dot(self.vectors, self.centroids.T)
            cluster_assignments = np.argmax(similarities, axis=1)
            new_centroids = np.zeros_like(self.centroids)
            for cluster_id in range(self.num_clusters):
                cluster_mask = (cluster_assignments == cluster_id)
                if np.any(cluster_mask):
                    cluster_mean = np.mean(self.vectors[cluster_mask], axis=0)
                    new_centroids[cluster_id] = cluster_mean / np.linalg.norm(cluster_mean)
                else:
                    new_centroids[cluster_id] = self.centroids[cluster_id]
            self.centroids = new_centroids

        final_similarities = np.dot(self.vectors, self.centroids.T)
        final_assignments = np.argmax(final_similarities, axis=1)
        for idx, cluster_id in enumerate(final_assignments):
            self.inverted_lists[cluster_id].append(idx)

    def search(self, query_vector, k=10):
        if k < 1:
            raise ValueError("k must be at least 1")
        q = self._normalize(query_vector)
        probe_count = min(self.n_probe, self.num_clusters)
        centroid_scores = np.dot(self.centroids, q)
        top_cluster_ids = np.argsort(centroid_scores)[-probe_count:][::-1]
        candidate_indices = []
        for cluster_id in top_cluster_ids:
            candidate_indices.extend(self.inverted_lists[cluster_id])
        if not candidate_indices:
            return [], []
        candidate_indices = np.asarray(candidate_indices, dtype=np.intp)
        candidate_indices = candidate_indices[~self.is_deleted[candidate_indices]]
        if len(candidate_indices) == 0:
            return [], []
        candidate_vectors = self.vectors[candidate_indices]
        scores = np.dot(candidate_vectors, q)
        actual_k = min(k, len(scores))
        top_k_relative_indices = np.argsort(scores)[-actual_k:][::-1]
        top_k_indices = candidate_indices[top_k_relative_indices]
        top_k_scores = scores[top_k_relative_indices]
        return top_k_indices.tolist(), top_k_scores.tolist()

    def insert(self, vector):
        v = self._normalize(vector)
        new_idx = len(self.vectors)
        self.vectors = np.vstack([self.vectors, v])
        self.is_deleted = np.append(self.is_deleted, False)
        centroid_scores = np.dot(self.centroids, v)
        best_cluster_id = np.argmax(centroid_scores)
        self.inverted_lists[best_cluster_id].append(new_idx)
        return new_idx

    def delete(self, idx):
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.vectors):
            if self.is_deleted[idx]:
                return False
            self.is_deleted[idx] = True
            return True
        return False