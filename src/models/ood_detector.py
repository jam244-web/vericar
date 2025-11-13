"""Out-of-Distribution detection."""

import numpy as np
from typing import Tuple


class KNNPlusOOD:
    """KNN+ Out-of-Distribution detection."""
    
    def __init__(self, k: int = 1, threshold: float = None):
        self.k = k
        self.threshold = threshold
        self.database_embeddings = None
        
    def fit(self, embeddings: np.ndarray, target_tpr: float = 0.95):
        """Fit the OOD detector on in-distribution data."""
        self.database_embeddings = embeddings
        
        distances = []
        for i, emb in enumerate(embeddings):
            dists = np.linalg.norm(embeddings - emb, axis=1)
            dists = np.delete(dists, i)
            k_dist = np.partition(dists, min(self.k-1, len(dists)-1))[self.k-1]
            distances.append(k_dist)
        
        self.threshold = np.percentile(distances, target_tpr * 100)
        
    def predict(self, query_embedding: np.ndarray) -> Tuple[bool, float]:
        """Predict if query is OOD."""
        distances = np.linalg.norm(self.database_embeddings - query_embedding, axis=1)
        k_dist = np.partition(distances, min(self.k-1, len(distances)-1))[self.k-1]
        
        is_ood = k_dist > self.threshold
        return is_ood, k_dist