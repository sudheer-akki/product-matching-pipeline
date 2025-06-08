import os
import faiss
import numpy as np
from cachetools import LRUCache
import hashlib
from utils import safe_log_event


# Cache for FAISS results (key = hash of embedding)
faiss_search_cache = LRUCache(maxsize=1000)

def hash_embedding(embedding: np.ndarray) -> str:
    return hashlib.md5(embedding.tobytes()).hexdigest()

class FaissIndexHandler:
    """
    Handles loading, creating, updating, and querying a FAISS index.
    Supports ID-mapped vector storage and distance-based search.
    """

    def __init__(self, dimension: int, saved_index_path: str | None = None):
        self.dimension = dimension
        self.saved_index_path = saved_index_path
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        """
        Loads the FAISS index from file, or creates a new one if not found.
        """
   
        try:
            if self.saved_index_path and os.path.exists(self.saved_index_path):
                index = faiss.read_index(self.saved_index_path)
                safe_log_event("info", "FAISS index loaded", {"path": self.saved_index_path})
            else:
                index = self._create_index(self.dimension)
                safe_log_event(
                    "info", "FAISS index created from scratch", {"dimension": self.dimension}
                )
            return index
        except Exception as e:
            safe_log_event("error", "Failed to initialize FAISS index", {"exception": str(e)})
            raise

    def add_to_index(self, ids: np.ndarray, embeddings: np.ndarray):
        """
        Adds embeddings with associated numeric IDs to the FAISS index.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :].astype(np.float32)

        embeddings = embeddings.astype(np.float32)
        ids = ids.astype(np.int64)

        assert embeddings.shape[0] == ids.shape[0], "IDs and embeddings must align"
        self.index.add_with_ids(embeddings, ids)

    def _create_index(self, dimension: int = 768):
        """
        Creates a new FAISS index with L2 distance and ID mapping.
        """
        try:
            base_index = faiss.IndexFlatL2(dimension)
            print(f"[INFO] FAISS index initialized from scratch (dimension: {dimension})")
            return faiss.IndexIDMap(base_index)
        except Exception as e:
            print(f"[ERROR] Failed to initialize FAISS index: {e}")
            raise

    def search(self, embedding: np.ndarray, top_k: int = 5,  distance_threshold: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for the top_k closest vectors to the given embedding 
        using saved Cache if available
        """
        try:
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]
            assert embedding.shape[1] == self.dimension, "Embedding dimension mismatch"

            # Use hash of embedding as cache key
            key = hash_embedding(embedding)
            if key in faiss_search_cache:
                safe_log_event("info", "FAISS search served from cache", {"top_k": top_k})
                return faiss_search_cache[key]

            # Perform search and cache result
            distances, ids = self.index.search(embedding.astype("float32"),top_k)
            
            if distance_threshold is not None:
                # Mask values greater than the threshold
                mask = distances <= distance_threshold
                filtered_ids = np.where(mask, ids, -1)  # Mark unmatched as -1
                filtered_distances = np.where(mask, distances, float("inf"))
                safe_log_event("info", "FAISS text filtered ids", {
                    "ids": ids.tolist(),
                    "distances": distances.tolist()
                })
                return filtered_distances, filtered_ids
            faiss_search_cache[key] = (distances, ids)
            safe_log_event("info", "FAISS search executed", {"top_k": top_k})
            return distances, ids
        except Exception as e:
            safe_log_event("error", "FAISS search failed", {"exception": str(e)})
            raise

    def search_batch(self, embeddings: np.ndarray, top_k: int = 5, distance_threshold: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for the top_k closest vectors to each embedding in a batch.

        Args:
            embeddings (np.ndarray): 2D array of shape (batch_size, dimension)
            top_k (int): Number of nearest neighbors to retrieve per embedding.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Distances array of shape (batch_size, top_k)
                - IDs array of shape (batch_size, top_k)
        """
        try:
            assert embeddings.ndim == 2 and embeddings.shape[1] == self.dimension, "Invalid embedding shape"
            distances, ids = self.index.search(embeddings.astype("float32"), k=top_k)
            safe_log_event("info", "FAISS batch search executed", {
            "batch_size": embeddings.shape[0],
            "top_k": top_k
            })
            if distance_threshold is not None:
                # Mask values greater than the threshold
                mask = distances <= distance_threshold
                filtered_ids = np.where(mask, ids, -1)  # Mark unmatched as -1
                filtered_distances = np.where(mask, distances, float("inf"))
                safe_log_event("info", "FAISS text filtered ids", {
                    "ids": ids.tolist(),
                    "distances": distances.tolist()
                })
                return filtered_distances, filtered_ids
        
            return distances, ids
        except Exception as e:
            safe_log_event("error", "FAISS batch search failed", {
            "exception": str(e),
            "input_shape": embeddings.shape
            })
            raise


    def save_index(self) -> None:
        """
        Saves the current FAISS index to disk.
        """
        try:
            if self.saved_index_path:
                os.makedirs(os.path.dirname(self.saved_index_path), exist_ok=True)
                faiss.write_index(self.index, self.saved_index_path)
                safe_log_event("info", "FAISS index saved", {"path": self.saved_index_path})
            else:
                safe_log_event("warning", "No path provided to save FAISS index", {})
        except Exception as e:
            safe_log_event("error", "Failed to save FAISS index", {"exception": str(e)})
            raise
