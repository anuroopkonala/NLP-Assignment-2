"""
Dense neural retrieval using sentence-transformers/all-MiniLM-L6-v2.
Builds a FAISS Flat Inner-Product index on L2-normalized embeddings,
ensuring inner products correspond directly to cosine similarities.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Dense retriever using all-MiniLM-L6-v2 and FAISS IndexFlatIP."""

    def __init__(
        self,
        corpus: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.corpus = corpus or []
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Loads precomputed numpy embeddings and builds the FAISS index."""
        embeddings = np.load(filepath).astype(np.float32)
        self.embeddings = embeddings
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return embeddings

    def save_embeddings(self, filepath: str) -> None:
        """Saves current embeddings matrix to disk."""
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Encode corpus first.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.embeddings)

    def index_corpus(
        self,
        corpus_texts: Optional[List[str]] = None,
        batch_size: int = 256,
        cache_path: Optional[str] = None,
    ) -> np.ndarray:
        """Encodes corpus texts with L2 normalization for exact cosine similarity."""
        if cache_path and os.path.exists(cache_path):
            return self.load_embeddings(cache_path)

        if corpus_texts is None:
            if not self.corpus:
                raise ValueError("No corpus provided to encode.")
            corpus_texts = [p.get("text", "") for p in self.corpus]

        embeddings = self.embedder.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.embeddings = embeddings
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        if cache_path:
            self.save_embeddings(cache_path)

        return embeddings

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Encodes query and returns structured list of top-k retrieved passages."""
        if self.index is None:
            raise RuntimeError("Index not built. Call index_corpus() or load_embeddings() first.")

        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for rank_idx in range(top_k):
            doc_idx = int(indices[0][rank_idx])
            score = float(scores[0][rank_idx])
            meta = self.corpus[doc_idx] if self.corpus and doc_idx < len(self.corpus) else {}
            results.append({
                "rank": rank_idx + 1,
                "doc_id": doc_idx,
                "score": score,
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
            })
        return results

    def retrieve_tuples(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Returns raw (doc_id, score) tuples."""
        hits = self.retrieve(query, top_k=top_k)
        return [(h["doc_id"], h["score"]) for h in hits]
