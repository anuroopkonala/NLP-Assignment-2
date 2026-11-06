"""
BM25 lexical retrieval module using rank_bm25 (k1=1.5, b=0.75).
Supports both standard passage text indexing and title-aware indexing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25Okapi lexical retriever over the Wikipedia passage corpus."""

    def __init__(
        self,
        corpus: Union[List[Dict[str, Any]], List[List[str]]],
        k1: float = 1.5,
        b: float = 0.75,
        use_title: bool = False,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.use_title = use_title

        if corpus and isinstance(corpus[0], dict):
            self.corpus: List[Dict[str, Any]] = corpus  # type: ignore
            if use_title:
                tokenized = [
                    f"{p.get('title', '')} {p.get('section', '')}: {p.get('text', '')}".lower().split()
                    for p in self.corpus
                ]
            else:
                tokenized = [p.get("text", "").lower().split() for p in self.corpus]
        else:
            self.corpus = []
            tokenized = corpus  # type: ignore

        self.tokenized_corpus = tokenized
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    @classmethod
    def from_passages(
        cls,
        passages: List[Dict[str, Any]],
        use_title: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":
        """Builds BM25 index from a list of passage dictionaries."""
        return cls(corpus=passages, k1=k1, b=b, use_title=use_title)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Scores corpus with query and returns structured list of top-k passages."""
        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank_idx, doc_id in enumerate(top_indices, start=1):
            doc_idx = int(doc_id)
            meta = self.corpus[doc_idx] if doc_idx < len(self.corpus) else {}
            results.append({
                "rank": rank_idx,
                "doc_id": doc_idx,
                "score": float(scores[doc_idx]),
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
            })
        return results

    def retrieve_tuples(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Returns raw (doc_id, score) tuples."""
        hits = self.retrieve(query, top_k=top_k)
        return [(h["doc_id"], h["score"]) for h in hits]
