"""
Hybrid retrieval combining lexical BM25 and dense neural rankings.
Uses Reciprocal Rank Fusion (RRF, constant c=60) to merge top candidate lists.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from open_domain_qa.retrieval.bm25 import BM25Retriever
from open_domain_qa.retrieval.dense import DenseRetriever


def compute_reciprocal_rank_fusion(
    bm25_ranked_ids: List[int],
    dense_ranked_ids: List[int],
    constant: int = 60,
) -> List[int]:
    """
    Combines two ranked candidate lists using Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum(1.0 / (constant + rank(d)))
    where rank(d) is 1-indexed.
    """
    scores: Dict[int, float] = defaultdict(float)

    for rank_idx, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] += 1.0 / (constant + rank_idx + 1)

    for rank_idx, doc_id in enumerate(dense_ranked_ids):
        scores[doc_id] += 1.0 / (constant + rank_idx + 1)

    sorted_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    return sorted_ids


class HybridRetriever:
    """Combines BM25 and Dense candidate pools using Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_constant: int = 60,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_constant = rrf_constant
        self.corpus = self.bm25.corpus

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k passages using Reciprocal Rank Fusion."""
        bm25_candidates = self.bm25.retrieve(query, top_k=candidate_pool_size)
        dense_candidates = self.dense.retrieve(query, top_k=candidate_pool_size)

        bm25_ids = [c["doc_id"] for c in bm25_candidates]
        dense_ids = [c["doc_id"] for c in dense_candidates]

        fused_ids = compute_reciprocal_rank_fusion(
            bm25_ids,
            dense_ids,
            constant=self.rrf_constant,
        )

        top_ids = fused_ids[:top_k]

        bm25_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(bm25_ids)}
        dense_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(dense_ids)}

        results = []
        for rank_idx, doc_id in enumerate(top_ids, start=1):
            score = 0.0
            if doc_id in bm25_rank_map:
                score += 1.0 / (self.rrf_constant + bm25_rank_map[doc_id])
            if doc_id in dense_rank_map:
                score += 1.0 / (self.rrf_constant + dense_rank_map[doc_id])

            passage_meta = self.corpus[doc_id] if doc_id < len(self.corpus) else {}
            results.append({
                "rank": rank_idx,
                "doc_id": doc_id,
                "score": float(score),
                "text": passage_meta.get("text", ""),
                "title": passage_meta.get("title", ""),
                "bm25_rank": bm25_rank_map.get(doc_id),
                "dense_rank": dense_rank_map.get(doc_id),
            })

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> List[List[Dict[str, Any]]]:
        """Runs batch hybrid retrieval across multiple queries."""
        return [self.retrieve(q, top_k=top_k, candidate_pool_size=candidate_pool_size) for q in queries]
