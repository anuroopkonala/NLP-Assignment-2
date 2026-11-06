"""
Information retrieval Recall@k evaluation:
A retrieval hit occurs if any acceptable reference answer appears as a substring
within any of the top-k retrieved passages.
"""

from typing import Any, Dict, List

from open_domain_qa.evaluation.normalization import normalize_answer


def compute_recall_at_k(
    retrieved_passages: List[str],
    ground_truths: List[str],
    k: int,
) -> bool:
    """Checks if any normalized gold answer string appears in the top-k passages."""
    norm_gts = [normalize_answer(gt) for gt in ground_truths if normalize_answer(gt)]
    if not norm_gts:
        return False

    for passage in retrieved_passages[:k]:
        norm_passage = normalize_answer(passage)
        for gt in norm_gts:
            if gt in norm_passage:
                return True
    return False


def evaluate_retrieval_recall(
    rankings: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Computes Recall@k percentages across all queries."""
    if not rankings:
        return {f"recall@{k}": 0.0 for k in k_values}

    hits = {k: 0 for k in k_values}
    total = len(rankings)

    for item in rankings:
        passages = item.get("passages", item.get("retrieved_passages", []))
        if passages and isinstance(passages[0], dict):
            passages = [p.get("text", "") for p in passages]

        gts = item.get("answers", item.get("answer", []))
        if isinstance(gts, str):
            gts = [gts]

        for k in k_values:
            if compute_recall_at_k(passages, gts, k=k):
                hits[k] += 1

    return {f"recall@{k}": round(100.0 * hits[k] / total, 2) for k in k_values}
