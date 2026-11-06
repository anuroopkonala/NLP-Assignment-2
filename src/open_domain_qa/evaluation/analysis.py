"""
Error analysis and failure mode categorization for open-domain QA:
- RETRIEVAL_FAILURE: No acceptable gold answer appears in the top-k passages.
- EXTRACTION_OR_MISATTRIBUTION: Answer is in context, but model extracted the wrong span.
- HALLUCINATION_OR_UNGROUNDED: Generated answer lacks sufficient evidence in context.
- BOUNDARY_OR_PARTIAL_OVERLAP: Partial token overlap (F1 > 0) without exact match.
- EXACT_MATCH: Correct answer prediction.
"""

from typing import Any, Dict, List
from open_domain_qa.evaluation.normalization import normalize_answer
from open_domain_qa.evaluation.metrics import compute_exact_match, compute_f1
from open_domain_qa.evaluation.recall import compute_recall_at_k
from open_domain_qa.verification.groundedness import compute_lexical_overlap


def categorize_error(
    question: str,
    prediction: str,
    ground_truths: List[str],
    retrieved_passages: List[str],
    k: int = 5,
    groundedness_threshold: float = 0.5,
) -> str:
    """Categorizes a prediction into a failure mode or exact match."""
    if compute_exact_match(prediction, ground_truths):
        return "EXACT_MATCH"

    has_retrieval_hit = compute_recall_at_k(retrieved_passages, ground_truths, k=k)
    f1 = compute_f1(prediction, ground_truths)
    combined_context = " ".join(retrieved_passages[:k])
    overlap = compute_lexical_overlap(prediction, combined_context)

    if not has_retrieval_hit:
        return "RETRIEVAL_FAILURE"

    if overlap < groundedness_threshold:
        return "HALLUCINATION_OR_UNGROUNDED"

    if f1 > 0.0:
        return "BOUNDARY_OR_PARTIAL_OVERLAP"

    return "EXTRACTION_OR_MISATTRIBUTION"


def analyze_pipeline_errors(
    results: List[Dict[str, Any]],
    k: int = 5,
    groundedness_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes error mode breakdown and gathers example queries per category."""
    counts = {
        "EXACT_MATCH": 0,
        "RETRIEVAL_FAILURE": 0,
        "EXTRACTION_OR_MISATTRIBUTION": 0,
        "BOUNDARY_OR_PARTIAL_OVERLAP": 0,
        "HALLUCINATION_OR_UNGROUNDED": 0,
    }
    examples_by_category: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in counts}

    total = len(results)
    for item in results:
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        gts = item.get("answers", item.get("answer", []))
        if isinstance(gts, str):
            gts = [gts]

        passages = item.get("passages", item.get("retrieved_passages", []))
        if passages and isinstance(passages[0], dict):
            passages = [p.get("text", "") for p in passages]

        cat = categorize_error(
            question=question,
            prediction=prediction,
            ground_truths=gts,
            retrieved_passages=passages,
            k=k,
            groundedness_threshold=groundedness_threshold,
        )

        counts[cat] += 1
        if len(examples_by_category[cat]) < 5:
            examples_by_category[cat].append({
                "question": question,
                "prediction": prediction,
                "ground_truths": gts,
                "category": cat,
            })

    breakdown = {
        cat: {
            "count": count,
            "percentage": round(100.0 * count / total, 2) if total > 0 else 0.0,
        }
        for cat, count in counts.items()
    }

    return {
        "total_instances": total,
        "breakdown": breakdown,
        "sample_cases": examples_by_category,
    }
