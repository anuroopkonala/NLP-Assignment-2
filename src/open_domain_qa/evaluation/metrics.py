"""
Core QA evaluation metrics: Exact Match (EM) and token-level F1.
Follows standard multi-reference evaluation protocols.
"""

from collections import Counter
from typing import Any, Dict, List

from open_domain_qa.evaluation.normalization import normalize_answer


def compute_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Returns True if normalized prediction matches at least one reference answer."""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_gt and norm_pred == norm_gt:
            return True
    return False


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Computes token-level F1 score against multiple gold references, returning the maximum."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 1.0 if any(len(normalize_answer(gt).split()) == 0 for gt in ground_truths) else 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > max_f1:
            max_f1 = f1

    return max_f1


def evaluate_qa_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes dataset-level macro Exact Match and Token F1 percentages."""
    if not predictions:
        return {"exact_match": 0.0, "f1": 0.0, "total_count": 0}

    total_em = 0
    total_f1 = 0.0

    for item in predictions:
        pred = item.get("prediction", "")
        gts = item.get("answers", item.get("answer", []))
        if isinstance(gts, str):
            gts = [gts]

        em = compute_exact_match(pred, gts)
        f1 = compute_f1(pred, gts)

        if em:
            total_em += 1
        total_f1 += f1

    n = len(predictions)
    return {
        "exact_match": round(100.0 * total_em / n, 2),
        "f1": round(100.0 * total_f1 / n, 2),
        "total_count": n,
    }
