"""Evaluation module for Exact Match, Token F1, Recall@k, and error diagnosis."""

from open_domain_qa.evaluation.normalization import normalize_answer
from open_domain_qa.evaluation.metrics import (
    compute_exact_match,
    compute_f1,
    evaluate_qa_predictions,
)
from open_domain_qa.evaluation.recall import (
    compute_recall_at_k,
    evaluate_retrieval_recall,
)
from open_domain_qa.evaluation.analysis import (
    analyze_pipeline_errors,
    categorize_error,
)

__all__ = [
    "normalize_answer",
    "compute_exact_match",
    "compute_f1",
    "evaluate_qa_predictions",
    "compute_recall_at_k",
    "evaluate_retrieval_recall",
    "categorize_error",
    "analyze_pipeline_errors",
]
