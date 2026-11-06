"""
Task 3 Groundedness Verification:
Computes content-token lexical overlap between generated answers and retrieved evidence.
Filters grammatical stopwords to assess entity grounding.
"""

import re
import string
from typing import Any, Dict, List, Optional, Set, Union


# Common English grammatical stopwords filtered out to measure content-word groundedness
STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "of", "for", "to", "and", "or",
    "is", "was", "are", "were", "it", "its", "that", "this", "by", "from",
    "with", "as", "be", "has", "had", "have"
}


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def compute_lexical_overlap(answer: str, context: str) -> float:
    """
    Computes the fraction of non-stopword answer tokens found in the context.
    If the answer consists only of stopwords or numbers, returns 1.0.
    """
    norm_ans_tokens = normalize_text(answer).split()
    content_tokens = [t for t in norm_ans_tokens if t not in STOPWORDS]

    if not content_tokens:
        return 1.0

    context_tokens = set(normalize_text(context).split())
    matched_count = sum(1 for t in content_tokens if t in context_tokens)

    return float(matched_count) / float(len(content_tokens))


class GroundednessVerifier:
    """Evaluates and enforces groundedness of generated answers against retrieved context."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def verify(
        self,
        answer: str,
        context_passages: Union[List[str], str],
        use_max_passage: bool = True,
    ) -> Dict[str, Any]:
        """
        Verifies if answer is sufficiently grounded in context passages.
        Uses single-best-passage maximum overlap by default to prevent cross-passage
        token concatenation artifacts.
        Returns a dictionary with overlap score and verification decision.
        """
        if isinstance(context_passages, list):
            if use_max_passage and len(context_passages) > 0:
                overlap = max(compute_lexical_overlap(answer, p) for p in context_passages)
            else:
                combined_context = " ".join(context_passages)
                overlap = compute_lexical_overlap(answer, combined_context)
        else:
            overlap = compute_lexical_overlap(answer, context_passages)

        is_grounded = overlap >= self.threshold

        return {
            "overlap_score": float(overlap),
            "threshold": self.threshold,
            "is_grounded": bool(is_grounded),
            "final_answer": answer if is_grounded else "[UNGROUNDED / ABSTAINED]",
        }

    @staticmethod
    def evaluate_threshold_sweep(
        predictions: List[Dict[str, Any]],
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Sweeps over multiple groundedness thresholds on an evaluation/calibration set."""
        if thresholds is None:
            thresholds = [round(x * 0.1, 2) for x in range(1, 11)]

        sweep_results = []
        for tau in thresholds:
            grounded_count = 0
            correct_grounded = 0
            incorrect_ungrounded = 0
            total = len(predictions)

            for item in predictions:
                overlap = item.get("lexical_overlap", item.get("groundedness", {}).get("overlap_score", 0.0))
                is_correct = bool(item.get("exact_match", False))
                is_grounded = (overlap >= tau)

                if is_grounded:
                    grounded_count += 1
                    if is_correct:
                        correct_grounded += 1
                else:
                    if not is_correct:
                        incorrect_ungrounded += 1

            acceptance_rate = grounded_count / total if total > 0 else 0.0
            accepted_precision = correct_grounded / grounded_count if grounded_count > 0 else 0.0
            hallucination_rejection = incorrect_ungrounded / (total - sum(1 for p in predictions if p.get("exact_match", False))) if (total - sum(1 for p in predictions if p.get("exact_match", False))) > 0 else 0.0

            sweep_results.append({
                "threshold": tau,
                "grounded_count": grounded_count,
                "acceptance_rate": round(acceptance_rate, 4),
                "accepted_precision": round(accepted_precision, 4),
                "hallucination_rejection_rate": round(hallucination_rejection, 4),
            })

        return sweep_results
