"""
Standard answer string normalization utility:
lowercases text, removes articles ('a', 'an', 'the'), strips punctuation, and fixes spacing.
"""

import re
import string


def normalize_answer(text: str) -> str:
    """Normalizes an answer string according to standard benchmark conventions."""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())
