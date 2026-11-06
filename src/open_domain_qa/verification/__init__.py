"""Verification module for evidence groundedness and hallucination detection."""

from open_domain_qa.verification.groundedness import (
    GroundednessVerifier,
    compute_lexical_overlap,
    normalize_text,
)

__all__ = [
    "GroundednessVerifier",
    "compute_lexical_overlap",
    "normalize_text",
]
