"""Answering module providing classical extractive answering and RAG generation."""

from open_domain_qa.answering.extraction import (
    ExtractiveAnswerer,
    extract_longest_candidate,
    select_best_sentence,
)
from open_domain_qa.answering.prompting import (
    format_rag_prompt,
    truncate_tokens,
)
from open_domain_qa.answering.generator import (
    MockRAGGenerator,
    RAGGenerator,
)

__all__ = [
    "ExtractiveAnswerer",
    "extract_longest_candidate",
    "select_best_sentence",
    "format_rag_prompt",
    "truncate_tokens",
    "RAGGenerator",
    "MockRAGGenerator",
]
