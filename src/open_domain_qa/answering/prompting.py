"""
Prompt construction and passage truncation for RAG generators.
Truncates evidence passages to 200 words and formats clean bracketed indices.
"""

from typing import Any, Dict, List, Optional


def truncate_tokens(text: str, max_tokens: int = 200) -> str:
    """Truncates text to at most max_tokens whitespace-separated tokens."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def format_rag_prompt(
    question: str,
    context_passages: List[str],
    max_tokens_per_passage: int = 200,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Formats an open-domain QA prompt for instruction-tuned LLMs."""
    formatted_passages = []
    for idx, passage in enumerate(context_passages):
        clean_text = truncate_tokens(passage, max_tokens=max_tokens_per_passage)
        formatted_passages.append(f"[{idx + 1}] {clean_text}")

    context_str = "\n\n".join(formatted_passages)

    few_shot_str = ""
    if few_shot_examples:
        blocks = []
        for ex in few_shot_examples:
            ex_context = truncate_tokens(ex.get("context", ""), max_tokens=max_tokens_per_passage)
            blocks.append(
                f"Context:\n[1] {ex_context}\n\n"
                f"Question: {ex.get('question', '')}\n"
                f"Answer: {ex.get('answer', '')}"
            )
        few_shot_str = "\n\n".join(blocks) + "\n\n"

    prompt = (
        "Answer the following question directly and concisely based only on the provided context passages. "
        "Provide only the short answer.\n\n"
        f"{few_shot_str}"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt
