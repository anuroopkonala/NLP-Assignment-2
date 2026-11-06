"""
Passage segmentation and word truncation utilities.
"""


def truncate_tokens(text: str, max_tokens: int = 200) -> str:
    """Truncates a passage to at most max_tokens whitespace words."""
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
