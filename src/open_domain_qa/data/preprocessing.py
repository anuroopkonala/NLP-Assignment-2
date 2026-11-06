"""
Query text preprocessing and cleanup utilities.
"""

import re


def clean_query_text(text: str) -> str:
    """Cleans a query string by trimming extra whitespace and trailing question marks."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\?+$", "", text).strip()
    return " ".join(text.split())
