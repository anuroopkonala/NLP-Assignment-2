"""
Wikipedia passage corpus container.
Manages the frozen 45,029 Wikipedia passages harvested from 1,806 articles.
"""

import os
from typing import Any, Dict, List, Optional
from open_domain_qa.data.dataset import load_jsonl


class WikipediaCorpus:
    """Manages the frozen Wikipedia passage collection."""

    def __init__(self, filepath: Optional[str] = None, corpus_path: Optional[str] = None) -> None:
        target = filepath or corpus_path or "data/wikipedia/nq_open_n300_seed42.jsonl"
        if not os.path.exists(target) and os.path.exists("data/nq_open_n300_seed42.jsonl"):
            target = "data/nq_open_n300_seed42.jsonl"

        self.filepath = target
        self.passages: List[Dict[str, Any]] = []
        self.passage_dict: Dict[Any, Dict[str, Any]] = {}
        if os.path.exists(self.filepath):
            self.load()

    def load(self) -> List[Dict[str, Any]]:
        """Loads passages from disk into memory."""
        if not self.passages:
            self.passages = load_jsonl(self.filepath)
            self.passage_dict = {p.get("id", idx): p for idx, p in enumerate(self.passages)}
        return self.passages

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.passages[idx]

    def get_by_id(self, passage_id: Any) -> Optional[Dict[str, Any]]:
        return self.passage_dict.get(passage_id)

    def get_texts(self) -> List[str]:
        return [p.get("text", "") for p in self.passages]

    def get_titles_and_texts(self) -> List[str]:
        """Returns title-prepended text for title-aware lexical indexing."""
        return [f"{p.get('title', '')} {p.get('section', '')}: {p.get('text', '')}" for p in self.passages]
