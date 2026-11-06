"""
Dataset loader and sampling interface for Natural Questions Open.
"""

import json
import os
from typing import Any, Dict, List, Optional

from open_domain_qa.data.sampling import sample_fixed_subset


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class NaturalQuestionsDataset:
    """Interface for loading, caching, and sampling Natural Questions Open questions."""

    def __init__(
        self,
        filepath: Optional[str] = None,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        sampled_path: str = "data/samples/sampled_questions.jsonl",
        seed: int = 42,
        num_samples: int = 300,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.sampled_path = filepath or sampled_path
        self.seed = seed
        self.num_samples = num_samples

        self._records: Optional[List[Dict[str, Any]]] = None

    def load_sampled(self) -> List[Dict[str, Any]]:
        """Loads sampled evaluation questions from disk cache or generates them."""
        target = self.sampled_path
        if not os.path.exists(target) and os.path.exists("data/sampled_questions.jsonl"):
            target = "data/sampled_questions.jsonl"

        if os.path.exists(target):
            self._records = load_jsonl(target)
            return self._records

        return self.prepare_evaluation_sample()

    def load_raw_split(self, split: str = "validation") -> List[Dict[str, Any]]:
        """Loads a split from local cache or downloads from HuggingFace."""
        cached_split = os.path.join(self.raw_dir, f"{split}.jsonl")
        if os.path.exists(cached_split):
            return load_jsonl(cached_split)

        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/nq_open", split=split)
        records = []
        for idx, item in enumerate(ds):
            records.append({
                "question_id": f"nq_{split}_{idx}",
                "question": item["question"],
                "answer": item["answer"],
                "answers": item["answer"],
            })
        os.makedirs(self.raw_dir, exist_ok=True)
        save_jsonl(records, cached_split)
        return records

    def prepare_evaluation_sample(self) -> List[Dict[str, Any]]:
        """Samples exactly num_samples questions deterministically using the configured seed."""
        val_records = self.load_raw_split("validation")
        sampled = sample_fixed_subset(val_records, size=self.num_samples, seed=self.seed)
        save_jsonl(sampled, self.sampled_path)
        self._records = sampled
        return self._records

    @property
    def records(self) -> List[Dict[str, Any]]:
        if self._records is None:
            self._records = self.load_sampled()
        return self._records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    def get_questions(self) -> List[str]:
        return [r["question"] for r in self.records]

    def get_gold_answers(self) -> List[List[str]]:
        return [r.get("answer", r.get("answers", [])) for r in self.records]


# Backward compatibility alias
NQOpenDataset = NaturalQuestionsDataset
