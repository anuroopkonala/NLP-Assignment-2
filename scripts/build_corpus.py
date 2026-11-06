#!/usr/bin/env python3
"""
Corpus verification script.
Checks the frozen Wikipedia passages, computes SHA-256 checksums,
and verifies token length distributions and coverage.
"""

import argparse
import hashlib
import json
import os
import sys

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from open_domain_qa.config import load_config
from open_domain_qa.corpus.wikipedia import WikipediaCorpus
from open_domain_qa.corpus.manifest import generate_corpus_manifest


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Verify Wikipedia passage corpus.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--corpus_path", type=str, default="data/wikipedia/nq_open_n300_seed42.jsonl")
    parser.add_argument("--manifest_path", type=str, default="data/wikipedia/corpus_manifest.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = args.corpus_path if os.path.exists(args.corpus_path) else cfg.corpus.output_path
    if not os.path.exists(path) and os.path.exists("data/nq_open_n300_seed42.jsonl"):
        path = "data/nq_open_n300_seed42.jsonl"

    print("=" * 70)
    print("Verifying Wikipedia Passage Corpus")
    print("=" * 70)
    print(f"Corpus location: {path}")

    corpus = WikipediaCorpus(corpus_path=path)
    passages = corpus.load()
    print(f"Loaded {len(passages):,} passages.")

    lengths = [len(p.get("text", "").split()) for p in passages]
    titles = set(p.get("title", "") for p in passages)

    print(f"Unique Wikipedia articles: {len(titles):,}")
    print(f"Mean passage length: {sum(lengths) / len(lengths):.2f} words")
    print(f"Max passage length: {max(lengths)} words")
    print(f"Min passage length: {min(lengths)} words")
    print(f"Passages <= 200 words: {sum(1 for l in lengths if l <= 200):,} ({100.0 * sum(1 for l in lengths if l <= 200) / len(lengths):.2f}%)")

    sha = compute_sha256(path)
    print(f"Corpus SHA-256: {sha}")

    # Generate or update manifest
    manifest = generate_corpus_manifest(
        corpus_path=path,
        output_manifest_path=args.manifest_path,
        metadata={"total_passages": len(passages), "unique_articles": len(titles), "sha256": sha},
    )
    print(f"Manifest written to: {args.manifest_path}")
    print("Corpus verification complete.")


if __name__ == "__main__":
    main()
