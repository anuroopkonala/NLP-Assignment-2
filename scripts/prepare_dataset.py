#!/usr/bin/env python3
"""
Dataset sampling script for Natural Questions Open (google-research-datasets/nq_open).
- Samples exactly 300 validation questions with seed 42 for final evaluation.
- Samples 100 non-overlapping development questions with seed 100 for threshold calibration.
"""

import argparse
import hashlib
import json
import os
import sys

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from open_domain_qa.config import load_config
from open_domain_qa.data.dataset import NaturalQuestionsDataset
from open_domain_qa.data.sampling import sample_fixed_subset


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare Natural Questions Open validation subsets.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--sample_size", type=int, default=300, help="Evaluation sample size (default: 300).")
    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for evaluation split (default: 42).")
    parser.add_argument("--dev_size", type=int, default=100, help="Development calibration sample size (default: 100).")
    parser.add_argument("--dev_seed", type=int, default=100, help="Seed for dev split (default: 100).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print("Preparing Natural Questions Open Subsets")
    print("=" * 70)

    nq = NaturalQuestionsDataset(
        raw_dir=cfg.dataset.raw_dir,
        processed_dir=cfg.dataset.processed_dir,
        sampled_path=cfg.dataset.sampled_path,
        seed=args.eval_seed,
        num_samples=args.sample_size,
    )

    # Load or sample evaluation subset (300 questions, seed 42)
    os.makedirs(os.path.dirname(cfg.dataset.sampled_path), exist_ok=True)
    if os.path.exists(cfg.dataset.sampled_path):
        print(f"Loading existing evaluation subset from: {cfg.dataset.sampled_path}")
        eval_samples = nq.load_sampled()
    else:
        print("Sampling 300 evaluation questions from validation split (seed 42)...")
        eval_samples = nq.prepare_evaluation_sample()

    print(f"Evaluation sample verified: {len(eval_samples)} items.")
    eval_sha = compute_sha256(cfg.dataset.sampled_path)
    print(f"Evaluation dataset SHA-256: {eval_sha}")

    # Load or sample development calibration subset (100 questions, seed 100)
    dev_path = "data/samples/dev_calibration_questions.jsonl"
    if os.path.exists(dev_path):
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_samples = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded existing dev calibration subset: {len(dev_samples)} items.")
    else:
        print("Sampling 100 non-overlapping dev calibration questions (seed 100)...")
        eval_ids = set(q.get("question_id", q.get("id")) for q in eval_samples)
        all_val = nq.load_raw_split("validation")
        remaining = [q for q in all_val if q.get("question_id", q.get("id")) not in eval_ids]
        dev_samples = sample_fixed_subset(remaining, size=args.dev_size, seed=args.dev_seed)
        with open(dev_path, "w", encoding="utf-8") as f:
            for item in dev_samples:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(dev_samples)} dev calibration questions to: {dev_path}")

    dev_sha = compute_sha256(dev_path)
    print(f"Dev calibration dataset SHA-256: {dev_sha}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
