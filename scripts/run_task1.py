#!/usr/bin/env python3
"""
CLI runner for Task 1: Classical BM25 Retrieval + Extractive Question Answering.
"""

import argparse
import json
import os
import sys

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from open_domain_qa.config import load_config
from open_domain_qa.corpus.wikipedia import WikipediaCorpus
from open_domain_qa.data.dataset import NaturalQuestionsDataset
from open_domain_qa.pipeline.task1 import Task1Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Task 1: BM25 + Extractive Baseline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus JSONL.")
    parser.add_argument("--questions_path", type=str, default=None, help="Path to evaluation questions JSONL.")
    parser.add_argument("--output_path", type=str, default="outputs/task1/task1_results.json", help="Path to save output JSON.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved passages for answering (default: 5).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print("Executing Task 1: Classical BM25 Retrieval & Extractive QA")
    print("=" * 70)

    corpus_file = args.corpus_path or cfg.corpus.output_path
    if not os.path.exists(corpus_file) and os.path.exists("data/nq_open_n300_seed42.jsonl"):
        corpus_file = "data/nq_open_n300_seed42.jsonl"

    questions_file = args.questions_path or cfg.dataset.sampled_path
    if not os.path.exists(questions_file) and os.path.exists("data/sampled_questions.jsonl"):
        questions_file = "data/sampled_questions.jsonl"

    print(f"Loading corpus from: {corpus_file}")
    corpus = WikipediaCorpus(corpus_path=corpus_file).load()
    print(f"Loaded {len(corpus):,} passages.")

    print(f"Loading questions from: {questions_file}")
    questions = NaturalQuestionsDataset(sampled_path=questions_file).load_sampled()
    print(f"Loaded {len(questions):,} questions.")

    pipeline = Task1Pipeline(
        corpus=corpus,
        k1=cfg.retrieval.bm25_k1,
        b=cfg.retrieval.bm25_b,
        top_k=args.top_k,
        recall_k_list=cfg.evaluation.recall_k_list,
    )

    results = pipeline.run(questions, show_progress=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Task 1 Results Summary:")
    print("=" * 50)
    for metric, val in results["metrics"].items():
        print(f"  {metric:25s}: {val:.2f}%")
    print(f"\nSaved detailed predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
