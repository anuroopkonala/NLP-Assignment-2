#!/usr/bin/env python3
"""
CLI runner for Task 3: Hybrid Retrieval with Groundedness Verification.
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
from open_domain_qa.pipeline.task3 import Task3Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Task 3: Hybrid RRF + Groundedness Verification.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus JSONL.")
    parser.add_argument("--questions_path", type=str, default=None, help="Path to evaluation questions JSONL.")
    parser.add_argument("--embeddings_path", type=str, default=None, help="Path to cached embeddings .npy.")
    parser.add_argument("--output_path", type=str, default="outputs/task3/task3_results.json", help="Path to save output JSON.")
    parser.add_argument("--rrf_constant", type=int, default=60, help="Reciprocal Rank Fusion constant c (default: 60).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Groundedness overlap threshold tau (default: 0.5).")
    parser.add_argument("--model_name", type=str, default=None, help="Generator model name.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu').")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print("Executing Task 3: Hybrid Retrieval (RRF) & Groundedness Verification")
    print("=" * 70)

    corpus_file = args.corpus_path or cfg.corpus.output_path
    if not os.path.exists(corpus_file) and os.path.exists("data/nq_open_n300_seed42.jsonl"):
        corpus_file = "data/nq_open_n300_seed42.jsonl"

    questions_file = args.questions_path or cfg.dataset.sampled_path
    if not os.path.exists(questions_file) and os.path.exists("data/sampled_questions.jsonl"):
        questions_file = "data/sampled_questions.jsonl"

    embeddings_file = args.embeddings_path or cfg.corpus.embeddings_path
    if not os.path.exists(embeddings_file) and os.path.exists("data/dense_embeddings_45k.npy"):
        embeddings_file = "data/dense_embeddings_45k.npy"

    generator_model = args.model_name or cfg.generation.model_name

    print(f"Loading corpus from: {corpus_file}")
    corpus = WikipediaCorpus(corpus_path=corpus_file).load()
    print(f"Loaded {len(corpus):,} passages.")

    print(f"Loading questions from: {questions_file}")
    questions = NaturalQuestionsDataset(sampled_path=questions_file).load_sampled()
    print(f"Loaded {len(questions):,} questions.")

    pipeline = Task3Pipeline(
        corpus=corpus,
        encoder_model_name=cfg.retrieval.dense_model_name,
        generator_model_name=generator_model,
        cached_embeddings_path=embeddings_file,
        rrf_constant=args.rrf_constant,
        groundedness_threshold=args.threshold,
        top_k=cfg.retrieval.hybrid_top_k,
        max_tokens_per_passage=cfg.generation.max_tokens_per_passage,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        recall_k_list=cfg.evaluation.recall_k_list,
        device=args.device or cfg.generation.device,
    )

    results = pipeline.run(questions, show_progress=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Task 3 Results Summary:")
    print("=" * 50)
    for metric, val in results["metrics"].items():
        if isinstance(val, float):
            print(f"  {metric:25s}: {val:.2f}%" if "overlap" not in metric and "ratio" not in metric else f"  {metric:25s}: {val:.4f}")
        else:
            print(f"  {metric:25s}: {val}")
    print(f"\nSaved detailed predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
