#!/usr/bin/env python3
"""
Consolidates and visualizes evaluation metrics across Tasks 1, 2, and 3:
- Computes Exact Match, Token F1, and Recall@1, 3, 5, 10
- Performs groundedness threshold sweeps
- Exports summary CSV tables and comparative plots
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from open_domain_qa.data.dataset import load_jsonl
from open_domain_qa.evaluation.analysis import analyze_pipeline_errors
from open_domain_qa.evaluation.metrics import evaluate_qa_predictions
from open_domain_qa.evaluation.recall import evaluate_retrieval_recall
from open_domain_qa.verification.groundedness import GroundednessVerifier


def load_task_predictions(task_idx: int) -> List[Dict[str, Any]]:
    """Loads prediction records from jsonl or json result files."""
    candidates = [
        f"outputs/task{task_idx}/task{task_idx}_predictions.jsonl",
        f"data/results/task{task_idx}_predictions.jsonl",
    ]
    for c in candidates:
        if os.path.exists(c):
            return load_jsonl(c)

    # Fallback to json with embedded predictions
    json_candidates = [
        f"outputs/task{task_idx}/task{task_idx}_results.json",
        f"data/results/task{task_idx}_results.json",
    ]
    for jc in json_candidates:
        if os.path.exists(jc):
            with open(jc, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "predictions" in data:
                    return data["predictions"]
    return []


def load_task_summary(task_idx: int) -> Optional[Dict[str, Any]]:
    """Loads precomputed summary metrics if available."""
    candidates = [
        f"outputs/task{task_idx}/task{task_idx}_results.json",
        f"data/results/task{task_idx}_results.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate and consolidate open-domain QA results.")
    parser.add_argument("--output_csv", type=str, default="outputs/evaluation/consolidated_results.csv")
    parser.add_argument("--figures_dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    print("=" * 70)
    print("Consolidating Evaluation Metrics Across Tasks 1, 2, and 3")
    print("=" * 70)

    task_configs = [
        (1, "Task 1 (BM25 + Extractive)"),
        (2, "Task 2 (Dense RAG Zero-Shot)"),
        (3, "Task 3 (Hybrid RRF + Groundedness)"),
    ]

    rows = []
    task_predictions_map = {}

    print("\n" + "-" * 90)
    print(f"{'Method / Task':<36} | {'EM (%)':<8} | {'F1 (%)':<8} | {'R@1 (%)':<8} | {'R@3 (%)':<8} | {'R@5 (%)':<8} | {'R@10 (%)':<8}")
    print("-" * 90)

    for idx, name in task_configs:
        preds = load_task_predictions(idx)
        summary = load_task_summary(idx)
        task_predictions_map[name] = preds

        em = 0.0
        f1 = 0.0
        r1, r3, r5, r10 = 0.0, 0.0, 0.0, 0.0

        if summary and "metrics" in summary:
            m = summary["metrics"]
            em = m.get("exact_match", 0.0)
            f1 = m.get("token_f1", m.get("f1", 0.0))
            r1 = m.get("recall@1", 0.0)
            r3 = m.get("recall@3", 0.0)
            r5 = m.get("recall@5", 0.0)
            r10 = m.get("recall@10", 0.0)

            # Convert fractions to percentages if needed
            if 0.0 < em <= 1.0:
                em *= 100.0
            if 0.0 < f1 <= 1.0:
                f1 *= 100.0
            if 0.0 < r1 <= 1.0:
                r1 *= 100.0
                r3 *= 100.0
                r5 *= 100.0
                r10 *= 100.0
        elif preds:
            qa_m = evaluate_qa_predictions(preds)
            em = qa_m["exact_match"]
            f1 = qa_m["f1"]
            ret_m = evaluate_retrieval_recall(preds, k_values=[1, 3, 5, 10])
            r1 = ret_m["recall@1"]
            r3 = ret_m["recall@3"]
            r5 = ret_m["recall@5"]
            r10 = ret_m["recall@10"]

        row = {
            "Task": name,
            "Exact Match (%)": round(em, 2),
            "Token F1 (%)": round(f1, 2),
            "Recall@1 (%)": round(r1, 2),
            "Recall@3 (%)": round(r3, 2),
            "Recall@5 (%)": round(r5, 2),
            "Recall@10 (%)": round(r10, 2),
        }
        rows.append(row)
        print(f"{name:<36} | {row['Exact Match (%)']:<8.2f} | {row['Token F1 (%)']:<8.2f} | {row['Recall@1 (%)']:<8.2f} | {row['Recall@3 (%)']:<8.2f} | {row['Recall@5 (%)']:<8.2f} | {row['Recall@10 (%)']:<8.2f}")

    print("-" * 90)

    # Write summary CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if rows:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nConsolidated metrics saved to: {args.output_csv}")

    # Groundedness sweep for Task 3
    task3_preds = task_predictions_map.get("Task 3 (Hybrid RRF + Groundedness)", [])
    if task3_preds:
        sweep = GroundednessVerifier.evaluate_threshold_sweep(task3_preds)
        sweep_path = "outputs/evaluation/groundedness_sweep.json"
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(sweep, f, indent=2)
        print(f"Groundedness threshold sweep saved to: {sweep_path}")

    # Plot performance figures
    os.makedirs(args.figures_dir, exist_ok=True)
    if rows:
        # Retrieval recall curves
        plt.figure(figsize=(8, 5))
        k_vals = [1, 3, 5, 10]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for idx, row in enumerate(rows):
            r_vals = [row["Recall@1 (%)"], row["Recall@3 (%)"], row["Recall@5 (%)"], row["Recall@10 (%)"]]
            plt.plot(k_vals, r_vals, marker="o", linewidth=2, label=row["Task"], color=colors[idx % len(colors)])
        plt.xlabel("Passage Rank Cutoff (k)")
        plt.ylabel("Recall@k (%)")
        plt.title("Retrieval Recall@k Comparison Across Tasks")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        ret_fig = os.path.join(args.figures_dir, "retrieval_recall_comparison.png")
        plt.savefig(ret_fig, dpi=300)
        plt.close()
        print(f"Saved figure: {ret_fig}")

        # QA Exact Match and Token F1 bars
        plt.figure(figsize=(8, 5))
        task_labels = ["Task 1\n(BM25+Extract)", "Task 2\n(Dense RAG)", "Task 3\n(Hybrid RRF)"]
        em_vals = [r["Exact Match (%)"] for r in rows]
        f1_vals = [r["Token F1 (%)"] for r in rows]
        x = range(len(rows))
        width = 0.35
        plt.bar([p - width/2 for p in x], em_vals, width=width, label="Exact Match", color="#2b5c8f")
        plt.bar([p + width/2 for p in x], f1_vals, width=width, label="Token F1", color="#d95f02")
        plt.xticks(x, task_labels)
        plt.ylabel("Score (%)")
        plt.title("QA Performance Comparison (Exact Match & Token F1)")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        qa_fig = os.path.join(args.figures_dir, "qa_em_f1_comparison.png")
        plt.savefig(qa_fig, dpi=300)
        plt.close()
        print(f"Saved figure: {qa_fig}")

    print("Evaluation and visualization completed successfully.")


if __name__ == "__main__":
    main()
