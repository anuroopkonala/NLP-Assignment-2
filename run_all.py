import os
import sys
import argparse
import json


def print_ablation_table():
    summary_path = os.path.join("data", "results", "refinements_summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print(" Ablation Study: Model Variants & Parameter Sensitivity (300 NQ Questions)")
    print("=" * 80)

    t1 = data.get("task1_bm25_ablation", {})
    t1_txt = t1.get("text_only", {})
    t1_ttl = t1.get("title_aware", {})
    print("1. Task 1: Lexical Representation (BM25Okapi, k1=1.5, b=0.75)")
    print(f"   - Text-Only BM25    : EM = {t1_txt.get('exact_match',0)*100:.2f}%, F1 = {t1_txt.get('token_f1',0)*100:.2f}%, R@1 = {t1_txt.get('recall@1',0)*100:.2f}%, R@10 = {t1_txt.get('recall@10',0)*100:.2f}%")
    print(f"   - Title-Aware BM25  : EM = {t1_ttl.get('exact_match',0)*100:.2f}%, F1 = {t1_ttl.get('token_f1',0)*100:.2f}%, R@1 = {t1_ttl.get('recall@1',0)*100:.2f}%, R@10 = {t1_ttl.get('recall@10',0)*100:.2f}%")

    t2 = data.get("task2_generator_ablation", {})
    t2_zero = t2.get("zero_shot_instruction", {})
    t2_few = t2.get("few_shot_exemplars", {})
    print("\n2. Task 2: Reader Prompting Strategy (Qwen2.5-1.5B-Instruct, max_new_tokens=64)")
    print(f"   - Zero-Shot Greedy  : EM = {t2_zero.get('exact_match',0)*100:.2f}%, F1 = {t2_zero.get('token_f1',0)*100:.2f}%")
    print(f"   - Few-Shot (3-Shot) : EM = {t2_few.get('exact_match',0)*100:.2f}%, F1 = {t2_few.get('token_f1',0)*100:.2f}%")

    t3 = data.get("task3_hybrid_fusion_ablation", {})
    t3_txt = t3.get("text_bm25_plus_dense_rrf", {})
    t3_ttl = t3.get("title_bm25_plus_dense_rrf", {})
    print("\n3. Task 3: Hybrid Retrieval Signal Fusion (RRF constant c=60)")
    print(f"   - Text BM25 + Dense RRF  : R@1 = {t3_txt.get('recall@1',0)*100:.2f}%, R@5 = {t3_txt.get('recall@5',0)*100:.2f}%, R@10 = {t3_txt.get('recall@10',0)*100:.2f}%")
    print(f"   - Title BM25 + Dense RRF : R@1 = {t3_ttl.get('recall@1',0)*100:.2f}%, R@5 = {t3_ttl.get('recall@5',0)*100:.2f}%, R@10 = {t3_ttl.get('recall@10',0)*100:.2f}%")

    sweep = data.get("task3_groundedness_threshold_sweep", [])
    print("\n4. Task 3: Groundedness Verification Sensitivity (Threshold tau)")
    print(f"   {'Threshold (tau)':<16} | {'Correct High-Overlap':<22} | {'Incorrect Low-Overlap':<23} | {'Groundedness Acc':<16}")
    print("   " + "-" * 81)
    for row in sweep:
        print(f"   {row['tau']:<16.1f} | {row['correct_high_overlap_pct']:6.2f}%               | {row['incorrect_low_overlap_pct']:6.2f}%                | {row['groundedness_accuracy_pct']:6.2f}%")
    print("=" * 80 + "\n")


def print_comparison_table():
    results_dir = os.path.join("data", "results")
    tasks = [
        ("Task 1 (Classical BM25)", "task1_results.json"),
        ("Task 2 (Dense RAG)", "task2_results.json"),
        ("Task 3 (Hybrid RRF)", "task3_results.json"),
    ]

    print("\n" + "=" * 90)
    print(" OPEN-DOMAIN QA: EVALUATION SUMMARY (NATURAL QUESTIONS OPEN, N=300)")
    print("=" * 90)
    print(f"{'Method / Task':<26} | {'EM (%)':<8} | {'F1 (%)':<8} | {'R@1 (%)':<9} | {'R@3 (%)':<9} | {'R@5 (%)':<9} | {'R@10 (%)':<9}")
    print("-" * 90)

    for label, filename in tasks:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                res = json.load(f)["metrics"]
            em = res["exact_match"] * 100
            f1 = res["token_f1"] * 100
            r1 = res["recall@1"] * 100
            r3 = res["recall@3"] * 100
            r5 = res["recall@5"] * 100
            r10 = res["recall@10"] * 100
            print(f"{label:<26} | {em:<8.2f} | {f1:<8.2f} | {r1:<9.2f} | {r3:<9.2f} | {r5:<9.2f} | {r10:<9.2f}")
        else:
            print(f"{label:<26} | [Results not found at {filepath}]")

    print("=" * 90)

    t3_path = os.path.join(results_dir, "task3_results.json")
    if os.path.exists(t3_path):
        with open(t3_path, "r", encoding="utf-8") as f:
            t3 = json.load(f)["metrics"]
        print("\nTask 3 Groundedness Verification (tau = 0.5):")
        print(f"  - Verification Decision Accuracy           : {t3.get('groundedness_check_accuracy', 0.0) * 100:.2f}%")
        print(f"  - Correct Answers with High Overlap (>=0.5): {t3.get('frac_correct_high_overlap', 0.0) * 100:.2f}%")
        print(f"  - Incorrect Answers with Low Overlap (<0.5): {t3.get('frac_incorrect_low_overlap', 0.0) * 100:.2f}%")
        print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Runner for NLP Assignment 2 (Group 29)")
    parser.add_argument("--task", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--eval_only", action="store_true", help="Display summary table from saved results")
    parser.add_argument("--ablation", action="store_true", help="Display ablation study table")
    args = parser.parse_args()

    if args.ablation:
        print_ablation_table()
        return

    if args.eval_only:
        print_comparison_table()
        return

    sys.path.insert(0, os.path.abspath("code"))

    if args.task in ["1", "all"]:
        from task1.task1 import run_task1
        run_task1()

    if args.task in ["2", "all"]:
        from task2.task2 import run_task2
        run_task2()

    if args.task in ["3", "all"]:
        from task3.task3 import run_task3
        run_task3()

    print_comparison_table()


if __name__ == "__main__":
    main()
