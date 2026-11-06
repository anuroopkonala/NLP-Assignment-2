import os
import sys
from collections import defaultdict
import numpy as np
import torch
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import (
    load_jsonl,
    save_json,
    compute_exact_match,
    compute_f1,
    compute_recall_at_k,
    truncate_tokens,
    compute_lexical_overlap,
    compute_reciprocal_rank_fusion
)


def format_rag_prompt(question, context_passages):
    formatted = [f"[{i+1}] {truncate_tokens(p, max_tokens=200)}" for i, p in enumerate(context_passages)]
    context_str = "\n\n".join(formatted)
    return (
        "Answer the following question directly and concisely based only on the provided context passages. "
        "Provide only the short answer.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def run_task3(corpus_path="data/nq_open_n300_seed42.jsonl",
              questions_path="data/sampled_questions.jsonl",
              embeddings_cache="data/dense_embeddings_45k.npy",
              output_path="data/results/task3_results.json",
              model_name="Qwen/Qwen2.5-1.5B-Instruct",
              overlap_threshold=0.5):
    print("=" * 60)
    print("Task 3: Hybrid Retrieval (RRF) + Groundedness Verification")
    print("=" * 60)

    corpus = load_jsonl(corpus_path)
    questions = load_jsonl(questions_path)
    print(f"Loaded {len(corpus)} passages and {len(questions)} questions.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    print("Building BM25 index...")
    tokenized_corpus = [p["text"].lower().split() for p in corpus]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    print("Loading dense encoder and embeddings...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    if os.path.exists(embeddings_cache):
        corpus_embeddings = np.load(embeddings_cache)
    else:
        texts = [p["text"] for p in corpus]
        corpus_embeddings = embedder.encode(texts, batch_size=256, normalize_embeddings=True)
        np.save(embeddings_cache, corpus_embeddings)

    dim = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(corpus_embeddings.astype(np.float32))

    print("Retrieving dense candidates...")
    q_texts = [q["question"] for q in questions]
    query_embeddings = embedder.encode(q_texts, batch_size=64, normalize_embeddings=True).astype(np.float32)
    _, dense_top50_all = faiss_index.search(query_embeddings, 50)

    print(f"Loading generator: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    em_scores, f1_scores = [], []
    r1, r3, r5, r10 = [], [], [], []

    num_correct = 0
    correct_with_high_overlap = 0
    num_incorrect = 0
    incorrect_with_low_overlap = 0
    correct_verification_decisions = 0

    for i, item in enumerate(tqdm(questions, desc="Task 3")):
        q_text = item["question"]
        gold_answers = item["answer"]

        q_tokens = q_text.lower().split()
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_top50 = np.argsort(bm25_scores)[::-1][:50].tolist()
        dense_top50 = dense_top50_all[i].tolist()

        fused_indices = compute_reciprocal_rank_fusion(bm25_top50, dense_top50, constant=60)
        retrieved_passages = [corpus[idx]["text"] for idx in fused_indices[:10]]

        prompt = format_rag_prompt(q_text, retrieved_passages[:5])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        clean_pred = generated.split("\n")[0].strip()

        combined_context = " ".join(retrieved_passages[:5])
        overlap_score = compute_lexical_overlap(clean_pred, combined_context)
        is_grounded = overlap_score >= overlap_threshold

        is_em = compute_exact_match(clean_pred, gold_answers)
        f1_val = compute_f1(clean_pred, gold_answers)

        em_scores.append(is_em)
        f1_scores.append(f1_val)
        r1.append(compute_recall_at_k(retrieved_passages, gold_answers, 1))
        r3.append(compute_recall_at_k(retrieved_passages, gold_answers, 3))
        r5.append(compute_recall_at_k(retrieved_passages, gold_answers, 5))
        r10.append(compute_recall_at_k(retrieved_passages, gold_answers, 10))

        if is_em:
            num_correct += 1
            if is_grounded:
                correct_with_high_overlap += 1
                correct_verification_decisions += 1
        else:
            num_incorrect += 1
            if not is_grounded:
                incorrect_with_low_overlap += 1
                correct_verification_decisions += 1

    frac_correct_high_overlap = (
        round(correct_with_high_overlap / num_correct, 4) if num_correct > 0 else 0.0
    )
    frac_incorrect_low_overlap = (
        round(incorrect_with_low_overlap / num_incorrect, 4) if num_incorrect > 0 else 0.0
    )
    groundedness_accuracy = round(correct_verification_decisions / len(questions), 4)

    results = {
        "dataset": "google-research-datasets/nq_open",
        "sample_size": len(questions),
        "corpus_size": len(corpus),
        "task": "Task 3 - Hybrid Retrieval with Groundedness Verification",
        "parameters": {
            "fusion_method": "Reciprocal Rank Fusion (RRF)",
            "rrf_constant": 60,
            "top_k_retrieval": 10,
            "top_k_answering": 5,
            "passage_max_tokens": 200,
            "generator": model_name,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "overlap_threshold": overlap_threshold,
            "device": device
        },
        "metrics": {
            "exact_match": round(float(np.mean(em_scores)), 4),
            "token_f1": round(float(np.mean(f1_scores)), 4),
            "recall@1": round(float(np.mean(r1)), 4),
            "recall@3": round(float(np.mean(r3)), 4),
            "recall@5": round(float(np.mean(r5)), 4),
            "recall@10": round(float(np.mean(r10)), 4),
            "groundedness_check_accuracy": groundedness_accuracy,
            "frac_correct_high_overlap": frac_correct_high_overlap,
            "frac_incorrect_low_overlap": frac_incorrect_low_overlap
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(results, output_path)

    print("\nTask 3 Results:")
    for k, v in results["metrics"].items():
        print(f"  {k:<30}: {v * 100:.2f}%" if "frac" in k or "exact" in k or "token" in k or "recall" in k or "accuracy" in k else f"  {k}: {v}")
    return results


if __name__ == "__main__":
    run_task3()
