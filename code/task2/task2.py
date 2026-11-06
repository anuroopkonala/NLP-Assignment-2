import os
import sys
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import (
    load_jsonl,
    save_json,
    compute_exact_match,
    compute_f1,
    compute_recall_at_k,
    truncate_tokens
)


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


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


def run_task2(corpus_path="data/nq_open_n300_seed42.jsonl",
              questions_path="data/sampled_questions.jsonl",
              embeddings_cache="data/dense_embeddings_45k.npy",
              output_path="data/results/task2_results.json",
              model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print("=" * 60)
    print("Task 2: Dense Retrieval-Augmented Generation (RAG)")
    print("=" * 60)

    corpus = load_jsonl(corpus_path)
    questions = load_jsonl(questions_path)
    print(f"Loaded {len(corpus)} passages and {len(questions)} questions.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    if os.path.exists(embeddings_cache):
        print(f"Loading cached embeddings from {embeddings_cache}...")
        corpus_embeddings = np.load(embeddings_cache)
    else:
        print("Encoding corpus passages...")
        texts = [p["text"] for p in corpus]
        corpus_embeddings = embedder.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        np.save(embeddings_cache, corpus_embeddings)

    faiss_index = build_faiss_index(corpus_embeddings)

    print("Encoding questions...")
    q_texts = [q["question"] for q in questions]
    query_embeddings = embedder.encode(
        q_texts,
        batch_size=64,
        normalize_embeddings=True
    ).astype(np.float32)

    _, top_indices = faiss_index.search(query_embeddings, 10)

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

    for i, item in enumerate(tqdm(questions, desc="Task 2")):
        q_text = item["question"]
        gold_answers = item["answer"]
        retrieved_passages = [corpus[idx]["text"] for idx in top_indices[i]]

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

        em_scores.append(compute_exact_match(clean_pred, gold_answers))
        f1_scores.append(compute_f1(clean_pred, gold_answers))
        r1.append(compute_recall_at_k(retrieved_passages, gold_answers, 1))
        r3.append(compute_recall_at_k(retrieved_passages, gold_answers, 3))
        r5.append(compute_recall_at_k(retrieved_passages, gold_answers, 5))
        r10.append(compute_recall_at_k(retrieved_passages, gold_answers, 10))

    results = {
        "dataset": "google-research-datasets/nq_open",
        "sample_size": len(questions),
        "corpus_size": len(corpus),
        "task": "Task 2 - Dense Retrieval-Augmented Generation (RAG)",
        "parameters": {
            "retriever": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_metric": "cosine (IndexFlatIP with normalized vectors)",
            "top_k_retrieval": 10,
            "top_k_answering": 5,
            "passage_max_tokens": 200,
            "generator": model_name,
            "max_new_tokens": 64,
            "temperature": 0.0,
            "device": device
        },
        "metrics": {
            "exact_match": round(float(np.mean(em_scores)), 4),
            "token_f1": round(float(np.mean(f1_scores)), 4),
            "recall@1": round(float(np.mean(r1)), 4),
            "recall@3": round(float(np.mean(r3)), 4),
            "recall@5": round(float(np.mean(r5)), 4),
            "recall@10": round(float(np.mean(r10)), 4)
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(results, output_path)

    print("\nTask 2 Results:")
    for k, v in results["metrics"].items():
        print(f"  {k:<12}: {v * 100:.2f}%")
    return results


if __name__ == "__main__":
    run_task2()
