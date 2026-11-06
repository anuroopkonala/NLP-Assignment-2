import os
import sys
import re
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import (
    load_jsonl,
    save_json,
    compute_exact_match,
    compute_f1,
    compute_recall_at_k
)

for res in ["punkt", "averaged_perceptron_tagger", "punkt_tab", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

NP_GRAMMAR = r"NP: {<DT|PRP\$>?<JJ.*|VBG|VBN>*<NN.*>+}"
NP_PARSER = nltk.RegexpParser(NP_GRAMMAR)
NUM_REGEX = re.compile(r"\b(?:\d{1,4}(?:,\d{3})*(?:\.\d+)?|\d+(?:st|nd|rd|th)?)\b")


def extract_longest_candidate(sentence):
    if not sentence or not sentence.strip():
        return ""

    candidates = []
    try:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        tree = NP_PARSER.parse(pos_tags)
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                candidates.append(" ".join(leaf[0] for leaf in subtree.leaves()))
    except Exception:
        pass

    candidates.extend(NUM_REGEX.findall(sentence))
    if candidates:
        return max(candidates, key=len).strip()
    return sentence.strip()


def select_best_sentence(question, passage_text):
    try:
        sentences = nltk.sent_tokenize(passage_text)
    except Exception:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage_text) if s.strip()]

    if not sentences:
        return passage_text

    q_words = set(re.findall(r"\b\w+\b", question.lower()))
    best_sent = sentences[0]
    max_overlap = -1

    for sent in sentences:
        sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
        overlap = len(q_words & sent_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sent = sent

    return best_sent


def run_task1(corpus_path="data/nq_open_n300_seed42.jsonl",
              questions_path="data/sampled_questions.jsonl",
              output_path="data/results/task1_results.json"):
    print("=" * 60)
    print("Task 1: Classical BM25 Retrieval + Extractive QA")
    print("=" * 60)

    corpus = load_jsonl(corpus_path)
    questions = load_jsonl(questions_path)
    print(f"Loaded {len(corpus)} passages and {len(questions)} questions.")

    tokenized_corpus = [
        re.findall(r"\b\w+\b", f"{p.get('title', '')} {p['text']}".lower())
        for p in corpus
    ]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    em_scores, f1_scores = [], []
    r1, r3, r5, r10 = [], [], [], []

    for item in tqdm(questions, desc="Task 1"):
        q_text = item["question"]
        gold_answers = item["answer"]
        q_tokens = re.findall(r"\b\w+\b", q_text.lower())

        doc_scores = bm25.get_scores(q_tokens)
        top10_indices = np.argsort(doc_scores)[::-1][:10]
        retrieved_passages = [corpus[idx]["text"] for idx in top10_indices]

        top_passage = retrieved_passages[0] if retrieved_passages else ""
        best_sentence = select_best_sentence(q_text, top_passage)
        pred_answer = extract_longest_candidate(best_sentence)

        em_scores.append(compute_exact_match(pred_answer, gold_answers))
        f1_scores.append(compute_f1(pred_answer, gold_answers))
        r1.append(compute_recall_at_k(retrieved_passages, gold_answers, 1))
        r3.append(compute_recall_at_k(retrieved_passages, gold_answers, 3))
        r5.append(compute_recall_at_k(retrieved_passages, gold_answers, 5))
        r10.append(compute_recall_at_k(retrieved_passages, gold_answers, 10))

    results = {
        "dataset": "google-research-datasets/nq_open",
        "sample_size": len(questions),
        "corpus_size": len(corpus),
        "task": "Task 1 - Classical Lexical Retrieve-and-Extract",
        "parameters": {
            "retriever": "BM25Okapi",
            "k1": 1.5,
            "b": 0.75,
            "top_k_retrieval": 10,
            "top_k_answering": 5,
            "extractor": "longest_noun_phrase_or_number_span"
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

    print("\nTask 1 Results:")
    for k, v in results["metrics"].items():
        print(f"  {k:<12}: {v * 100:.2f}%")
    return results


if __name__ == "__main__":
    run_task1()
