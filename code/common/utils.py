import json
import re
import string
from collections import Counter, defaultdict

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "for", "to", "and", "or",
    "is", "was", "are", "were", "it", "its", "that", "this", "by", "from",
    "with", "as", "be", "has", "had", "have"
}


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_answer(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def compute_exact_match(prediction, ground_truths):
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_gt and norm_pred == norm_gt:
            return True
    return False


def compute_f1(prediction, ground_truths):
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 1.0 if any(len(normalize_answer(gt).split()) == 0 for gt in ground_truths) else 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1

    return max_f1


def compute_recall_at_k(retrieved_passages, ground_truths, k):
    norm_gts = [normalize_answer(gt) for gt in ground_truths if normalize_answer(gt)]
    if not norm_gts:
        return False

    for passage in retrieved_passages[:k]:
        norm_passage = normalize_answer(passage)
        for gt in norm_gts:
            if gt in norm_passage:
                return True
    return False


def truncate_tokens(text, max_tokens=200):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def compute_lexical_overlap(answer, context):
    norm_ans_tokens = normalize_answer(answer).split()
    content_tokens = [t for t in norm_ans_tokens if t not in STOPWORDS]
    if not content_tokens:
        return 1.0

    context_tokens = set(normalize_answer(context).split())
    matched = sum(1 for t in content_tokens if t in context_tokens)
    return float(matched) / len(content_tokens)


def compute_reciprocal_rank_fusion(bm25_ranked_ids, dense_ranked_ids, constant=60):
    rrf_scores = defaultdict(float)
    for rank_idx, doc_id in enumerate(bm25_ranked_ids):
        rrf_scores[doc_id] += 1.0 / (constant + rank_idx + 1)
    for rank_idx, doc_id in enumerate(dense_ranked_ids):
        rrf_scores[doc_id] += 1.0 / (constant + rank_idx + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
