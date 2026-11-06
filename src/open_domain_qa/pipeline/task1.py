"""
Task 1 Pipeline: Classical Lexical Retrieve-and-Extract Baseline.
Indexes passages with BM25Okapi, selects the highest unigram overlap sentence,
and extracts the longest noun phrase chunk or numeric entity span.
"""

from typing import Any, Dict, List, Optional
from tqdm import tqdm

from open_domain_qa.answering.extraction import ExtractiveAnswerer
from open_domain_qa.evaluation.metrics import evaluate_qa_predictions
from open_domain_qa.evaluation.recall import evaluate_retrieval_recall
from open_domain_qa.retrieval.bm25 import BM25Retriever


class Task1Pipeline:
    """End-to-end pipeline for Task 1."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 5,
        recall_k_list: Optional[List[int]] = None,
    ) -> None:
        self.corpus = corpus
        self.top_k = top_k
        self.recall_k_list = recall_k_list or [1, 3, 5, 10]
        self.retriever = BM25Retriever(corpus, k1=k1, b=b)
        self.answerer = ExtractiveAnswerer(top_passages_to_consider=top_k)

    def run(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Runs Task 1 across question instances."""
        predictions = []
        iterator = tqdm(questions, desc="Task 1 (BM25 + Extractive)", disable=not show_progress)

        for q in iterator:
            query_text = q.get("question", "")
            gts = q.get("answers", q.get("answer", []))
            if isinstance(gts, str):
                gts = [gts]
            qid = q.get("question_id", q.get("id", ""))

            max_k = max(self.recall_k_list + [self.top_k])
            retrieved = self.retriever.retrieve(query_text, top_k=max_k)

            extraction_result = self.answerer.extract_answer(query_text, retrieved[:self.top_k])

            predictions.append({
                "question_id": qid,
                "question": query_text,
                "answers": gts,
                "prediction": extraction_result["prediction"],
                "selected_sentence": extraction_result["selected_sentence"],
                "passage_rank": extraction_result["passage_rank"],
                "retrieved_passages": [p["text"] for p in retrieved[:max_k]],
                "retrieved_metadata": [
                    {"doc_id": p["doc_id"], "score": p["score"], "title": p["title"]}
                    for p in retrieved[:max_k]
                ],
            })

        qa_metrics = evaluate_qa_predictions(predictions)
        retrieval_metrics = evaluate_retrieval_recall(predictions, k_values=self.recall_k_list)

        results = {
            "task": "Task 1: BM25 + Extractive Baseline",
            "metrics": {
                "exact_match": qa_metrics["exact_match"],
                "f1": qa_metrics["f1"],
                **retrieval_metrics,
            },
            "num_questions": len(questions),
            "predictions": predictions,
        }
        return results
