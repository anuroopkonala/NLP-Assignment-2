"""
Task 2 Pipeline: Dense Retrieval-Augmented Generation (RAG).
Encodes passages with all-MiniLM-L6-v2, retrieves via FAISS IndexFlatIP cosine similarity,
and queries Qwen2.5-1.5B-Instruct using greedy decoding.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from tqdm import tqdm

from open_domain_qa.answering.generator import MockRAGGenerator, RAGGenerator
from open_domain_qa.evaluation.metrics import evaluate_qa_predictions
from open_domain_qa.evaluation.recall import evaluate_retrieval_recall
from open_domain_qa.retrieval.dense import DenseRetriever


class Task2Pipeline:
    """End-to-end pipeline for Task 2."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cached_embeddings_path: Optional[str] = None,
        top_k: int = 5,
        max_tokens_per_passage: int = 200,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        recall_k_list: Optional[List[int]] = None,
        generator_instance: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.corpus = corpus
        self.top_k = top_k
        self.max_tokens_per_passage = max_tokens_per_passage
        self.recall_k_list = recall_k_list or [1, 3, 5, 10]

        self.retriever = DenseRetriever(
            corpus=corpus,
            model_name=encoder_model_name,
            device=device,
        )

        if cached_embeddings_path:
            import os
            if os.path.exists(cached_embeddings_path):
                self.retriever.load_embeddings(cached_embeddings_path)
            else:
                self.retriever.index_corpus()
                self.retriever.save_embeddings(cached_embeddings_path)
        else:
            self.retriever.index_corpus()

        if generator_instance is not None:
            self.generator = generator_instance
        else:
            self.generator = RAGGenerator(
                model_name=generator_model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def run(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Runs Task 2 across question instances."""
        predictions = []
        iterator = tqdm(questions, desc="Task 2 (Dense + RAG)", disable=not show_progress)
        max_k = max(self.recall_k_list + [self.top_k])

        for q in iterator:
            query_text = q.get("question", "")
            gts = q.get("answers", q.get("answer", []))
            if isinstance(gts, str):
                gts = [gts]
            qid = q.get("question_id", q.get("id", ""))

            retrieved = self.retriever.retrieve(query_text, top_k=max_k)
            context_passages = [p["text"] for p in retrieved[:self.top_k]]

            predicted_answer = self.generator.generate_answer(
                question=query_text,
                context_passages=context_passages,
                max_tokens_per_passage=self.max_tokens_per_passage,
                few_shot_examples=few_shot_examples,
            )

            predictions.append({
                "question_id": qid,
                "question": query_text,
                "answers": gts,
                "prediction": predicted_answer,
                "retrieved_passages": [p["text"] for p in retrieved[:max_k]],
                "retrieved_metadata": [
                    {"doc_id": p["doc_id"], "score": p["score"], "title": p["title"]}
                    for p in retrieved[:max_k]
                ],
            })

        qa_metrics = evaluate_qa_predictions(predictions)
        retrieval_metrics = evaluate_retrieval_recall(predictions, k_values=self.recall_k_list)

        results = {
            "task": "Task 2: Dense RAG",
            "metrics": {
                "exact_match": qa_metrics["exact_match"],
                "f1": qa_metrics["f1"],
                **retrieval_metrics,
            },
            "num_questions": len(questions),
            "predictions": predictions,
        }
        return results
