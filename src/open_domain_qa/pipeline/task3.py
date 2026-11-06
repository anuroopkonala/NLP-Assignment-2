"""
Task 3 Pipeline: Hybrid Retrieval (BM25 + Dense RRF) with Groundedness Verification.
Combines BM25 and dense rankings via Reciprocal Rank Fusion (c=60),
generates answers on fused context, and verifies evidence grounding via content overlap.
"""

from typing import Any, Dict, List, Optional
from tqdm import tqdm

from open_domain_qa.answering.generator import MockRAGGenerator, RAGGenerator
from open_domain_qa.evaluation.metrics import evaluate_qa_predictions
from open_domain_qa.evaluation.recall import evaluate_retrieval_recall
from open_domain_qa.retrieval.bm25 import BM25Retriever
from open_domain_qa.retrieval.dense import DenseRetriever
from open_domain_qa.retrieval.hybrid import HybridRetriever
from open_domain_qa.verification.groundedness import GroundednessVerifier


class Task3Pipeline:
    """End-to-end pipeline for Task 3."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        generator_instance: Optional[Any] = None,
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cached_embeddings_path: Optional[str] = None,
        rrf_constant: int = 60,
        groundedness_threshold: float = 0.5,
        top_k: int = 5,
        max_tokens_per_passage: int = 200,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        recall_k_list: Optional[List[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.corpus = corpus
        self.top_k = top_k
        self.max_tokens_per_passage = max_tokens_per_passage
        self.recall_k_list = recall_k_list or [1, 3, 5, 10]
        self.rrf_constant = rrf_constant
        self.groundedness_threshold = groundedness_threshold

        if bm25_retriever is not None:
            self.bm25 = bm25_retriever
        else:
            self.bm25 = BM25Retriever(corpus)

        if dense_retriever is not None:
            self.dense = dense_retriever
        else:
            self.dense = DenseRetriever(corpus=corpus, model_name=encoder_model_name, device=device)
            if cached_embeddings_path:
                import os
                if os.path.exists(cached_embeddings_path):
                    self.dense.load_embeddings(cached_embeddings_path)
                else:
                    self.dense.index_corpus()
                    self.dense.save_embeddings(cached_embeddings_path)
            else:
                self.dense.index_corpus()

        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            rrf_constant=rrf_constant,
        )

        if generator_instance is not None:
            self.generator = generator_instance
        else:
            self.generator = RAGGenerator(
                model_name=generator_model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        self.verifier = GroundednessVerifier(threshold=groundedness_threshold)

    def run(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Runs Task 3 across question instances."""
        predictions = []
        iterator = tqdm(questions, desc="Task 3 (Hybrid + Groundedness)", disable=not show_progress)
        max_k = max(self.recall_k_list + [self.top_k])

        for q in iterator:
            query_text = q.get("question", "")
            gts = q.get("answers", q.get("answer", []))
            if isinstance(gts, str):
                gts = [gts]
            qid = q.get("question_id", q.get("id", ""))

            retrieved = self.hybrid_retriever.retrieve(
                query_text,
                top_k=max_k,
                candidate_pool_size=50,
            )
            context_passages = [p["text"] for p in retrieved[:self.top_k]]

            predicted_answer = self.generator.generate_answer(
                question=query_text,
                context_passages=context_passages,
                max_tokens_per_passage=self.max_tokens_per_passage,
                few_shot_examples=few_shot_examples,
            )

            verification_result = self.verifier.verify(
                answer=predicted_answer,
                context_passages=context_passages,
            )

            predictions.append({
                "question_id": qid,
                "question": query_text,
                "answers": gts,
                "prediction": predicted_answer,
                "grounded_answer": verification_result["final_answer"],
                "lexical_overlap": verification_result["overlap_score"],
                "is_grounded": verification_result["is_grounded"],
                "retrieved_passages": [p["text"] for p in retrieved[:max_k]],
                "retrieved_metadata": [
                    {
                        "doc_id": p["doc_id"],
                        "score": p["score"],
                        "title": p["title"],
                        "bm25_rank": p.get("bm25_rank"),
                        "dense_rank": p.get("dense_rank"),
                    }
                    for p in retrieved[:max_k]
                ],
            })

        qa_metrics = evaluate_qa_predictions(predictions)
        retrieval_metrics = evaluate_retrieval_recall(predictions, k_values=self.recall_k_list)

        overlaps = [p["lexical_overlap"] for p in predictions]
        grounded_count = sum(1 for p in predictions if p["is_grounded"])
        n = len(predictions)

        results = {
            "task": "Task 3: Hybrid Retrieval with Groundedness Verification",
            "metrics": {
                "exact_match": qa_metrics["exact_match"],
                "f1": qa_metrics["f1"],
                **retrieval_metrics,
                "mean_lexical_overlap": round(float(sum(overlaps) / n), 4) if n > 0 else 0.0,
                "grounded_ratio": round(float(grounded_count / n), 4) if n > 0 else 0.0,
                "groundedness_threshold": self.groundedness_threshold,
                "rrf_constant": self.rrf_constant,
            },
            "num_questions": n,
            "predictions": predictions,
        }
        return results
