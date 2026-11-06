"""
Unified pipeline runner orchestrating Tasks 1, 2, 3, evaluations, and ablations.
"""

import json
import os
from typing import Any, Dict, List, Optional

from open_domain_qa.config import PipelineConfig, load_config
from open_domain_qa.corpus.wikipedia import WikipediaCorpus
from open_domain_qa.data.dataset import NaturalQuestionsDataset
from open_domain_qa.evaluation.analysis import analyze_pipeline_errors
from open_domain_qa.evaluation.metrics import evaluate_qa_predictions
from open_domain_qa.evaluation.recall import evaluate_retrieval_recall
from open_domain_qa.pipeline.task1 import Task1Pipeline
from open_domain_qa.pipeline.task2 import Task2Pipeline
from open_domain_qa.pipeline.task3 import Task3Pipeline
from open_domain_qa.verification.groundedness import GroundednessVerifier


class PipelineRunner:
    """Master orchestrator for running, evaluating, and analyzing open-domain QA tasks."""

    def __init__(self, config: Optional[PipelineConfig] = None, config_path: Optional[str] = None) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = load_config()

        self._corpus_data: Optional[List[Dict[str, Any]]] = None
        self._questions_data: Optional[List[Dict[str, Any]]] = None

    def get_corpus(self) -> List[Dict[str, Any]]:
        """Loads or returns cached Wikipedia corpus."""
        if self._corpus_data is None:
            path = self.config.corpus.output_path
            if not os.path.exists(path):
                alt_path = "data/nq_open_n300_seed42.jsonl"
                if os.path.exists(alt_path):
                    path = alt_path
            corpus = WikipediaCorpus(corpus_path=path)
            self._corpus_data = corpus.load()
        return self._corpus_data

    def get_questions(self) -> List[Dict[str, Any]]:
        """Loads or returns cached validation questions."""
        if self._questions_data is None:
            path = self.config.dataset.sampled_path
            if not os.path.exists(path):
                alt_path = "data/sampled_questions.jsonl"
                if os.path.exists(alt_path):
                    path = alt_path
            nq = NaturalQuestionsDataset(sampled_path=path)
            self._questions_data = nq.load_sampled()
        return self._questions_data

    def run_eval_only(
        self,
        results_dir: str = "data/results",
        output_summary_path: str = "outputs/evaluation/consolidated_results.csv",
    ) -> Dict[str, Any]:
        """Loads precomputed task predictions, verifies metrics, and outputs summary tables."""
        print("Running in fast evaluation mode (--eval_only) on saved task outputs...")

        task_files = {
            "Task 1 (BM25 + Extractive)": os.path.join(results_dir, "task1_results.json"),
            "Task 2 (Dense MiniLM + RAG)": os.path.join(results_dir, "task2_results.json"),
            "Task 3 (Hybrid RRF + Groundedness)": os.path.join(results_dir, "task3_results.json"),
        }

        all_results = {}
        for task_name, path in task_files.items():
            if not os.path.exists(path):
                base = os.path.basename(path)
                alt = os.path.join("outputs", base.replace("_results.json", ""), base)
                if os.path.exists(alt):
                    path = alt

            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_results[task_name] = data
            else:
                print(f"Notice: Result file for {task_name} not found at {path}")

        print(f"Loaded {len(all_results)} evaluated tasks successfully.")
        return all_results

    def run_task(self, task_num: int) -> Dict[str, Any]:
        """Executes a specific task pipeline (1, 2, or 3)."""
        corpus = self.get_corpus()
        questions = self.get_questions()

        if task_num == 1:
            pipeline = Task1Pipeline(
                corpus=corpus,
                k1=self.config.retrieval.bm25_k1,
                b=self.config.retrieval.bm25_b,
                top_k=self.config.retrieval.bm25_top_k,
                recall_k_list=self.config.evaluation.recall_k_list,
            )
            return pipeline.run(questions)

        elif task_num == 2:
            pipeline = Task2Pipeline(
                corpus=corpus,
                encoder_model_name=self.config.retrieval.dense_model_name,
                generator_model_name=self.config.generation.model_name,
                cached_embeddings_path=self.config.corpus.embeddings_path,
                top_k=self.config.retrieval.dense_top_k,
                max_tokens_per_passage=self.config.generation.max_tokens_per_passage,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                recall_k_list=self.config.evaluation.recall_k_list,
                device=self.config.generation.device,
            )
            return pipeline.run(questions)

        elif task_num == 3:
            pipeline = Task3Pipeline(
                corpus=corpus,
                encoder_model_name=self.config.retrieval.dense_model_name,
                generator_model_name=self.config.generation.model_name,
                cached_embeddings_path=self.config.corpus.embeddings_path,
                rrf_constant=self.config.retrieval.rrf_constant,
                groundedness_threshold=self.config.verification.groundedness_threshold,
                top_k=self.config.retrieval.hybrid_top_k,
                max_tokens_per_passage=self.config.generation.max_tokens_per_passage,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                recall_k_list=self.config.evaluation.recall_k_list,
                device=self.config.generation.device,
            )
            return pipeline.run(questions)

        else:
            raise ValueError(f"Invalid task number {task_num}. Choose from 1, 2, or 3.")
