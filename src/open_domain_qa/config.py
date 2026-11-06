"""Configuration dataclasses and YAML loader for open-domain QA pipelines."""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional
import yaml


def get_project_root() -> str:
    """Returns absolute path to the repository root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_yaml_file(path: str) -> Dict[str, Any]:
    """Loads a YAML file safely if it exists, otherwise returns empty dict."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DatasetConfig:
    name: str = "google-research-datasets/nq_open"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    sampled_path: str = "data/samples/sampled_questions.jsonl"
    dev_path: str = "data/samples/dev_calibration_questions.jsonl"
    eval_sample_size: int = 300
    eval_seed: int = 42
    dev_sample_size: int = 100
    dev_seed: int = 100


@dataclass
class CorpusConfig:
    output_path: str = "data/wikipedia/nq_open_n300_seed42.jsonl"
    embeddings_path: str = "data/wikipedia/dense_embeddings_45k.npy"
    manifest_path: str = "data/wikipedia/corpus_manifest.json"
    max_tokens_per_passage: int = 200


@dataclass
class RetrievalConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_top_k: int = 5
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_top_k: int = 5
    hybrid_top_k: int = 5
    rrf_constant: int = 60
    batch_size: int = 256
    device: Optional[str] = None


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens_per_passage: int = 200
    max_new_tokens: int = 64
    temperature: float = 0.0
    device: Optional[str] = None


@dataclass
class VerificationConfig:
    groundedness_threshold: float = 0.5
    method: str = "lexical_content_word_overlap"


@dataclass
class EvaluationConfig:
    recall_k_list: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    results_dir: str = "outputs/evaluation"


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Loads configuration from YAML file or returns default parameters."""
    root = get_project_root()
    cfg = PipelineConfig()

    base_path = config_path if config_path else os.path.join(root, "configs", "default.yaml")
    if os.path.exists(base_path):
        data = load_yaml_file(base_path)

        paths = data.get("paths", {})
        if "wikipedia_corpus" in paths:
            cfg.corpus.output_path = paths["wikipedia_corpus"]
        if "dense_embeddings" in paths:
            cfg.corpus.embeddings_path = paths["dense_embeddings"]
        if "evaluation_samples" in paths:
            cfg.dataset.sampled_path = paths["evaluation_samples"]
        if "dev_calibration" in paths:
            cfg.dataset.dev_path = paths["dev_calibration"]
        if "corpus_manifest" in paths:
            cfg.corpus.manifest_path = paths["corpus_manifest"]

    return cfg


# Backward compatibility alias
ProjectConfig = PipelineConfig
