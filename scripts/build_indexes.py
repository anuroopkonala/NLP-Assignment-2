#!/usr/bin/env python3
"""
Index builder for lexical and dense retrieval:
- BM25Okapi inverted index (k1=1.5, b=0.75)
- all-MiniLM-L6-v2 embeddings (45,029 x 384) with FAISS IndexFlatIP
"""

import argparse
import os
import sys
import numpy as np

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from open_domain_qa.config import load_config
from open_domain_qa.corpus.wikipedia import WikipediaCorpus
from open_domain_qa.retrieval.bm25 import BM25Retriever
from open_domain_qa.retrieval.dense import DenseRetriever


def main():
    parser = argparse.ArgumentParser(description="Build or verify BM25 and Dense retrieval indexes.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--force_reencode", action="store_true", help="Re-encode dense embeddings from scratch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print("Building / Verifying Retrieval Indexes")
    print("=" * 70)

    corpus_path = cfg.corpus.output_path
    if not os.path.exists(corpus_path) and os.path.exists("data/nq_open_n300_seed42.jsonl"):
        corpus_path = "data/nq_open_n300_seed42.jsonl"

    corpus = WikipediaCorpus(corpus_path=corpus_path).load()
    print(f"Loaded {len(corpus):,} passages.")

    # 1. BM25 Verification
    print("Initializing BM25 index...")
    bm25 = BM25Retriever(corpus, k1=cfg.retrieval.bm25_k1, b=cfg.retrieval.bm25_b)
    test_hits = bm25.retrieve("who won the 1994 world cup", top_k=3)
    print(f"BM25 test query success: top doc_id={test_hits[0]['doc_id']}, score={test_hits[0]['score']:.4f}")

    # 2. Dense Verification
    emb_path = cfg.corpus.embeddings_path
    if not os.path.exists(emb_path) and os.path.exists("data/dense_embeddings_45k.npy"):
        emb_path = "data/dense_embeddings_45k.npy"

    print(f"Checking dense embeddings at: {emb_path}")
    dense = DenseRetriever(corpus=corpus, model_name=cfg.retrieval.dense_model_name, device=cfg.retrieval.device)

    if os.path.exists(emb_path) and not args.force_reencode:
        print("Loading cached dense embeddings...")
        dense.load_embeddings(emb_path)
        print(f"Loaded embeddings shape: {dense.embeddings.shape}, dtype: {dense.embeddings.dtype}")
    else:
        print("Encoding corpus passages using sentence-transformers/all-MiniLM-L6-v2...")
        dense.index_corpus(batch_size=cfg.retrieval.batch_size)
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        dense.save_embeddings(emb_path)
        print(f"Saved dense embeddings to: {emb_path}")

    dense_hits = dense.retrieve("who won the 1994 world cup", top_k=3)
    print(f"Dense test query success: top doc_id={dense_hits[0]['doc_id']}, cosine={dense_hits[0]['score']:.4f}")
    print("All indexes built and verified successfully.")


if __name__ == "__main__":
    main()
