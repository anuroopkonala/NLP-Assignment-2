.PHONY: help install setup eval ablation task1 task2 task3 evaluate clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install    - Install package dependencies"
	@echo "  make setup      - Verify corpus and prepare validation subsets"
	@echo "  make eval       - Quick verification of all metrics (--eval_only)"
	@echo "  make ablation   - Display multi-approach ablation study table"
	@echo "  make task1      - Run Task 1 BM25 + Extractive Baseline"
	@echo "  make task2      - Run Task 2 Dense RAG Reproduction"
	@echo "  make task3      - Run Task 3 Hybrid Fusion + Groundedness"
	@echo "  make evaluate   - Run consolidated evaluation script and regenerate plots"

install:
	$(PYTHON) -m pip install -r requirements.txt

setup:
	$(PYTHON) scripts/build_corpus.py
	$(PYTHON) scripts/prepare_dataset.py

eval:
	$(PYTHON) run_all.py --eval_only

ablation:
	$(PYTHON) run_all.py --ablation

task1:
	$(PYTHON) scripts/run_task1.py

task2:
	$(PYTHON) scripts/run_task2.py

task3:
	$(PYTHON) scripts/run_task3.py

evaluate:
	$(PYTHON) scripts/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
