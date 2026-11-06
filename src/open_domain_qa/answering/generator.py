"""
Generative reader module for RAG pipelines.
Uses an open instruction-tuned LLM (Qwen2.5-1.5B-Instruct) with greedy decoding
to generate direct short answers based strictly on retrieved context.
"""

from typing import Any, Dict, List, Optional
import torch

from open_domain_qa.answering.prompting import format_rag_prompt


class RAGGenerator:
    """Wrapper around HuggingFace causal LM for deterministic RAG answer generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self) -> None:
        """Loads model weights and tokenizer lazily when first needed."""
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        self.model.eval()

    def generate_answer(
        self,
        question: str,
        context_passages: List[str],
        max_tokens_per_passage: int = 200,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generates a short, factual answer given question and retrieved passages."""
        self._ensure_loaded()

        prompt = format_rag_prompt(
            question=question,
            context_passages=context_passages,
            max_tokens_per_passage=max_tokens_per_passage,
            few_shot_examples=few_shot_examples,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.temperature <= 0.0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        generated_tokens = outputs[0][prompt_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Isolate the first non-empty line as concise short answer
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()

        return answer


class MockRAGGenerator:
    """Mock generator for rapid testing or environments without GPU."""

    def __init__(self, return_text: str = "mock answer") -> None:
        self.return_text = return_text

    def generate_answer(
        self,
        question: str,
        context_passages: List[str],
        max_tokens_per_passage: int = 200,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return self.return_text
