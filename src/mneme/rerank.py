"""Pluggable rerankers.

The default is a lexical overlap scorer so CI and a CPU box can boot without
the 0.6B weights. Set ``MNEME_RERANKER=qwen3`` and install ``mneme[qwen]`` to
score with Qwen/Qwen3-Reranker-0.6B (Apache-2.0). Both return relevance in
``[0, 1]``, which is the σ(rerank logit) term in the final score.
"""

from __future__ import annotations

import logging
from typing import Protocol

from mneme.tokenize import tokenize

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance in [0, 1] per text, higher is more relevant."""


class LexicalReranker:
    """Query-token recall against each text. CJK uses the same jieba tokens as FTS."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0 for _ in texts]
        out: list[float] = []
        for text in texts:
            t = set(tokenize(text))
            if not t:
                out.append(0.0)
                continue
            out.append(len(q & t) / len(q))
        return out


class Qwen3Reranker:
    """Cross-encoder scores from Qwen3-Reranker (yes/no generative head).

    The prompt format follows the Qwen3-Embedding model card
    (https://github.com/QwenLM/Qwen3-Embedding). Weights download on first use.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 reranker is not installed. Run pip install 'mneme[qwen]' "
                "(torch and transformers) and set MNEME_RERANKER=qwen3. "
                "The default MNEME_RERANKER=lexical needs neither."
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and the Instruct provided. Note that the answer can "
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.instruction = "Given a memory search query, retrieve memories that are relevant evidence."
        self.max_length = 8192

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        torch = self._torch
        pairs = [self._format(query, text) for text in texts]
        prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        scores: list[float] = []
        for pair in pairs:
            body_ids = self.tokenizer.encode(pair, add_special_tokens=False)
            keep = self.max_length - len(prefix_ids) - len(suffix_ids)
            ids = prefix_ids + body_ids[:keep] + suffix_ids
            input_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0, -1]
            pair_logits = torch.stack([logits[self.token_false_id], logits[self.token_true_id]], dim=0)
            probs = torch.nn.functional.softmax(pair_logits, dim=0)
            scores.append(float(probs[1].item()))
        return scores

    def _format(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"


def build_reranker(name: str, qwen_model: str) -> Reranker:
    if name in {"", "lexical", "default"}:
        return LexicalReranker()
    if name in {"qwen3", "qwen", "qwen3-reranker"}:
        log.info("loading Qwen3 reranker %s", qwen_model)
        return Qwen3Reranker(qwen_model)
    raise RuntimeError(
        f"unknown MNEME_RERANKER={name!r}; use 'lexical' (default) or 'qwen3'"
    )
