"""Tokenization shared by FTS pre-segmentation, lexical rerank, and overlap checks.

Chinese is segmented with jieba. Callers that index FTS must store these tokens
(or an FTS5 ``simple`` tokenizer). Do not index raw CJK with unicode61 alone:
unicode61 treats an entire Han run as one token and Chinese search fails silently.
"""

from __future__ import annotations

import re

import jieba

# jieba prints a load message; keep it quiet for service logs.
jieba.setLogLevel(jieba.logging.INFO)

_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased jieba tokens with punctuation dropped."""
    if not text:
        return []
    out: list[str] = []
    for raw in jieba.cut(text, cut_all=False):
        tok = raw.strip().lower()
        if not tok or tok.isspace():
            continue
        if _PUNCT.fullmatch(tok):
            continue
        out.append(tok)
    return out


def join_tokens(text: str) -> str:
    return " ".join(tokenize(text))
