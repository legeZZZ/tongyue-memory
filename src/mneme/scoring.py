"""Fusion and final ranking. Formulas are fixed so eval behavior is inspectable.

RRF (Cormack, SIGIR 2009), k = 60:

    rrf(d) = sum_lists  1 / (k + rank_list(d))     # ranks start at 1

Final score, after the reranker has already returned σ(logit) in ``[0, 1]``:

    final(d) = rerank(d) + β · time_fit(d) + γ · src_bonus(d)

Defaults: β = 0.5, γ = 0.2, TAU = 0.25 (auditor override of the 0.35 blueprint
value). Abstain when the best pure rerank relevance is below TAU. ``final``
orders the survivors; time fit and source bonus must not revive a weak hit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone


def rrf_fuse(rank_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict[str, float]:
    """Fuse ranked id lists. Empty lists are ignored. Rank 1 is the head of each list."""
    scores: dict[str, float] = defaultdict(float)
    active = [(lst, (weights[i] if weights and i < len(weights) else 1.0)) for i, lst in enumerate(rank_lists) if lst]
    for lst, weight in active:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] += weight / (k + rank)
    return dict(scores)


def ranked_ids(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def time_fit(
    *,
    tense: str,
    kind: str,
    status: str | None,
    valid_from: str | None,
    valid_to: str | None,
    ts_event: str,
    as_of: str | None,
    in_update_pair: bool,
    order_position: float,
) -> float:
    """Time agreement in roughly [-1, 1]. Non-temporal queries contribute 0."""
    if tense in {"", "none"}:
        return 0.0
    if tense == "current":
        if kind == "fact":
            return 1.0 if status == "active" and not valid_to else -1.0
        return 0.0
    if tense == "as-of":
        if not as_of:
            return 0.0
        if kind == "fact":
            start = valid_from or ts_event
            if start <= as_of and (valid_to is None or as_of < valid_to):
                return 1.0
            return 0.0
        return 1.0 if ts_event <= as_of else 0.0
    if tense == "delta":
        return 1.0 if in_update_pair else 0.0
    if tense == "order":
        if kind == "fact":
            return order_position
        return 0.0
    return 0.0


def src_bonus(confidence: float, provenance: str, ts_event: str, now: datetime | None = None) -> float:
    """Stated facts outweigh inferences. Recency is a small nudge, not a ranker."""
    mult = 1.0 if provenance == "stated" else 0.6
    base = max(0.0, min(1.0, confidence)) * mult
    moment = now or datetime.now(timezone.utc)
    recency = 0.15
    try:
        if ts_event.endswith("Z"):
            event = datetime.fromisoformat(ts_event[:-1] + "+00:00")
        else:
            event = datetime.fromisoformat(ts_event)
            if event.tzinfo is None:
                event = event.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (moment - event.astimezone(timezone.utc)).total_seconds() / 86400.0)
        recency = 0.15 * math.exp(-age_days / 180.0)
    except ValueError:
        recency = 0.15
    return base + recency


def combine(rerank: float, fit: float, source: float, beta: float, gamma: float) -> float:
    return rerank + beta * fit + gamma * source
