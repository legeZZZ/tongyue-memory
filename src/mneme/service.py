"""Add and Search pipelines.

Add commits verbatim turns, document embeddings, and FTS before it returns,
then waits up to ``fact_timeout_s`` for atomic-fact extraction and bitemporal
reconcile. On timeout the chunk stays in the offline queue and Add still
returns 200: the raw text is already searchable.

Search returns evidence only. It does not answer the question and it does not
read ``options`` (choice text is not memory and is not a gold label).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from mneme.attributes import same_mutable_attribute, text_referent
from mneme.config import Settings
from mneme.embeddings import Embedder, UpstreamError, build_embedder
from mneme.llm import (
    Analysis,
    FactDraft,
    HeuristicLLM,
    LLM,
    TurnView,
    Verdict,
    build_llm,
    parse_spo,
    sensitivity_of,
)
from mneme.rerank import LexicalReranker, Reranker, build_reranker
from mneme.schemas import AddRequest, AddResponse, SearchHit, SearchRequest, SearchResponse
from mneme.scoring import combine, ranked_ids, rrf_fuse, src_bonus, time_fit
from mneme.store import Conflict, NewTurn, Store, StoredFact, StoredTurn
from mneme.timeparse import from_unix_ms, parse_event_time, to_iso
from mneme.tokenize import tokenize

log = logging.getLogger(__name__)

_STOP = {
    "的", "了", "是", "在", "我", "你", "吗", "什么", "怎么", "哪", "谁", "和", "与", "也",
    "就", "都", "还", "把", "被", "一个", "我们", "a", "an", "the", "is", "are", "was",
    "were", "my", "your", "what", "who", "how", "did", "does", "do", "i", "me", "to", "of",
}


@dataclass
class Hit:
    key: str
    kind: str
    text: str
    content_raw: str
    created_at: str
    ts_event: str
    user_id: str
    status: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 1.0
    provenance: str = "stated"
    sensitivity: str = "normal"
    chunk_id: str | None = None
    ordinal: int | None = None
    src_turn_id: int | None = None
    record_id: int | None = None
    final: float = 0.0
    relevance: float = 0.0
    in_update_pair: bool = False
    order_position: float = 0.0


class MemoryService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        store: Store | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or Store(settings.db_path, settings.fts_simple_path)
        self.embedder = embedder if embedder is not None else build_embedder(settings)
        self.llm = llm if llm is not None else build_llm(settings)
        self.reranker = reranker if reranker is not None else build_reranker(settings.reranker, settings.qwen_model)
        self._lexical = LexicalReranker()
        self._cache: dict[str, Analysis] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.settings.disable_worker or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="mneme-enrich", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.store.close()
        for obj in (self.embedder, self.llm):
            closer = getattr(obj, "close", None)
            if closer:
                closer()

    def add(self, req: AddRequest) -> AddResponse:
        digest = body_hash(req)
        existing = self.store.get_request(req.request_id)
        if existing is not None:
            if existing["body_hash"] != digest:
                raise Conflict(req.request_id)
            self.enrich_bounded(req.request_id, req.user_id)
            return _ok(req)

        anchor = _chunk_anchor(req)
        texts = [msg.content for msg in req.messages]
        vectors = self.embedder.embed_texts(texts, text_type="document")
        ingest = to_iso(datetime.now(timezone.utc))
        prepared: list[NewTurn] = []
        for ordinal, (msg, vec) in enumerate(zip(req.messages, vectors, strict=True)):
            msg_anchor = from_unix_ms(msg.timestamp) if msg.timestamp is not None else anchor
            event = parse_event_time(msg.content, msg_anchor)
            prepared.append(
                NewTurn(
                    ordinal=ordinal,
                    role=msg.role,
                    content=msg.content,
                    ts_event=to_iso(event),
                    emb=_blob(vec),
                )
            )
        self.store.persist_turns(
            request_id=req.request_id,
            user_id=req.user_id,
            session_id=req.session_id,
            body_hash=digest,
            turns=prepared,
            ts_ingest=ingest,
        )
        # Raw rows are committed. Fact extraction is best-effort inside the budget.
        self.enrich_bounded(req.request_id, req.user_id)
        return _ok(req)

    def search(self, req: SearchRequest) -> SearchResponse:
        # ``options`` is accepted by the contract and deliberately unused.
        del req.options
        if not self.store.has_user(req.user_id):
            return SearchResponse(data=[])

        analysis = self._analyze(req.query)
        q_vec = self.embedder.embed_texts([req.query], text_type="query")[0]
        lists: list[list[str]] = [self.store.vector_topk(req.user_id, q_vec, self.settings.recall_limit)]
        recall = (analysis.recall_q or "").strip()
        if recall and recall != req.query.strip():
            rq = self.embedder.embed_texts([recall], text_type="query")[0]
            lists.append(self.store.vector_topk(req.user_id, rq, self.settings.recall_limit))
        lists.append(self.store.fts_topk(req.user_id, req.query, self.settings.recall_limit))
        fused = rrf_fuse(lists, k=self.settings.rrf_k)
        keys = ranked_ids(fused)[: self.settings.recall_limit]
        if analysis.hops > 1:
            keys = self._bridge(req.user_id, keys)
        hits = self._hits_for(req.user_id, keys)
        # ``hidden`` is local to this search. Do not stash it on the service:
        # concurrent searches would otherwise revive each other's stale turns.
        hits, hidden = self._temporal_filter(hits, analysis, req.user_id)
        if not hits:
            return SearchResponse(data=[])
        hits = hits[: self.settings.rerank_max]
        scored = self._score(req.query, hits, analysis)
        # Abstain on pure rerank. time_fit and src_bonus stay in ``final`` for rank only.
        scored = [hit for hit in scored if hit.relevance >= self.settings.tau]
        if not scored:
            return SearchResponse(data=[])
        scored.sort(key=lambda hit: hit.final, reverse=True)
        picked = self._expand(scored, req.top_k, analysis, hidden)
        picked = _disclose(picked, req.query)
        picked = [hit for hit in picked if hit.relevance >= self.settings.tau and hit.text.strip()]
        if not picked or max(hit.relevance for hit in picked) < self.settings.tau:
            return SearchResponse(data=[])
        picked = _raw_first(picked, analysis.tense)[: req.top_k]
        return SearchResponse(
            data=[
                SearchHit(
                    id=hit.key,
                    content=hit.text,
                    score=float(round(hit.final, 6)),
                    created_at=hit.created_at,
                )
                for hit in picked
            ]
        )

    def enrich_bounded(self, chunk_id: str, user_id: str) -> bool:
        """Run fact extraction until the budget. Timeout leaves the job queued."""
        if self.store.enrich_status(chunk_id, user_id) == "done":
            return True
        done = threading.Event()
        box: dict[str, object] = {}

        def target() -> None:
            try:
                box["ok"] = self.enrich_once(chunk_id, user_id)
            except Exception as exc:  # noqa: BLE001 — raw write already durable
                box["err"] = exc
                log.warning("enrich failed for chunk %s: %s", chunk_id, exc)
            finally:
                done.set()

        threading.Thread(target=target, name=f"enrich-{chunk_id[:24]}", daemon=True).start()
        if not done.wait(self.settings.fact_timeout_s):
            log.info("enrich budget exceeded for chunk %s; raw memory stays searchable", chunk_id)
            return False
        return bool(box.get("ok"))

    def enrich_once(self, chunk_id: str, user_id: str) -> bool:
        turns = self.store.claim_enrich(chunk_id, user_id)
        if turns is None:
            return self.store.enrich_status(chunk_id, user_id) == "done"
        try:
            self._write_facts(user_id, turns)
        except Exception as exc:
            self.store.mark_enrich_pending(chunk_id, user_id, str(exc))
            raise
        self.store.mark_enrich_done(chunk_id, user_id)
        return True

    def drain_once(self) -> bool:
        job = self.store.claim_next_pending()
        if job is None:
            return False
        chunk_id, user_id = job
        try:
            self.enrich_once(chunk_id, user_id)
        except Exception:
            log.warning("queued enrich failed for %s", chunk_id, exc_info=True)
        return True

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self.drain_once()
            except Exception:
                log.warning("enrich loop error", exc_info=True)

    def _write_facts(self, user_id: str, turns: list[StoredTurn]) -> None:
        views = [TurnView(id=t.id, role=t.role, content=t.content, ts_event=t.ts_event) for t in turns]
        drafts = [_bind_source(draft, turns) for draft in self.llm.extract(views)]
        drafts = [draft for draft in drafts if draft.statement.strip()]
        if not drafts:
            return
        vectors = self.embedder.embed_texts([draft.statement for draft in drafts], text_type="document")
        ingest = to_iso(datetime.now(timezone.utc))
        inserted: list[tuple[int, str]] = []
        for draft, vec in zip(drafts, vectors, strict=True):
            ts = draft.ts_event or turns[0].ts_event
            candidates = self._reconcile_candidates(user_id, draft, vec)
            verdict = self._judge(draft, candidates)
            confidence = draft.confidence
            valid_to: str | None = None
            status = "active"
            superseded_by: int | None = None
            close_id: int | None = None
            if verdict.kind == "contradicts":
                confidence *= 0.5
            if verdict.kind == "updates" and verdict.target_id is not None:
                old = next((cand for cand in candidates if cand["id"] == verdict.target_id), None)
                if old is not None and ts >= str(old["ts_event"]):
                    close_id = int(old["id"])
                elif old is not None:
                    valid_to = str(old["valid_from"])
                    status = "superseded"
                    superseded_by = int(old["id"])
            fact_id = self.store.insert_fact(
                user_id=user_id,
                src_turn_id=draft.src_turn_id,
                subject=draft.subject,
                predicate=draft.predicate,
                object_=draft.object,
                statement=draft.statement,
                ts_event=ts,
                ts_ingest=ingest,
                valid_from=ts,
                valid_to=valid_to,
                status=status,
                superseded_by=superseded_by,
                confidence=confidence,
                provenance=draft.provenance if draft.provenance in {"stated", "inferred"} else "stated",
                sensitivity=sensitivity_of(draft.statement, draft.sensitivity),
                emb=_blob(vec),
            )
            if close_id is not None:
                self.store.supersede_fact(user_id, close_id, ts, fact_id)
                self.store.add_edge(user_id, close_id, fact_id, "updates")
            elif superseded_by is not None:
                self.store.add_edge(user_id, fact_id, superseded_by, "updates")
            inserted.append((fact_id, ts))
        ordered = sorted(inserted, key=lambda item: item[1])
        for (left, left_ts), (right, right_ts) in zip(ordered, ordered[1:], strict=False):
            if left_ts < right_ts:
                self.store.add_edge(user_id, left, right, "before")
                self.store.add_edge(user_id, right, left, "after")
            elif left_ts == right_ts:
                self.store.add_edge(user_id, left, right, "same_as")

    def _reconcile_candidates(self, user_id: str, draft: FactDraft, vec: np.ndarray) -> list[dict]:
        facts = self.store.fact_rows_for_reconcile(user_id)
        vectors = {fact_id: emb for fact_id, emb in self.store.fact_embeddings(user_id)}
        ranked: list[tuple[int, float, StoredFact, bool, bool]] = []
        for fact in facts:
            exact = bool(
                draft.subject
                and fact.subject.casefold() == draft.subject.casefold()
                and fact.predicate.casefold() == draft.predicate.casefold()
            )
            attribute = same_mutable_attribute(
                draft.subject,
                draft.predicate,
                draft.object,
                draft.statement,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.statement,
            )
            other = vectors.get(fact.id)
            sim = float(np.dot(vec, other)) if other is not None and other.shape[0] == vec.shape[0] else 0.0
            ranked.append((int(exact or attribute), sim, fact, exact, attribute))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            _candidate(fact, sim=sim, same_spo=exact, same_attribute=attribute)
            for _rank, sim, fact, exact, attribute in ranked[:10]
        ]

    def _judge(self, draft: FactDraft, candidates: list[dict]) -> Verdict:
        """Ask the configured judge, then apply attribute, sim, or SPO if it abstains.

        gpt-4o-mini sometimes returns unrelated when the predicate wording
        drifted. The heuristic treats a known mutable attribute (residence, job,
        cat count, favorite director) as the same referent even when cosine is
        under 0.82, and still supersedes on ``sim`` / ``same_spo``. Fake models
        already are that heuristic; calling it twice would not change the verdict.
        """
        verdict = self.llm.judge(draft, candidates)
        if isinstance(self.llm, HeuristicLLM):
            return verdict
        known: set[int] = set()
        for cand in candidates:
            try:
                known.add(int(cand["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        if verdict.kind == "updates" and verdict.target_id in known:
            return verdict
        backup = HeuristicLLM().judge(draft, candidates)
        if backup.kind == "updates" and backup.target_id is not None:
            return backup
        return verdict

    def _analyze(self, query: str) -> Analysis:
        cached = self._cache.get(query)
        if cached is not None:
            return cached
        analysis = _coerce_tense(query, self.llm.analyze(query))
        if len(self._cache) > 1024:
            self._cache.clear()
        self._cache[query] = analysis
        return analysis

    def _bridge(self, user_id: str, keys: list[str]) -> list[str]:
        seeds = [int(key[2:]) for key in keys if key.startswith("f:")]
        extra = self.store.bridge_fact_ids(user_id, seeds, limit=30)
        seen = set(keys)
        for fact_id in extra:
            key = f"f:{fact_id}"
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    def _hits_for(self, user_id: str, keys: list[str]) -> list[Hit]:
        hits: list[Hit] = []
        for key, record in self.store.load_records(user_id, keys):
            if isinstance(record, StoredTurn):
                hits.append(_hit_turn(record))
            else:
                hits.append(_hit_fact(record))
        return hits

    def _temporal_filter(self, hits: list[Hit], analysis: Analysis, user_id: str) -> tuple[list[Hit], set[int]]:
        """Filter hits and return turn ids current-tense expansion must not revive."""
        tense = analysis.tense
        if tense == "current":
            hidden = self._current_hidden(hits, user_id)
            kept: list[Hit] = []
            for hit in hits:
                if hit.kind == "fact" and hit.status == "active" and not hit.valid_to:
                    kept.append(hit)
                elif hit.kind == "turn" and hit.record_id not in hidden:
                    kept.append(hit)
            drop = _older_attribute_fact_keys(kept)
            for hit in kept:
                if hit.key in drop and hit.src_turn_id is not None:
                    hidden.add(hit.src_turn_id)
            if drop or hidden:
                kept = [
                    hit
                    for hit in kept
                    if hit.key not in drop and not (hit.kind == "turn" and hit.record_id in hidden)
                ]
            return kept, hidden
        if tense == "as-of" and analysis.as_of:
            moment = analysis.as_of
            stale_turns = self.store.turn_ids_invalid_at(user_id, moment)
            kept = []
            for hit in hits:
                if hit.kind == "turn":
                    if hit.ts_event <= moment and hit.record_id not in stale_turns:
                        kept.append(hit)
                else:
                    start = hit.valid_from or hit.ts_event
                    if start <= moment and (hit.valid_to is None or moment < hit.valid_to):
                        kept.append(hit)
            return kept, stale_turns
        if tense == "delta":
            pair = self.store.update_pair_ids(user_id)
            if not pair:
                return hits, set()
            kept = []
            for hit in hits:
                if hit.kind == "fact" and hit.record_id in pair:
                    hit.in_update_pair = True
                    kept.append(hit)
                elif hit.kind == "turn":
                    kept.append(hit)
            return kept, set()
        if tense == "order":
            facts = sorted((hit for hit in hits if hit.kind == "fact"), key=lambda hit: hit.ts_event)
            span = max(len(facts) - 1, 1)
            for index, hit in enumerate(facts):
                hit.order_position = index / span
            return hits, set()
        return hits, set()

    def _current_hidden(self, hits: list[Hit], user_id: str) -> set[int]:
        """Superseded sources, plus older same-referent turns enrich never linked.

        Same mutable attribute, or the same ``parse_spo`` subject and predicate,
        keeps only the latest ``ts_event``. Losers are returned to the caller so
        ``_expand`` cannot pull them back in as neighbors. Computed per search,
        not stored.
        """
        hidden = set(self.store.suppressed_turn_ids(user_id))
        hidden |= _spo_loser_ids(hits)
        hidden |= self._older_same_spo_neighbor_ids(hits, hidden)
        return hidden

    def _older_same_spo_neighbor_ids(self, hits: list[Hit], already: set[int]) -> set[int]:
        """Older same-referent turns just outside the recalled set.

        Radius matches neighbor expansion. A current hit can sit next to a
        stale line that recall dropped; expansion would otherwise append it.
        """
        radius = self.settings.neighbor_radius
        if radius <= 0:
            return set()
        found: set[int] = set()
        for hit in hits:
            if hit.kind == "turn":
                if hit.record_id is None or hit.record_id in already:
                    continue
                group = _turn_group_key(hit.content_raw)
                if group is None or hit.chunk_id is None or hit.ordinal is None:
                    continue
                neighbor_ids = self.store.neighbor_turn_ids(
                    hit.user_id, hit.chunk_id, hit.ordinal, radius
                )
                self._mark_older_turns(
                    hit.user_id, neighbor_ids, group, hit.ts_event, None, found
                )
            elif (
                hit.kind == "fact"
                and hit.status == "active"
                and not hit.valid_to
                and hit.src_turn_id is not None
            ):
                group = _turn_group_key(hit.content_raw)
                if group is None:
                    continue
                loaded = self.store.load_records(hit.user_id, [f"t:{hit.src_turn_id}"])
                for _key, record in loaded:
                    if not isinstance(record, StoredTurn):
                        continue
                    around = self.store.neighbor_turn_ids(hit.user_id, record.chunk_id, record.ordinal, radius)
                    self._mark_older_turns(
                        hit.user_id, around, group, hit.ts_event, record.id, found
                    )
        return found

    def _mark_older_turns(
        self,
        user_id: str,
        turn_ids: list[int],
        group: tuple[str, str],
        ts_event: str,
        keep_id: int | None,
        found: set[int],
    ) -> None:
        if not turn_ids:
            return
        keys = [f"t:{turn_id}" for turn_id in turn_ids]
        for _key, record in self.store.load_records(user_id, keys):
            if not isinstance(record, StoredTurn):
                continue
            if keep_id is not None and record.id == keep_id:
                continue
            if record.ts_event >= ts_event:
                continue
            if _turn_group_key(record.content) == group:
                found.add(record.id)

    def _score(self, query: str, hits: list[Hit], analysis: Analysis) -> list[Hit]:
        try:
            reranks = self.reranker.score(query, [hit.text for hit in hits])
        except Exception:
            log.warning("reranker failed; using lexical scores", exc_info=True)
            reranks = self._lexical.score(query, [hit.text for hit in hits])
        if len(reranks) != len(hits):
            reranks = self._lexical.score(query, [hit.text for hit in hits])
        now = datetime.now(timezone.utc)
        kept: list[Hit] = []
        for hit, rerank in zip(hits, reranks, strict=True):
            relevance = max(0.0, min(1.0, float(rerank)))
            if relevance < self.settings.rerank_floor:
                continue
            fit = time_fit(
                tense=analysis.tense,
                kind=hit.kind,
                status=hit.status,
                valid_from=hit.valid_from,
                valid_to=hit.valid_to,
                ts_event=hit.ts_event,
                as_of=analysis.as_of,
                in_update_pair=hit.in_update_pair,
                order_position=hit.order_position,
            )
            source = src_bonus(hit.confidence, hit.provenance, hit.ts_event, now)
            hit.relevance = relevance
            hit.final = combine(relevance, fit, source, self.settings.beta, self.settings.gamma)
            kept.append(hit)
        return kept

    def _expand(self, ranked: list[Hit], top_k: int, analysis: Analysis, hidden: set[int]) -> list[Hit]:
        selected: list[Hit] = []
        seen: set[str] = set()
        for anchor in ranked:
            block = [anchor, *self._neighbors(anchor)]
            for item in block:
                if item.key in seen:
                    continue
                if not _neighbor_ok(item, analysis, hidden):
                    continue
                if item is not anchor:
                    if item.final < anchor.final:
                        item.final = anchor.final * 0.99
                    # Expansion does not rerank the neighbor. Inherit the anchor
                    # so abstention, which reads relevance, keeps the context turn.
                    item.relevance = anchor.relevance
                seen.add(item.key)
                selected.append(item)
                if len(selected) >= top_k:
                    return selected
        return selected

    def _neighbors(self, hit: Hit) -> list[Hit]:
        radius = self.settings.neighbor_radius
        if radius <= 0:
            return []
        turn_ids: list[int] = []
        if hit.kind == "turn" and hit.chunk_id is not None and hit.ordinal is not None:
            turn_ids = self.store.neighbor_turn_ids(hit.user_id, hit.chunk_id, hit.ordinal, radius)
        elif hit.kind == "fact" and hit.src_turn_id is not None:
            records = self.store.load_records(hit.user_id, [f"t:{hit.src_turn_id}"])
            for _key, record in records:
                if isinstance(record, StoredTurn):
                    turn_ids.append(record.id)
                    turn_ids.extend(
                        self.store.neighbor_turn_ids(hit.user_id, record.chunk_id, record.ordinal, radius)
                    )
        hits: list[Hit] = []
        if not turn_ids:
            return hits
        keys = [f"t:{turn_id}" for turn_id in turn_ids]
        for _key, record in self.store.load_records(hit.user_id, keys):
            if isinstance(record, StoredTurn):
                hits.append(_hit_turn(record))
        return hits


def body_hash(req: AddRequest) -> str:
    payload = {
        "messages": [
            {"content": msg.content, "role": msg.role, "timestamp": msg.timestamp} for msg in req.messages
        ],
        "request_id": req.request_id,
        "session_id": req.session_id,
        "user_id": req.user_id,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _ok(req: AddRequest) -> AddResponse:
    return AddResponse(
        success=True,
        request_id=req.request_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )


def _chunk_anchor(req: AddRequest) -> datetime:
    for msg in req.messages:
        if msg.timestamp is not None:
            return from_unix_ms(msg.timestamp)
    return datetime.now(timezone.utc)


def _blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _bind_source(draft: FactDraft, turns: list[StoredTurn]) -> FactDraft:
    known = {turn.id for turn in turns}
    if draft.src_turn_id in known:
        if not draft.ts_event:
            match = next(turn for turn in turns if turn.id == draft.src_turn_id)
            draft.ts_event = match.ts_event
        return draft
    for turn in turns:
        if draft.statement and (draft.statement in turn.content or turn.content in draft.statement):
            draft.src_turn_id = turn.id
            draft.ts_event = draft.ts_event or turn.ts_event
            return draft
    if turns:
        draft.src_turn_id = turns[-1].id
        draft.ts_event = draft.ts_event or turns[-1].ts_event
    return draft


def _spo_loser_ids(hits: list[Hit]) -> set[int]:
    """Turn ids that lose a same-referent current-tense comparison inside ``hits``."""
    groups: dict[tuple[str, str], list[Hit]] = {}
    for hit in hits:
        if hit.kind != "turn" or hit.record_id is None:
            continue
        key = _turn_group_key(hit.content_raw)
        if key is None:
            continue
        groups.setdefault(key, []).append(hit)
    losers: set[int] = set()
    for group in groups.values():
        winner = max(group, key=lambda hit: (hit.ts_event, hit.record_id or 0))
        for hit in group:
            if hit.record_id != winner.record_id:
                losers.add(int(hit.record_id))
    return losers


def _older_attribute_fact_keys(hits: list[Hit]) -> set[str]:
    """Active facts that lose a same-attribute comparison to a newer fact in ``hits``."""
    groups: dict[tuple[str, str], list[Hit]] = {}
    for hit in hits:
        if hit.kind != "fact" or hit.status != "active" or hit.valid_to:
            continue
        key = text_referent(hit.content_raw)
        if key is None:
            continue
        groups.setdefault(key, []).append(hit)
    drop: set[str] = set()
    for group in groups.values():
        winner = max(group, key=lambda hit: (hit.ts_event, hit.record_id or 0))
        for hit in group:
            if hit.key != winner.key:
                drop.add(hit.key)
    return drop


def _turn_group_key(text: str) -> tuple[str, str] | None:
    ref = text_referent(text)
    if ref is not None:
        return ref
    subject, predicate, _obj = parse_spo(text)
    if subject and predicate:
        return ("spo", subject.casefold() + "\0" + predicate.casefold())
    return None


def _coerce_tense(query: str, analysis: Analysis) -> Analysis:
    """Fill a missing or non-current label when the query is clearly present-tense.

    gpt-4o-mini sometimes returns ``none`` or ``delta`` for 「现在」 / "right now".
    Those labels skip the filter that hides superseded values. As-of questions
    carry a date, so the heuristic does not call them current.
    """
    hinted = HeuristicLLM().analyze(query)
    if hinted.tense == "current" and analysis.tense != "current":
        return Analysis(
            tense="current",
            as_of=None,
            hops=analysis.hops,
            recall_q=analysis.recall_q or hinted.recall_q,
        )
    if analysis.tense == "none" and hinted.tense != "none":
        return Analysis(
            tense=hinted.tense,
            as_of=hinted.as_of,
            hops=analysis.hops,
            recall_q=analysis.recall_q or hinted.recall_q,
        )
    return analysis


def _candidate(fact: StoredFact, *, sim: float, same_spo: bool, same_attribute: bool = False) -> dict:
    clipped = max(0.0, min(1.0, float(sim)))
    return {
        "id": fact.id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "statement": fact.statement,
        "ts_event": fact.ts_event,
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "status": fact.status,
        "sim": round(clipped, 4),
        "same_spo": bool(same_spo),
        "same_attribute": bool(same_attribute),
    }


def _hit_turn(turn: StoredTurn) -> Hit:
    return Hit(
        key=f"t:{turn.id}",
        kind="turn",
        text=f"[{turn.ts_event}] {turn.role}: {turn.content}",
        content_raw=turn.content,
        created_at=turn.ts_event,
        ts_event=turn.ts_event,
        user_id=turn.user_id,
        confidence=1.0,
        provenance="stated",
        sensitivity=sensitivity_of(turn.content),
        chunk_id=turn.chunk_id,
        ordinal=turn.ordinal,
        record_id=turn.id,
    )


def _hit_fact(fact: StoredFact) -> Hit:
    return Hit(
        key=f"f:{fact.id}",
        kind="fact",
        text=f"[{fact.ts_event}] {fact.statement}",
        content_raw=fact.statement,
        created_at=fact.ts_event,
        ts_event=fact.ts_event,
        user_id=fact.user_id,
        status=fact.status,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        confidence=fact.confidence,
        provenance=fact.provenance,
        sensitivity=fact.sensitivity,
        src_turn_id=fact.src_turn_id,
        record_id=fact.id,
    )


def _disclose(hits: list[Hit], query: str) -> list[Hit]:
    query_tokens = {tok for tok in tokenize(query) if tok not in _STOP}
    kept: list[Hit] = []
    for hit in hits:
        if hit.sensitivity == "sensitive":
            text_tokens = {tok for tok in tokenize(hit.content_raw) if tok not in _STOP}
            if not (query_tokens & text_tokens):
                continue
        kept.append(hit)
    return kept


def _raw_first(hits: list[Hit], tense: str) -> list[Hit]:
    raw = [hit for hit in hits if hit.kind == "turn"]
    facts = [hit for hit in hits if hit.kind == "fact"]
    if tense == "order":
        raw.sort(key=lambda hit: hit.ts_event)
        facts.sort(key=lambda hit: hit.ts_event)
    return raw + facts


def _neighbor_ok(hit: Hit, analysis: Analysis, hidden: set[int]) -> bool:
    if hit.kind == "turn" and hit.record_id in hidden and analysis.tense in {"current", "as-of"}:
        return False
    if analysis.tense == "as-of" and analysis.as_of and hit.ts_event > analysis.as_of:
        return False
    return True
