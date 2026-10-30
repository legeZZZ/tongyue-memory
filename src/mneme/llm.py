"""gpt-4o-mini calls used only on the memory path.

The model extracts facts, classifies question tense, rewrites a recall probe,
and judges whether a new fact updates an old one. It never answers the Search
question and never sees benchmark gold.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from mneme.attributes import same_mutable_attribute
from mneme.embeddings import UpstreamError
from mneme.timeparse import parse_iso, to_iso

log = logging.getLogger(__name__)

EXTRACT_SYSTEM = (
    "You extract atomic self-contained memory facts from dialogue turns. "
    "Return a JSON object with key \"facts\", a list of objects with keys "
    "statement, subject, predicate, object, ts_event (ISO-8601 or null), "
    "provenance (stated or inferred), confidence (0 to 1), sensitivity "
    "(normal or sensitive). Each statement must be supported by the turns. "
    "Do not invent events. Do not answer questions. Preserve names, numbers, "
    "and dates verbatim. Prefer stated over inferred."
)

ANALYZE_SYSTEM = (
    "You classify a retrieval question. Do not answer it. "
    "Return JSON with keys tense (current, as-of, delta, order, or none), "
    "as_of (ISO-8601 timestamp or null), hops (1 or 2), and recall_q. "
    "recall_q is a first-person memory probe that restates what to look up "
    "in the user's own voice. It must not contain an answer, a guess, or "
    "any text that was not asked."
)

# Cosine at or above this means the same referent even when predicate wording drifted.
SIM_SAME_REFERENT = 0.82

JUDGE_SYSTEM = (
    "You compare one new memory fact with candidate old facts. Do not answer "
    "an external question. Each candidate includes same_spo (true when the "
    "subject and predicate strings match), same_attribute (true when both facts "
    "are the same single-valued attribute for the same speaker, such as "
    "residence, job, cat count, or favorite director), and sim (cosine "
    "similarity of the two fact embeddings, from 0 to 1). Return JSON with keys "
    "kind (updates, contradicts, or unrelated) and target_id (the old fact id, "
    "or null). Treat the new fact as the same referent when same_spo is true, "
    "when same_attribute is true even if the predicate wording differs, or when "
    f"sim is at least {SIM_SAME_REFERENT:.2f} even if the predicate wording differs. "
    "Use updates when it is the same referent, the objects conflict, and the "
    "new fact is the later state. Use contradicts when both cannot be true but "
    "order is unclear. Otherwise unrelated."
)

_SENSITIVE = re.compile(
    r"密码|身份证|护照|诊断|病历|艾滋|怀孕|住址|password|passcode|ssn|diagnosis|medical record",
    re.IGNORECASE,
)
_SPO_CN = re.compile(r"^(.+?)的(.+?)是(.+)$")
_SPO_EN = re.compile(r"^my\s+(.+?)\s+is\s+(.+)$", re.IGNORECASE)


@dataclass
class FactDraft:
    statement: str
    subject: str
    predicate: str
    object: str
    ts_event: str | None
    provenance: str
    confidence: float
    sensitivity: str
    src_turn_id: int | None = None


@dataclass
class Analysis:
    tense: str
    as_of: str | None
    hops: int
    recall_q: str


@dataclass
class Verdict:
    kind: str
    target_id: int | None


@dataclass
class TurnView:
    id: int
    role: str
    content: str
    ts_event: str


class LLM(Protocol):
    def extract(self, turns: list[TurnView]) -> list[FactDraft]: ...

    def analyze(self, query: str) -> Analysis: ...

    def judge(self, new: FactDraft, candidates: list[dict]) -> Verdict: ...


def sensitivity_of(text: str, hinted: str | None = None) -> str:
    if hinted == "sensitive" or (text and _SENSITIVE.search(text)):
        return "sensitive"
    return "normal"


def parse_spo(text: str) -> tuple[str, str, str]:
    cleaned = text.strip().rstrip("。.!！?？")
    cn = _SPO_CN.match(cleaned)
    if cn:
        return cn.group(1).strip(), cn.group(2).strip(), cn.group(3).strip()
    en = _SPO_EN.match(cleaned)
    if en:
        return "user", en.group(1).strip(), en.group(2).strip()
    return "", "", ""


def _strip_fence(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    return body.strip()


def parse_json_object(text: str) -> dict:
    body = _strip_fence(text)
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("LLM JSON was not an object")
    return data


class HeuristicLLM:
    """Rule stand-in for gpt-4o-mini. Used when MNEME_FAKE_MODELS=1 and on LLM failure."""

    def extract(self, turns: list[TurnView]) -> list[FactDraft]:
        drafts: list[FactDraft] = []
        for turn in turns:
            subject, predicate, obj = parse_spo(turn.content)
            drafts.append(
                FactDraft(
                    statement=turn.content.strip(),
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    ts_event=turn.ts_event,
                    provenance="stated",
                    confidence=1.0,
                    sensitivity=sensitivity_of(turn.content),
                    src_turn_id=turn.id,
                )
            )
        return drafts

    def analyze(self, query: str) -> Analysis:
        as_of = _as_of_from_query(query)
        if re.search(r"变了|改变|变化|以前|之前是|used to|changed|不再", query, re.IGNORECASE):
            tense = "delta"
        elif re.search(r"先.+还是|before or after|哪一个先|哪个先|顺序", query, re.IGNORECASE):
            tense = "order"
        elif as_of or re.search(r"的时候|之时|as of|at that time|时喜欢|时的|当时", query, re.IGNORECASE):
            tense = "as-of"
        elif re.search(r"现在|目前|当前|如今|currently|right now", query, re.IGNORECASE):
            tense = "current"
        else:
            tense = "none"
        hops = 2 if re.search(r"以及|并且|both|and also", query, re.IGNORECASE) else 1
        recall = f"我记得相关的事：{query.strip()}"
        return Analysis(tense=tense, as_of=as_of, hops=hops, recall_q=recall)

    def judge(self, new: FactDraft, candidates: list[dict]) -> Verdict:
        for cand in candidates:
            if cand.get("status") not in {None, "active"}:
                continue
            if cand.get("valid_to"):
                continue
            same_referent = (
                _is_same_spo(new, cand)
                or bool(cand.get("same_attribute"))
                or same_mutable_attribute(
                    new.subject,
                    new.predicate,
                    new.object,
                    new.statement,
                    str(cand.get("subject") or ""),
                    str(cand.get("predicate") or ""),
                    str(cand.get("object") or ""),
                    str(cand.get("statement") or ""),
                )
                or _as_float(cand.get("sim")) >= SIM_SAME_REFERENT
            )
            if not same_referent:
                continue
            if _values_conflict(new, cand):
                target = cand.get("id")
                if target is None:
                    continue
                return Verdict("updates", int(target))
        return Verdict("unrelated", None)


def _values_conflict(new: FactDraft, cand: dict) -> bool:
    left = (new.object or "").strip()
    right = str(cand.get("object") or "").strip()
    if left and right:
        return not _same(left, right)
    if not left and not right:
        left_stmt = (new.statement or "").strip()
        right_stmt = str(cand.get("statement") or "").strip()
        return bool(left_stmt and right_stmt and not _same(left_stmt, right_stmt))
    return False


def _same(left: object, right: object) -> bool:
    return str(left or "").casefold().strip() == str(right or "").casefold().strip()


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _is_same_spo(new: FactDraft, cand: dict) -> bool:
    if cand.get("same_spo") is True:
        return True
    return bool(new.subject) and bool(new.predicate) and _same(cand.get("subject"), new.subject) and _same(
        cand.get("predicate"), new.predicate
    )


def _as_of_from_query(query: str) -> str | None:
    iso = re.search(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?", query)
    if iso:
        raw = iso.group(0)
        if "T" not in raw and " " not in raw:
            raw = raw + "T00:00:00Z"
        elif raw.endswith("Z") is False and "T" in raw and "+" not in raw:
            if len(raw) == 16:
                raw = raw + ":00Z"
            elif not raw.endswith("Z"):
                raw = raw + "Z"
        try:
            return to_iso(parse_iso(raw.replace(" ", "T")))
        except ValueError:
            return None
    cn = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", query)
    if cn:
        return f"{int(cn.group(1)):04d}-{int(cn.group(2)):02d}-{int(cn.group(3)):02d}T00:00:00Z"
    return None


class OpenAILLM:
    """Chat-completions client pinned to gpt-4o-mini unless LLM_MODEL overrides it."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise UpstreamError("OPENAI_API_KEY is not configured")
        self.model = model
        self._fallback = HeuristicLLM()
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self._url = base_url.rstrip("/") + "/chat/completions"

    def close(self) -> None:
        self._client.close()

    def extract(self, turns: list[TurnView]) -> list[FactDraft]:
        payload = {
            "turns": [
                {"id": t.id, "role": t.role, "content": t.content, "ts_event": t.ts_event} for t in turns
            ]
        }
        try:
            data = self._complete(EXTRACT_SYSTEM, json.dumps(payload, ensure_ascii=False))
        except (UpstreamError, ValueError, KeyError) as exc:
            log.warning("fact extract fell back to heuristic: %s", exc)
            return self._fallback.extract(turns)
        facts = data.get("facts") or []
        by_id = {t.id: t for t in turns}
        drafts: list[FactDraft] = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement") or "").strip()
            if not statement:
                continue
            src = item.get("src_turn_id")
            src_id = int(src) if isinstance(src, int) else (turns[0].id if turns else None)
            ts = item.get("ts_event") or (by_id[src_id].ts_event if src_id in by_id else None)
            provenance = item.get("provenance") if item.get("provenance") in {"stated", "inferred"} else "stated"
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            drafts.append(
                FactDraft(
                    statement=statement,
                    subject=str(item.get("subject") or ""),
                    predicate=str(item.get("predicate") or ""),
                    object=str(item.get("object") or ""),
                    ts_event=str(ts) if ts else None,
                    provenance=provenance,
                    confidence=max(0.0, min(1.0, confidence)),
                    sensitivity=sensitivity_of(statement, str(item.get("sensitivity") or "normal")),
                    src_turn_id=src_id,
                )
            )
        return drafts or self._fallback.extract(turns)

    def analyze(self, query: str) -> Analysis:
        try:
            data = self._complete(ANALYZE_SYSTEM, query)
        except (UpstreamError, ValueError, KeyError) as exc:
            log.warning("analyze fell back to heuristic: %s", exc)
            return self._fallback.analyze(query)
        tense = str(data.get("tense") or "none")
        if tense not in {"current", "as-of", "delta", "order", "none"}:
            tense = "none"
        as_of = data.get("as_of")
        as_of_s = str(as_of) if as_of else None
        try:
            hops = int(data.get("hops") or 1)
        except (TypeError, ValueError):
            hops = 1
        recall = str(data.get("recall_q") or "").strip() or f"我记得相关的事：{query.strip()}"
        return Analysis(tense=tense, as_of=as_of_s, hops=1 if hops < 2 else 2, recall_q=recall)

    def judge(self, new: FactDraft, candidates: list[dict]) -> Verdict:
        payload = {
            "new": {
                "statement": new.statement,
                "subject": new.subject,
                "predicate": new.predicate,
                "object": new.object,
                "ts_event": new.ts_event,
            },
            "candidates": candidates,
        }
        try:
            data = self._complete(JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False))
        except (UpstreamError, ValueError, KeyError) as exc:
            log.warning("judge fell back to heuristic: %s", exc)
            return self._fallback.judge(new, candidates)
        kind = str(data.get("kind") or "unrelated")
        if kind not in {"updates", "contradicts", "unrelated"}:
            kind = "unrelated"
        target = data.get("target_id")
        target_id = int(target) if isinstance(target, int) or (isinstance(target, str) and target.isdigit()) else None
        return Verdict(kind=kind, target_id=target_id)

    def _complete(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._client.post(self._url, json=body)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"llm request failed: {exc.__class__.__name__}") from exc
        if response.status_code >= 400:
            raise UpstreamError(f"llm upstream HTTP {response.status_code}")
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise UpstreamError("llm response missing content") from exc
        return parse_json_object(content)


def build_llm(settings) -> LLM:
    if settings.fake_models:
        return HeuristicLLM()
    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY missing; fact extract and tense labels use the local heuristic")
        return HeuristicLLM()
    return OpenAILLM(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
    )
