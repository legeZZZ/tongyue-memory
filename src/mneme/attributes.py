"""Speaker-scoped mutable attributes.

Exact subject/predicate equality misses live extractor drift (``have`` vs
``has``, ``live in`` vs ``lives in``, ``最喜欢的导演是`` vs ``最推崇的电影导演``).
Embedding cosine often stays under 0.82 when the object changed too, so those
pairs never supersede.

Recognized families supersede on their own. Anything else, including the pinned
``最喜欢的导演`` / ``目前最爱的导演`` pair, still needs cosine >= 0.82. Unrelated
families (cat name vs city, cat count vs dog) do not match.
"""

from __future__ import annotations

import re

_RESIDENCE = re.compile(
    r"\b(?:used\s+to\s+|now\s+|currently\s+)?(?:live|lives|lived|living)\s+in\b"
    r"|住在|居住在|居住于|现居|搬到|居住(?!证)",
    re.IGNORECASE,
)
_JOB = re.compile(
    r"\bwork(?:s|ed|ing)?\s+as\b|\b(?:occupation|job)\s+is\b|职业|工作是|工作为",
    re.IGNORECASE,
)
_JOB_PREDICATE = re.compile(r"(?:jobs?|occupations?|职业|工作)", re.IGNORECASE)
_CAT_COUNT = re.compile(
    r"\b(?:have|has|had)\b(?:\s+\w+){0,4}\s+"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+cats?\b",
    re.IGNORECASE,
)
_DIRECTOR = re.compile(r"导演|director", re.IGNORECASE)
_DIRECTOR_FAVOR = re.compile(r"推崇|喜欢|favourite|favorite|admired", re.IGNORECASE)
_SPEAKER_WORD = re.compile(r"(?:the\s+)?(?:user|i|my)\b", re.IGNORECASE)
_SPEAKER_LABELS = {"i", "me", "my", "user", "the user", "我", "用户", "本人"}


def attribute_family(subject: str, predicate: str, object_: str, statement: str) -> str | None:
    """Family id, or None when the fact is not a known single-valued attribute."""
    pred = (predicate or "").strip()
    if pred and _JOB_PREDICATE.fullmatch(pred):
        return "job"
    pred_blob = " ".join(part.strip() for part in (predicate, object_) if part and part.strip())
    if pred_blob:
        found = _match_family(pred_blob)
        if found:
            return found
        combined = " ".join(
            part.strip() for part in (subject, predicate, object_) if part and part.strip()
        )
        return _match_family(combined)
    return _match_family(statement or "")


def canonical_subject(subject: str, statement: str = "") -> str:
    """Fold speaker labels (I / User / 我 / 用户). Other subjects stay distinct."""
    raw = (subject or "").strip()
    if raw and _is_speaker_label(raw):
        return "speaker"
    if not raw and _looks_like_speaker(statement):
        return "speaker"
    return raw.casefold()


def same_mutable_attribute(
    left_subject: str,
    left_predicate: str,
    left_object: str,
    left_statement: str,
    right_subject: str,
    right_predicate: str,
    right_object: str,
    right_statement: str,
) -> bool:
    left = attribute_family(left_subject, left_predicate, left_object, left_statement)
    right = attribute_family(right_subject, right_predicate, right_object, right_statement)
    if not left or left != right:
        return False
    left_who = canonical_subject(left_subject, left_statement)
    right_who = canonical_subject(right_subject, right_statement)
    return bool(left_who) and left_who == right_who


def text_referent(text: str) -> tuple[str, str] | None:
    """``(speaker, family)`` when ``text`` states one of the user's mutable attributes."""
    family = attribute_family("", "", "", text)
    if not family or canonical_subject("", text) != "speaker":
        return None
    return ("speaker", family)


def _match_family(blob: str) -> str | None:
    if not blob:
        return None
    if _DIRECTOR.search(blob) and _DIRECTOR_FAVOR.search(blob):
        return "favorite:director"
    if _CAT_COUNT.search(blob):
        return "pet-count:cat"
    if _JOB.search(blob):
        return "job"
    if _RESIDENCE.search(blob):
        return "residence"
    return None


def _is_speaker_label(raw: str) -> bool:
    folded = raw.casefold().strip()
    if folded in _SPEAKER_LABELS:
        return True
    if folded.startswith("我们"):
        return False
    if folded.startswith("我") or folded.startswith("用户"):
        return True
    return _SPEAKER_WORD.match(folded) is not None


def _looks_like_speaker(text: str) -> bool:
    head = (text or "").strip()
    if not head or head.startswith("我们"):
        return False
    if head.startswith("我") or head.startswith("用户"):
        return True
    return _SPEAKER_WORD.match(head) is not None
