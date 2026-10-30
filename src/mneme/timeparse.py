"""Event-time parsing. A memory never stores an empty event time.

Absolute dates win. Relative phrases anchor to the message timestamp, then to
ingest time. Rules cover the cases the blueprint calls out; gpt-4o-mini is only
asked when rules return nothing but the text still looks temporal.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_ISO = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(Z|[+-]\d{2}:?\d{2})?)?"
)
_CN_ABS = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?")
_CN_NUM = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_REL = re.compile(
    r"([0-9]+|[零一二两三四五六七八九十]+)\s*(天|周|星期|个月|月)\s*(前|后|之后|以后)"
)
_EN_REL = re.compile(
    r"(\d+)\s+days?\s+(ago|later|after)|(\d+)\s+weeks?\s+(ago|later|after)",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_EN_MONTH = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})",
    re.IGNORECASE,
)


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def from_unix_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _cn_number(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    if raw in _CN_NUM and raw != "十":
        return _CN_NUM[raw]
    if raw == "十":
        return 10
    if raw.startswith("十") and len(raw) == 2:
        return 10 + _CN_NUM.get(raw[1], 0)
    if "十" in raw:
        left, right = raw.split("十", 1)
        tens = _CN_NUM.get(left, 1) if left else 1
        ones = _CN_NUM.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _shift_months(anchor: datetime, months: int) -> datetime:
    month_index = anchor.month - 1 + months
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    day = min(anchor.day, _days_in_month(year, month))
    return anchor.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    cur = datetime(year, month, 1, tzinfo=timezone.utc)
    return (nxt - cur).days


def _apply_unit(anchor: datetime, n: int, unit: str, forward: bool) -> datetime:
    sign = 1 if forward else -1
    if unit in {"天"}:
        return anchor + timedelta(days=sign * n)
    if unit in {"周", "星期"}:
        return anchor + timedelta(days=sign * 7 * n)
    if unit in {"月", "个月"}:
        return _shift_months(anchor, sign * n)
    return anchor


def parse_event_time(content: str, anchor: datetime) -> datetime:
    """Return an aware UTC event time. Falls back to ``anchor`` and never None."""
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    else:
        anchor = anchor.astimezone(timezone.utc)
    text = content or ""

    iso = _ISO.search(text)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
        hour = int(iso.group(4) or anchor.hour)
        minute = int(iso.group(5) or 0)
        second = int(iso.group(6) or 0)
        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            pass

    cn_abs = _CN_ABS.search(text)
    if cn_abs:
        try:
            return datetime(
                int(cn_abs.group(1)),
                int(cn_abs.group(2)),
                int(cn_abs.group(3)),
                anchor.hour,
                anchor.minute,
                anchor.second,
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass

    en_month = _EN_MONTH.search(text)
    if en_month:
        month = _MONTHS[en_month.group(1).lower()]
        try:
            return datetime(
                int(en_month.group(3)),
                month,
                int(en_month.group(2)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass

    fixed = (
        ("前天", timedelta(days=-2)),
        ("昨天", timedelta(days=-1)),
        ("明天", timedelta(days=1)),
        ("后天", timedelta(days=2)),
        ("上周", timedelta(days=-7)),
        ("上星期", timedelta(days=-7)),
        ("下周", timedelta(days=7)),
        ("下星期", timedelta(days=7)),
        ("上个月", None),
        ("下个月", None),
    )
    for phrase, delta in fixed:
        if phrase in text:
            if phrase == "上个月":
                return _shift_months(anchor, -1)
            if phrase == "下个月":
                return _shift_months(anchor, 1)
            assert delta is not None
            return anchor + delta

    lowered = text.lower()
    en_fixed = (
        ("day before yesterday", timedelta(days=-2)),
        ("yesterday", timedelta(days=-1)),
        ("tomorrow", timedelta(days=1)),
        ("last week", timedelta(days=-7)),
        ("next week", timedelta(days=7)),
    )
    for phrase, delta in en_fixed:
        if phrase in lowered:
            return anchor + delta

    rel = _REL.search(text)
    if rel:
        n = _cn_number(rel.group(1))
        if n is not None:
            forward = rel.group(3) in {"后", "之后", "以后"}
            return _apply_unit(anchor, n, rel.group(2), forward)

    en_rel = _EN_REL.search(text)
    if en_rel:
        if en_rel.group(1):
            n = int(en_rel.group(1))
            forward = en_rel.group(2).lower() in {"later", "after"}
            return anchor + timedelta(days=n if forward else -n)
        n = int(en_rel.group(3))
        forward = en_rel.group(4).lower() in {"later", "after"}
        return anchor + timedelta(days=7 * (n if forward else -n))

    return anchor


def looks_temporal(content: str) -> bool:
    """True when the text likely contains a time the rules did not fully read."""
    if not content:
        return False
    if _ISO.search(content) or _CN_ABS.search(content) or _REL.search(content):
        return False
    hints = ("时候", "期间", "那年", "当时", "as of", "back when", "at the time")
    lowered = content.lower()
    return any(h in content or h in lowered for h in hints)
