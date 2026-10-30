from datetime import datetime, timezone

from mneme.timeparse import from_unix_ms, parse_event_time, to_iso


ANCHOR = datetime(2026, 3, 1, 8, 0, tzinfo=timezone.utc)


def test_absolute_iso_and_chinese_date():
    assert to_iso(parse_event_time("see you 2026-03-05", ANCHOR)) == "2026-03-05T08:00:00Z"
    assert to_iso(parse_event_time("2026年3月5日见面", ANCHOR)) == "2026-03-05T08:00:00Z"


def test_relative_chinese_and_english():
    assert to_iso(parse_event_time("三天后出发", ANCHOR)) == "2026-03-04T08:00:00Z"
    assert to_iso(parse_event_time("上周的安排", ANCHOR)) == "2026-02-22T08:00:00Z"
    assert to_iso(parse_event_time("3 days ago", ANCHOR)) == "2026-02-26T08:00:00Z"
    assert to_iso(parse_event_time("yesterday", ANCHOR)) == "2026-02-28T08:00:00Z"


def test_falls_back_to_anchor_and_never_blank():
    got = parse_event_time("no time here", ANCHOR)
    assert to_iso(got) == "2026-03-01T08:00:00Z"
    assert to_iso(from_unix_ms(1704067200000)) == "2024-01-01T00:00:00Z"
