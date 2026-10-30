from mneme.scoring import combine, rrf_fuse, src_bonus, time_fit


def test_rrf_uses_k60_and_one_based_ranks():
    single = rrf_fuse([["a", "b"]], k=60)
    assert single["a"] == 1 / 61
    assert single["b"] == 1 / 62

    fused = rrf_fuse([["a", "b"], ["a"], ["b"]], k=60)
    assert fused["a"] == (1 / 61) + (1 / 61)
    assert fused["b"] == (1 / 62) + (1 / 61)
    assert fused["a"] > fused["b"]


def test_rrf_skips_empty_lists():
    scores = rrf_fuse([[], ["only"]], k=60)
    assert scores == {"only": 1 / 61}


def test_time_fit_current_and_as_of():
    assert (
        time_fit(
            tense="current",
            kind="fact",
            status="active",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            ts_event="2024-01-01T00:00:00Z",
            as_of=None,
            in_update_pair=False,
            order_position=0.0,
        )
        == 1.0
    )
    assert (
        time_fit(
            tense="current",
            kind="fact",
            status="superseded",
            valid_from="2024-01-01T00:00:00Z",
            valid_to="2025-01-01T00:00:00Z",
            ts_event="2024-01-01T00:00:00Z",
            as_of=None,
            in_update_pair=False,
            order_position=0.0,
        )
        == -1.0
    )
    assert (
        time_fit(
            tense="as-of",
            kind="fact",
            status="superseded",
            valid_from="2024-01-01T00:00:00Z",
            valid_to="2025-01-01T00:00:00Z",
            ts_event="2024-01-01T00:00:00Z",
            as_of="2024-06-01T00:00:00Z",
            in_update_pair=False,
            order_position=0.0,
        )
        == 1.0
    )
    assert (
        time_fit(
            tense="none",
            kind="fact",
            status="active",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            ts_event="2024-01-01T00:00:00Z",
            as_of=None,
            in_update_pair=False,
            order_position=0.0,
        )
        == 0.0
    )


def test_combine_matches_written_formula():
    score = combine(rerank=0.5, fit=1.0, source=1.0, beta=0.5, gamma=0.2)
    assert score == 0.5 + 0.5 * 1.0 + 0.2 * 1.0


def test_src_bonus_prefers_stated_over_inferred():
    stated = src_bonus(1.0, "stated", "2020-01-01T00:00:00Z")
    inferred = src_bonus(1.0, "inferred", "2020-01-01T00:00:00Z")
    assert stated > inferred
