"""SQLite store partitioned by user_id.

Tables: turns (verbatim), facts (atomic, bitemporal), FTS, event edges,
idempotency log, enrich jobs. WAL is on. ``session_id`` is stored on turns
and is never a Search predicate.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from mneme.tokenize import join_tokens, tokenize

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  ts_event TEXT NOT NULL,
  ts_ingest TEXT NOT NULL,
  emb BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_chunk ON turns(user_id, chunk_id, ordinal);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL,
  src_turn_id INTEGER,
  subject TEXT,
  predicate TEXT,
  object TEXT,
  statement TEXT NOT NULL,
  ts_event TEXT NOT NULL,
  ts_ingest TEXT NOT NULL,
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  superseded_by INTEGER,
  confidence REAL NOT NULL DEFAULT 1.0,
  provenance TEXT NOT NULL DEFAULT 'stated',
  sensitivity TEXT NOT NULL DEFAULT 'normal',
  emb BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, status);
CREATE TABLE IF NOT EXISTS event_edges (
  id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL,
  a_fact INTEGER NOT NULL,
  b_fact INTEGER NOT NULL,
  rel TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_user ON event_edges(user_id, a_fact, b_fact);
CREATE TABLE IF NOT EXISTS add_requests (
  request_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  body_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS enrich_jobs (
  chunk_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  started_at TEXT,
  last_error TEXT,
  PRIMARY KEY (chunk_id, user_id)
);
"""


@dataclass
class StoredTurn:
    id: int
    user_id: str
    session_id: str
    chunk_id: str
    ordinal: int
    role: str
    content: str
    ts_event: str
    ts_ingest: str


@dataclass
class StoredFact:
    id: int
    user_id: str
    src_turn_id: int | None
    subject: str
    predicate: str
    object: str
    statement: str
    ts_event: str
    ts_ingest: str
    valid_from: str
    valid_to: str | None
    status: str
    superseded_by: int | None
    confidence: float
    provenance: str
    sensitivity: str


@dataclass
class NewTurn:
    ordinal: int
    role: str
    content: str
    ts_event: str
    emb: bytes


class Conflict(Exception):
    """Same request_id was already stored with a different body."""


class Store:
    def __init__(self, path: str, fts_simple_path: str | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self.fts_mode, self.has_trigram = self._ensure_fts(fts_simple_path)
        self._recover_jobs()
        log.info("fts mode=%s trigram=%s db=%s", self.fts_mode, self.has_trigram, path)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _ensure_fts(self, simple_path: str | None) -> tuple[str, bool]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'fts_mode'").fetchone()
        tri = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fts_tri'"
        ).fetchone()
        if row:
            return row["value"], bool(tri)
        mode = "jieba+trigram"
        has_trigram = False
        if simple_path:
            try:
                self.conn.enable_load_extension(True)
                self.conn.load_extension(simple_path)
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE fts USING fts5(
                      content,
                      user_id UNINDEXED,
                      doc_key UNINDEXED,
                      kind UNINDEXED,
                      tokenize = 'simple'
                    )
                    """
                )
                mode = "simple"
            except (sqlite3.OperationalError, sqlite3.DatabaseError, AttributeError) as exc:
                log.warning("FTS simple extension failed (%s); using jieba + trigram", exc)
        if mode != "simple":
            # Indexed column is jieba tokens joined by spaces, never raw CJK.
            # unicode61 then splits on whitespace. A second trigram table covers
            # substrings when the simple extension is absent.
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE fts USING fts5(
                  tokens,
                  user_id UNINDEXED,
                  doc_key UNINDEXED,
                  kind UNINDEXED,
                  tokenize = "unicode61 remove_diacritics 2"
                )
                """
            )
            try:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE fts_tri USING fts5(
                      content,
                      user_id UNINDEXED,
                      doc_key UNINDEXED,
                      kind UNINDEXED,
                      tokenize = "trigram case_sensitive 0"
                    )
                    """
                )
                has_trigram = True
                mode = "jieba+trigram"
            except sqlite3.OperationalError as exc:
                log.warning("FTS5 trigram unavailable (%s); jieba token index only", exc)
                mode = "jieba"
                has_trigram = False
        self.conn.execute("INSERT INTO meta(key, value) VALUES ('fts_mode', ?)", (mode,))
        return mode, has_trigram

    def _recover_jobs(self) -> None:
        cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Jobs left running by a crash go back to pending. Fresh running jobs
        # from this process are not old; startup is the only caller.
        self.conn.execute(
            "UPDATE enrich_jobs SET status = 'pending', last_error = 'recovered' WHERE status = 'running'"
        )
        del cutoff

    def has_user(self, user_id: str) -> bool:
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM turns WHERE user_id = ? LIMIT 1", (user_id,)).fetchone()
            return row is not None

    def get_request(self, request_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT request_id, user_id, session_id, body_hash FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()

    def persist_turns(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        body_hash: str,
        turns: list[NewTurn],
        ts_ingest: str,
    ) -> str:
        """Insert a chunk once. Returns 'created', 'replay', or raises Conflict.

        The raw turns, their embeddings, and FTS rows commit together, so a
        200 after this returns means Search can already see the text.
        """
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self.conn.execute(
                    "SELECT body_hash FROM add_requests WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if existing is not None:
                    self.conn.rollback()
                    if existing["body_hash"] != body_hash:
                        raise Conflict(request_id)
                    return "replay"
                for turn in turns:
                    cur = self.conn.execute(
                        """
                        INSERT INTO turns (
                          user_id, session_id, chunk_id, ordinal, role, content,
                          ts_event, ts_ingest, emb
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            session_id,
                            request_id,
                            turn.ordinal,
                            turn.role,
                            turn.content,
                            turn.ts_event,
                            ts_ingest,
                            turn.emb,
                        ),
                    )
                    self._index(f"t:{cur.lastrowid}", user_id, "turn", turn.content)
                self.conn.execute(
                    """
                    INSERT INTO add_requests (request_id, user_id, session_id, body_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (request_id, user_id, session_id, body_hash, ts_ingest),
                )
                self.conn.execute(
                    """
                    INSERT INTO enrich_jobs (chunk_id, user_id, status, started_at, last_error)
                    VALUES (?, ?, 'pending', NULL, NULL)
                    ON CONFLICT(chunk_id, user_id) DO NOTHING
                    """,
                    (request_id, user_id),
                )
                self.conn.commit()
                return "created"
            except Exception:
                self.conn.rollback()
                raise

    def _index(self, doc_key: str, user_id: str, kind: str, raw_text: str) -> None:
        if self.fts_mode == "simple":
            self.conn.execute(
                "INSERT INTO fts (content, user_id, doc_key, kind) VALUES (?, ?, ?, ?)",
                (raw_text, user_id, doc_key, kind),
            )
            return
        tokens = join_tokens(raw_text)
        self.conn.execute(
            "INSERT INTO fts (tokens, user_id, doc_key, kind) VALUES (?, ?, ?, ?)",
            (tokens, user_id, doc_key, kind),
        )
        if self.has_trigram and len(raw_text) >= 3:
            self.conn.execute(
                "INSERT INTO fts_tri (content, user_id, doc_key, kind) VALUES (?, ?, ?, ?)",
                (raw_text, user_id, doc_key, kind),
            )

    def fts_tokens(self, doc_key: str) -> str | None:
        if self.fts_mode == "simple":
            return None
        with self._lock:
            row = self.conn.execute("SELECT tokens FROM fts WHERE doc_key = ?", (doc_key,)).fetchone()
            return None if row is None else row["tokens"]

    def claim_enrich(self, chunk_id: str, user_id: str) -> list[StoredTurn] | None:
        """Mark a pending job running and return its turns. None if not claimable."""
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM enrich_jobs WHERE chunk_id = ? AND user_id = ?",
                (chunk_id, user_id),
            ).fetchone()
            if row is None or row["status"] in {"done", "running"}:
                return None
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.conn.execute(
                """
                UPDATE enrich_jobs SET status = 'running', started_at = ?, last_error = NULL
                WHERE chunk_id = ? AND user_id = ? AND status = 'pending'
                """,
                (now, chunk_id, user_id),
            )
            if self.conn.execute("SELECT changes()").fetchone()[0] == 0:
                return None
            return self._load_chunk(chunk_id, user_id)

    def enrich_status(self, chunk_id: str, user_id: str) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM enrich_jobs WHERE chunk_id = ? AND user_id = ?",
                (chunk_id, user_id),
            ).fetchone()
            return None if row is None else row["status"]

    def mark_enrich_done(self, chunk_id: str, user_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE enrich_jobs SET status = 'done', last_error = NULL WHERE chunk_id = ? AND user_id = ?",
                (chunk_id, user_id),
            )

    def mark_enrich_pending(self, chunk_id: str, user_id: str, error: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE enrich_jobs
                SET attempts = attempts + 1,
                    last_error = ?,
                    status = CASE WHEN attempts + 1 >= 5 THEN 'error' ELSE 'pending' END
                WHERE chunk_id = ? AND user_id = ? AND status != 'done'
                """,
                (error[:500], chunk_id, user_id),
            )

    def claim_next_pending(self) -> tuple[str, str] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT chunk_id, user_id FROM enrich_jobs WHERE status = 'pending' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return row["chunk_id"], row["user_id"]

    def _load_chunk(self, chunk_id: str, user_id: str) -> list[StoredTurn]:
        rows = self.conn.execute(
            """
            SELECT id, user_id, session_id, chunk_id, ordinal, role, content, ts_event, ts_ingest
            FROM turns WHERE chunk_id = ? AND user_id = ? ORDER BY ordinal
            """,
            (chunk_id, user_id),
        ).fetchall()
        return [_turn(row) for row in rows]

    def load_chunk(self, chunk_id: str, user_id: str) -> list[StoredTurn]:
        with self._lock:
            return self._load_chunk(chunk_id, user_id)

    def vector_topk(self, user_id: str, query: np.ndarray, limit: int) -> list[str]:
        with self._lock:
            rows = []
            for table, prefix in (("turns", "t"), ("facts", "f")):
                fetched = self.conn.execute(
                    f"SELECT id, emb FROM {table} WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
                rows.append((prefix, fetched))
        scored: list[tuple[float, str]] = []
        q = np.asarray(query, dtype=np.float32)
        for prefix, fetched in rows:
            for row in fetched:
                blob = row["emb"]
                if not blob:
                    continue
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.shape[0] != q.shape[0]:
                    continue
                scored.append((float(np.dot(q, vec)), f"{prefix}:{row['id']}"))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out: list[str] = []
        seen: set[str] = set()
        for _score, key in scored:
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= limit:
                break
        return out

    def fts_topk(self, user_id: str, query: str, limit: int) -> list[str]:
        ranks: dict[str, int] = {}
        with self._lock:
            self._accumulate_fts(ranks, user_id, query, limit)
        ordered = sorted(ranks, key=lambda key: (ranks[key], key))
        return ordered[:limit]

    def _accumulate_fts(self, ranks: dict[str, int], user_id: str, query: str, limit: int) -> None:
        if self.fts_mode == "simple":
            expr = '"' + query.replace('"', '""') + '"'
            self._match_into(ranks, "fts", expr, user_id, limit)
        else:
            tokens = tokenize(query)
            if tokens:
                parts = ['"' + tok.replace('"', '""') + '"' for tok in tokens]
                expr = " OR ".join(parts)
                self._match_into(ranks, "fts", expr, user_id, limit)
            if self.has_trigram and len(query.strip()) >= 3:
                expr = '"' + query.replace('"', '""') + '"'
                self._match_into(ranks, "fts_tri", expr, user_id, limit, better_only=True)

    def _match_into(
        self,
        ranks: dict[str, int],
        table: str,
        expr: str,
        user_id: str,
        limit: int,
        better_only: bool = False,
    ) -> None:
        try:
            rows = self.conn.execute(
                f"""
                SELECT doc_key FROM {table}
                WHERE {table} MATCH ? AND user_id = ?
                ORDER BY bm25({table})
                LIMIT ?
                """,
                (expr, user_id, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("fts query failed on %s: %s", table, exc)
            return
        for index, row in enumerate(rows):
            key = row["doc_key"]
            if key not in ranks or (better_only and index < ranks[key]):
                ranks[key] = index
            elif not better_only and key not in ranks:
                ranks[key] = index

    def load_records(self, user_id: str, keys: list[str]) -> list[tuple[str, StoredTurn | StoredFact]]:
        turns: list[int] = []
        facts: list[int] = []
        for key in keys:
            if key.startswith("t:"):
                turns.append(int(key[2:]))
            elif key.startswith("f:"):
                facts.append(int(key[2:]))
        with self._lock:
            turn_map = self._fetch_turns(user_id, turns)
            fact_map = self._fetch_facts(user_id, facts)
        out: list[tuple[str, StoredTurn | StoredFact]] = []
        for key in keys:
            if key.startswith("t:") and int(key[2:]) in turn_map:
                out.append((key, turn_map[int(key[2:])]))
            elif key.startswith("f:") and int(key[2:]) in fact_map:
                out.append((key, fact_map[int(key[2:])]))
        return out

    def _fetch_turns(self, user_id: str, ids: list[int]) -> dict[int, StoredTurn]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT id, user_id, session_id, chunk_id, ordinal, role, content, ts_event, ts_ingest
            FROM turns WHERE user_id = ? AND id IN ({q})
            """,
            (user_id, *ids),
        ).fetchall()
        return {row["id"]: _turn(row) for row in rows}

    def _fetch_facts(self, user_id: str, ids: list[int]) -> dict[int, StoredFact]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM facts WHERE user_id = ? AND id IN ({q})",
            (user_id, *ids),
        ).fetchall()
        return {row["id"]: _fact(row) for row in rows}

    def neighbor_turn_ids(self, user_id: str, chunk_id: str, ordinal: int, radius: int) -> list[int]:
        if radius <= 0:
            return []
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id FROM turns
                WHERE user_id = ? AND chunk_id = ? AND ordinal BETWEEN ? AND ? AND ordinal != ?
                ORDER BY ordinal
                """,
                (user_id, chunk_id, ordinal - radius, ordinal + radius, ordinal),
            ).fetchall()
            return [row["id"] for row in rows]

    def fact_rows_for_reconcile(self, user_id: str) -> list[StoredFact]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM facts
                WHERE user_id = ? AND status = 'active' AND valid_to IS NULL
                """,
                (user_id,),
            ).fetchall()
            return [_fact(row) for row in rows]

    def fact_embeddings(self, user_id: str, limit_ids: list[int] | None = None) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            if limit_ids:
                q = ",".join("?" * len(limit_ids))
                rows = self.conn.execute(
                    f"SELECT id, emb FROM facts WHERE user_id = ? AND id IN ({q})",
                    (user_id, *limit_ids),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT id, emb FROM facts WHERE user_id = ? AND status = 'active' AND valid_to IS NULL",
                    (user_id,),
                ).fetchall()
        out: list[tuple[int, np.ndarray]] = []
        for row in rows:
            out.append((row["id"], np.frombuffer(row["emb"], dtype=np.float32).copy()))
        return out

    def insert_fact(
        self,
        *,
        user_id: str,
        src_turn_id: int | None,
        subject: str,
        predicate: str,
        object_: str,
        statement: str,
        ts_event: str,
        ts_ingest: str,
        valid_from: str,
        valid_to: str | None,
        status: str,
        superseded_by: int | None,
        confidence: float,
        provenance: str,
        sensitivity: str,
        emb: bytes,
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO facts (
                  user_id, src_turn_id, subject, predicate, object, statement,
                  ts_event, ts_ingest, valid_from, valid_to, status, superseded_by,
                  confidence, provenance, sensitivity, emb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    src_turn_id,
                    subject,
                    predicate,
                    object_,
                    statement,
                    ts_event,
                    ts_ingest,
                    valid_from,
                    valid_to,
                    status,
                    superseded_by,
                    confidence,
                    provenance,
                    sensitivity,
                    emb,
                ),
            )
            fact_id = int(cur.lastrowid)
            self._index(f"f:{fact_id}", user_id, "fact", statement)
            return fact_id

    def supersede_fact(self, user_id: str, fact_id: int, valid_to: str, superseded_by: int) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE facts
                SET valid_to = ?, status = 'superseded', superseded_by = ?
                WHERE id = ? AND user_id = ?
                """,
                (valid_to, superseded_by, fact_id, user_id),
            )

    def add_edge(self, user_id: str, a_fact: int, b_fact: int, rel: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO event_edges (user_id, a_fact, b_fact, rel) VALUES (?, ?, ?, ?)",
                (user_id, a_fact, b_fact, rel),
            )

    def update_pair_ids(self, user_id: str) -> set[int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT a_fact, b_fact FROM event_edges WHERE user_id = ? AND rel = 'updates'",
                (user_id,),
            ).fetchall()
            ids: set[int] = set()
            for row in rows:
                ids.add(row["a_fact"])
                ids.add(row["b_fact"])
            return ids

    def turn_ids_invalid_at(self, user_id: str, moment: str) -> set[int]:
        """Turns whose facts are all outside ``moment``. Turns with no facts are kept."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT src_turn_id, ts_event, valid_from, valid_to
                FROM facts
                WHERE user_id = ? AND src_turn_id IS NOT NULL
                """,
                (user_id,),
            ).fetchall()
        by_src: dict[int, list] = {}
        for row in rows:
            by_src.setdefault(int(row["src_turn_id"]), []).append(row)
        stale: set[int] = set()
        for src, facts in by_src.items():
            if facts and not any(_fact_row_valid_at(row, moment) for row in facts):
                stale.add(src)
        return stale

    def suppressed_turn_ids(self, user_id: str) -> set[int]:
        """Turns that only support a superseded fact. Current-tense search hides them."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT src_turn_id, status, valid_to FROM facts WHERE user_id = ? AND src_turn_id IS NOT NULL",
                (user_id,),
            ).fetchall()
        active: set[int] = set()
        superseded: set[int] = set()
        for row in rows:
            src = int(row["src_turn_id"])
            if row["status"] == "active" and row["valid_to"] is None:
                active.add(src)
            elif row["status"] == "superseded":
                superseded.add(src)
        return superseded - active

    def bridge_fact_ids(self, user_id: str, seed_fact_ids: list[int], limit: int) -> list[int]:
        if not seed_fact_ids:
            return []
        q = ",".join("?" * len(seed_fact_ids))
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT a_fact, b_fact FROM event_edges
                WHERE user_id = ? AND (a_fact IN ({q}) OR b_fact IN ({q}))
                """,
                (user_id, *seed_fact_ids, *seed_fact_ids),
            ).fetchall()
        seeds = set(seed_fact_ids)
        extra: list[int] = []
        seen: set[int] = set()
        for row in rows:
            for fact_id in (row["a_fact"], row["b_fact"]):
                if fact_id in seeds or fact_id in seen:
                    continue
                seen.add(fact_id)
                extra.append(fact_id)
                if len(extra) >= limit:
                    return extra
        return extra

    def count_turns(self, user_id: str | None = None) -> int:
        with self._lock:
            if user_id is None:
                row = self.conn.execute("SELECT COUNT(*) AS n FROM turns").fetchone()
            else:
                row = self.conn.execute("SELECT COUNT(*) AS n FROM turns WHERE user_id = ?", (user_id,)).fetchone()
            return int(row["n"])


def _fact_row_valid_at(row: sqlite3.Row, moment: str) -> bool:
    start = row["valid_from"] or row["ts_event"]
    valid_to = row["valid_to"]
    return start <= moment and (valid_to is None or moment < valid_to)


def _turn(row: sqlite3.Row) -> StoredTurn:
    return StoredTurn(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        chunk_id=row["chunk_id"],
        ordinal=row["ordinal"],
        role=row["role"],
        content=row["content"],
        ts_event=row["ts_event"],
        ts_ingest=row["ts_ingest"],
    )


def _fact(row: sqlite3.Row) -> StoredFact:
    return StoredFact(
        id=row["id"],
        user_id=row["user_id"],
        src_turn_id=row["src_turn_id"],
        subject=row["subject"] or "",
        predicate=row["predicate"] or "",
        object=row["object"] or "",
        statement=row["statement"],
        ts_event=row["ts_event"],
        ts_ingest=row["ts_ingest"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        status=row["status"],
        superseded_by=row["superseded_by"],
        confidence=row["confidence"],
        provenance=row["provenance"],
        sensitivity=row["sensitivity"],
    )
