# 同月记忆

同月记忆 (tongyue-memory) is a textual memory HTTP service for [Agent Memory Challenge Cycle 2](https://agentmemoryleaderboard.ai/), academic track, open-source methods. Version **0.1.0**. Register and cite this system as **同月记忆** / **tongyue-memory**.

The importable Python package is still `mneme` (`python -m mneme`, `MNEME_*` settings, SQLite file `mneme.db`). That module name is historical and is not the product name.

It stores what was said, retrieves evidence for a question, and does not answer the question. The platform's Answer model reads the `data` array.

## 部署清单

评测前确认这三件事：

1. `MNEME_FAKE_MODELS` 关闭。嵌入用 DashScope 兼容接口上的 **text-embedding-v4**（1024 维，`text_type=document|query`）。抽取、时态分类、回忆改写、冲突判断用 **gpt-4o-mini**。
2. 进程只提供 HTTP。TLS 放在 Caddy 或 Nginx 上。平台要求公网 HTTPS，密钥不要写进 URL。
3. `MEMORY_SYSTEM_KEY` 与申请 Eval Key 时登记的 Memory System Key 相同。`/add` 和 `/search` 接受 `Authorization: Bearer`、`Authorization: Token`、`X-Api-Key`。`GET /health` 不鉴权，任意 2xx 即健康。

## Contract

| Method | Path | Auth | Success |
| --- | --- | --- | --- |
| `GET` | `/health` | none | any 2xx, body includes `status` |
| `POST` | `/add` | key, if configured | `200` and `{"success": true, "request_id", "user_id", "session_id"}` |
| `POST` | `/search` | key, if configured | `200` and `{"data": [{"id", "content", "score", "created_at"}]}` |

Add returns 200 only after the verbatim turns, their embeddings, and the FTS rows are committed. Search can see that text immediately. Fact extraction waits up to `MNEME_FACT_TIMEOUT_S` (default 12s). If it overruns, the chunk stays on an offline queue and Add still returns 200.

The same `request_id` with the same body returns 200 and does not write again. The same `request_id` with a different body returns 409.

Search filters on `user_id` only. `session_id` is stored as provenance and is not a filter. The response has at most `top_k` items (formal eval uses 100). Empty evidence is `{"data": []}`. `options` is accepted and ignored: choice text is not memory and is not a label.

Search never emits an answer. `content` is a verbatim turn or an atomic fact drawn from those turns, with an ISO-8601 timestamp prefix so the Answer model can see when it was said.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export MNEME_FAKE_MODELS=1          # offline smoke; do not use this for eval
export MNEME_DB=./mneme.db
python -m mneme                    # listens on 0.0.0.0:8097
# equivalent console script after install: tongyue-memory
```

With real models, leave `MNEME_FAKE_MODELS` unset and export `DASHSCOPE_API_KEY` and `OPENAI_API_KEY`. See `.env.example`.

```bash
curl -sS http://127.0.0.1:8097/health

curl -sS http://127.0.0.1:8097/add \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "local-1",
    "user_id": "ge",
    "session_id": "s1",
    "messages": [{"role": "user", "timestamp": 1704067200000, "content": "My cat is named Luna."}]
  }'

curl -sS http://127.0.0.1:8097/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is my cat named?", "user_id": "ge", "top_k": 5}'
```

When `MEMORY_SYSTEM_KEY` is set, send one of:

```bash
-H "Authorization: Bearer $MEMORY_SYSTEM_KEY"
-H "Authorization: Token $MEMORY_SYSTEM_KEY"
-H "X-Api-Key: $MEMORY_SYSTEM_KEY"
```

Tests with fake models (no network, no keys):

```bash
pytest
```

`pytest -m "not real_llm and not integration"` is the offline suite, including `tests/test_live_failure_shapes.py` (the six shapes that failed on live gpt-4o-mini: city leak, empty job search, pet / residence / director drift, and the as-of city timeline). `pytest` from the repo root adds `src` to `PYTHONPATH` (see `pyproject.toml`). `pytest -m integration` calls DashScope once and is skipped unless `DASHSCOPE_API_KEY` or `EMBEDDING_API_KEY` is set.

Real gpt-4o-mini regression (17 cases: current-state updates, as-of, wording drift, extract/supersede). The harness forces `fake_models=False` and does not read `MNEME_FAKE_MODELS`. It skips unless both an LLM key and an embedding key are exported. Nothing in the repo stores those keys. Re-run it with:

```bash
# Chat: tryallapi gateway, or https://api.openai.com/v1
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://tryallapi.com/v1"
export LLM_MODEL="gpt-4o-mini"

# Embeddings stay on DashScope text-embedding-v4 (1024d)
export DASHSCOPE_API_KEY="..."
export EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export EMBEDDING_MODEL="text-embedding-v4"
export EMBEDDING_DIM="1024"

pytest -m real_llm
```

International DashScope hosts can set `EMBEDDING_BASE_URL` to the `dashscope-intl` compatible-mode `/v1` URL instead. Omit `OPENAI_BASE_URL` to use `https://api.openai.com/v1`.

## Docker

The image is the app only. Terminate TLS at a reverse proxy.

```bash
cp .env.example .env
# fill DASHSCOPE_API_KEY, OPENAI_API_KEY, MEMORY_SYSTEM_KEY
docker compose up --build -d
curl -sS http://127.0.0.1:8097/health
```

SQLite lives in the `mneme-data` volume (`/data/mneme.db`). Run one process. The store uses a process lock around SQLite; extra uvicorn workers would serialize on that lock and are not the intended deploy.

Caddy, on the host:

```caddy
memory.example.com {
    reverse_proxy 127.0.0.1:8097
}
```

Nginx, with the platform's 30 minute Add/Search budget:

```nginx
server {
    listen 443 ssl;
    server_name memory.example.com;
    # ssl_certificate and ssl_certificate_key here

    location / {
        proxy_pass http://127.0.0.1:8097;
        proxy_read_timeout 1800s;
        proxy_send_timeout 1800s;
    }
}
```

Submit the public `https://memory.example.com/add` and `/search` URLs plus `GET /health`. Register the same `MEMORY_SYSTEM_KEY` and the matching scheme (Bearer, Token, or X-Api-Key).

## Environment

| Variable | Default | Role |
| --- | --- | --- |
| `MEMORY_SYSTEM_KEY` | unset | If set, required on `/add` and `/search`. `/health` ignores it. |
| `DASHSCOPE_API_KEY` / `EMBEDDING_API_KEY` | unset | Key for text-embedding-v4. Missing key: health stays up, Add returns 503. |
| `EMBEDDING_BASE_URL` | DashScope compatible-mode `/v1` | OpenAI-compatible base. International hosts can point this at `dashscope-intl`. |
| `EMBEDDING_MODEL` | `text-embedding-v4` | Locked for the academic track. |
| `EMBEDDING_DIM` | `1024` | Sent as `dimensions`. |
| `OPENAI_API_KEY` | unset | gpt-4o-mini. If unset, extraction falls back to a local heuristic. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Chat completions base. |
| `LLM_MODEL` | `gpt-4o-mini` | Locked for the academic track. |
| `MNEME_RERANKER` | `lexical` | `lexical` or `qwen3`. |
| `QWEN_RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Used only when the reranker is `qwen3`. |
| `MNEME_TAU` | `0.25` | Abstain when the best combined score is below this. |
| `MNEME_BETA` / `MNEME_GAMMA` | `0.5` / `0.2` | Weights on time fit and source bonus. |
| `MNEME_FACT_TIMEOUT_S` | `12` | Bounded wait for fact extraction inside Add. |
| `MNEME_DB` | `mneme.db` | SQLite path. WAL is on. |
| `MNEME_FTS_SIMPLE` | unset | Path of the `simple` FTS5 extension, if you have it. |
| `MNEME_FAKE_MODELS` | `0` | `1` uses a hash embedder and rule-based LLM. Not for eval. |
| `PORT` / `HOST` | `8097` / `0.0.0.0` | Bind address. |

Embedding requests are batches of 10, with at most 16 batches in flight, and each call sends `text_type` of `document` or `query`. An embedding failure before the write returns **503** with `Retry-After: 2` and leaves no partial chunk.

## What the pipeline does

**Add.** Parse an event time from the text (absolute dates, then relatives such as `三天后` / `last week`, anchored on the message timestamp). If no time is present, the message timestamp or ingest time is stored. Event time is never null. Embed with `text_type=document`, then commit turns + FTS. A background step asks gpt-4o-mini for atomic facts and reconciles conflicts in bitemporal form: the old fact gets `valid_to`, `status=superseded`, and `superseded_by`. It is not deleted.

**Search.** Classify the question as current, as-of, delta, order, or none, and rewrite a first-person recall probe. Neither step is allowed to answer. Recall is dense over turns and facts for the question, dense again for the recall probe, and sparse BM25, fused with RRF at k=60. A temporal filter then drops superseded facts for "current" questions, keeps the valid interval for "as-of", and keeps both sides of an update for "delta". The reranker scores survivors. Final score is

```text
rerank + β · time_fit + γ · src_bonus
```

Items under `MNEME_RERANK_FLOOR` (default 0.08) are dropped so a time bonus cannot revive an unrelated memory. Abstention looks at pure rerank relevance, not the combined score: if the best relevance is below `TAU`, Search returns `{"data": []}`. Time fit and source bonus still change order among survivors, and the returned `score` is that combined value. Otherwise neighbors (one turn on either side of a hit, same chunk) are appended and inherit the anchor's relevance, then the list is cut to `top_k`. Verbatim turns are ordered before atomic facts. Sensitive text (health, identity documents, passwords) is omitted when the question does not mention it.

A current-tense question also drops an older turn that states the same subject and predicate as a later turn (`X的Y是Z` / `my X is Y`), even when fact extraction has not finished. Those turn ids are hidden for this search only, including when neighbor expansion would otherwise copy them back in. Conflict judgment treats two facts as the same referent when the subject and predicate strings match, or when their embedding cosine is at least 0.82.

## Reranker

The default reranker is lexical overlap. It needs no GPU and no download, which is what CI runs.

To use Qwen3-Reranker-0.6B:

```bash
pip install -e '.[qwen]'
# distribution name, once published or installed by name: tongyue-memory[qwen]
export MNEME_RERANKER=qwen3
```

The class loads `Qwen/Qwen3-Reranker-0.6B` and scores each passage by P(yes). The first call downloads the weights. If torch or transformers is missing, startup fails with the install command instead of silently scoring with the lexical model. The weights are Apache-2.0; the image published here does not bundle them.

## Full-text Chinese

FTS5's default unicode61 tokenizer treats a run of Han characters as one token, so Chinese queries miss. 同月记忆 does not index raw Chinese that way.

- If `MNEME_FTS_SIMPLE` points at a loadable [wangfenjin/simple](https://github.com/wangfenjin/simple) extension, the FTS table uses `tokenize='simple'`.
- Otherwise the indexed text is jieba tokens separated by spaces, and a second FTS5 `trigram` table indexes the raw string. `GET /health` reports the mode (`jieba+trigram`, `jieba`, or `simple`).

## Layout

```text
src/mneme/app.py        HTTP, auth
src/mneme/service.py    Add / Search pipelines
src/mneme/store.py      SQLite WAL, FTS, bitemporal facts
src/mneme/scoring.py    RRF k=60 and the final score
src/mneme/embeddings.py text-embedding-v4 client
src/mneme/llm.py        gpt-4o-mini client
src/mneme/rerank.py     lexical default, Qwen3 optional
```

## Attribution

The dual store (verbatim turns plus atomic facts), radius-1 neighbor expansion, and dual-channel recall (original question plus a first-person recall probe) follow the recipe measured by [ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex) (linxuhao; Cycle 1 open-source textual). 同月记忆 does not fork that repository.

Bitemporal validity (`valid_from`, `valid_to`, `superseded`), the cross-encoder rerank slot, and abstention below `TAU` are part of 同月记忆. The interval model follows the idea in Zep/Graphiti (arXiv 2501.13956), not that codebase. RRF with k=60 is Cormack, Clarke, and Buettcher, SIGIR 2009. The Qwen3 reranker, when enabled, is QwenLM/Qwen3-Embedding (Apache-2.0).

## License

No license is granted by this repository until one is added. The optional Qwen3 weights remain under their own Apache-2.0 terms.
