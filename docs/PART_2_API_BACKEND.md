# Part 2 — API & Backend Engineering

## Objective

Serve the existing dense, sparse, and hybrid retrieval capabilities as an HTTP application without needing an interactive notebook. This is a **local development service**, not a production deployment.

## Request flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI / Pydantic
    participant Service as Async search service
    participant Model as Sentence Transformer
    participant DB as PostgreSQL + pgvector
    Client->>API: POST /search {question, brand, method, top_k}
    API->>API: Validate request
    API->>Service: search(...)
    opt method includes dense
        Service->>Model: Embed question in worker thread
        Model-->>Service: 384D query vector
        Service->>DB: Async cosine-distance query
        DB-->>Service: Dense candidates
    end
    opt method includes BM25
        Service->>DB: Async load matching corpus
        DB-->>Service: Text and IDs
        Service->>Service: BM25 scoring in worker thread
    end
    opt method is hybrid
        Service->>Service: RRF fuse dense + sparse ranks
    end
    Service-->>API: Selected ranked chunks
    API-->>Client: JSON response
```

## Module ownership

| Location | Responsibility |
|---|---|
| `src/ai_search/api.py` | FastAPI app, `GET /health`, `GET /health/db`, `POST /search`, `GET /analytics/visibility`; Pydantic schemas and HTTP error mapping |
| `src/ai_search/service.py` | Embedding model cache, question vector creation, async psycopg connections and queries, search orchestration, timing decorator, synthetic visibility summary |
| `src/ai_search/retrieval.py` | Lexical tokenization, BM25 scoring and RRF rank combination; designed for unit testing |
| `tests/test_retrieval.py` | Retrieval algorithm correctness checks |
| `tests/test_api.py` | Endpoint and validation tests |
| `Dockerfile`, `compose.api.yaml` | Repeatable API container connected to an already-running Docker PostgreSQL server |

## API contract

| Route | Method | Returns |
|---|---|---|
| `/health` | GET | Process liveness only; not database readiness |
| `/health/db` | GET | Database connectivity check; 503 if unavailable |
| `/analytics/visibility` | GET | Aggregate statistics from the *synthetic* provider dataset |
| `/search` | POST | Ranked source chunks with IDs, URLs, positions and method-dependent scores |

Example request:

```json
{"question":"What connectivity does UNO-2484G support?","brand":"Advantech","method":"hybrid","top_k":5}
```

`method` is validated as `dense`, `bm25` or `hybrid`; `top_k` must be between 1 and 50. `/docs` displays the generated OpenAPI contract.

## Async and decorators: why they help

- `psycopg.AsyncConnection` allows other requests to progress while waiting on the database.
- `asyncio.to_thread(...)` moves synchronous Sentence Transformer inference and Python BM25 scoring off the event loop; it **does not** make model computation itself inherently faster.
- `@timed` wraps the search coroutine and logs duration in a `finally` block, centralizing timing without duplicating code.
- `@app.post` / `@app.get` are FastAPI routing decorators registering endpoints.
- Pydantic validates request bounds and shapes before the service is called.
- Local model caching avoids loading the model on each query, but each Uvicorn worker would have its own cache.

## Run and test

```bash
# At repository root, with compatible dedicated Python environment activated
python -m pip install -r requirements-dev.txt
docker start ai-search-db
export PYTHONPATH=src
export DATABASE_URL='postgresql://postgres:localdev@localhost:5433/ai_search'
python -m uvicorn ai_search.api:app --reload
```

Try `/health`, `/health/db`, then `/search` at `http://127.0.0.1:8000/docs`. The latter requires an initialized and populated `website_chunks` table as well as a working local embedding model.

Run unit tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Or build the API container on Docker Desktop for Mac:

```bash
docker compose -f compose.api.yaml up --build -d
```

It connects to the existing database via `host.docker.internal:5433` and does **not** provision a new database. Do not remove the existing PostgreSQL volume when resetting the API.

## Engineering limitations

BM25 currently loads candidate brand documents and computes scores per request in Python; this does not scale like an indexed sparse search engine. CPU/model-memory bottlenecks remain with multiple concurrent requests. This prototype also has no authentication, request limits, background ingestion job, observability backend or deployment hardening. API and retrieval unit tests do not replace integration and performance testing on real hardware.
