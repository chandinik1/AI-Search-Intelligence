# AI Search Intelligence Platform

**A local, reproducible project for investigating synthetic AI-search brand visibility and evaluating evidence retrieval, with a FastAPI backend.**

The project has two complementary parts:

- **Part 1 — Retrieval Research & Evaluation:** A documented Jupyter notebook for ingesting public industrial-IoT product pages, cleaning and chunking text, computing 384-dimensional sentence embeddings, storing vectors in PostgreSQL + pgvector, retrieving with dense and BM25 search, fusing rankings with Reciprocal Rank Fusion (RRF), and evaluating results with human relevance judgments.
- **Part 2 — API & Backend Engineering:** Reusable Python retrieval modules, a FastAPI HTTP service with Pydantic validation, asynchronous database operations, timing instrumentation, tests, and Docker packaging.

The target brand is **Advantech**. Provider-named ChatGPT/Claude/Gemini scenarios in the dataset are **synthetic examples**, not real provider outputs or measurements of search visibility. This project **retrieves candidate evidence**; it does not prove that publishing content will change an AI provider's answers. No commercial LLM API is required.

## Architecture

```mermaid
flowchart TD
    A["Synthetic provider response dataset"] --> B["Brand mentions, recommendations & gaps"]
    B --> C["Investigation questions"]
    D["Permitted public pages or curated offline fixtures"] --> E["Robots-aware fetch & HTML cleaning"]
    E --> F["Chunk text and retain URL provenance"]
    F --> G["Local MiniLM 384D embeddings"]
    G --> H[("PostgreSQL + pgvector")]
    F --> H
    C --> I["Dense retrieval: cosine distance"]
    C --> J["Sparse retrieval: BM25"]
    H --> I
    H --> J
    I --> K["RRF hybrid ranking"]
    J --> K
    K --> L["Evidence candidates"]
    L --> M["Human-reviewed relevance labels"]
    M --> N["Recall@5 and MRR@5"]
    H --> P["FastAPI /search"]
    P --> I
    P --> J
    A --> Q["FastAPI /analytics/visibility"]
```

The notebook is the place to explore, visualize, and evaluate the approach; the API is the reusable serving interface. The API **does not automatically run ingestion**—ingest documents with the notebook first.

## Repository structure

```text
ai-search-intelligence/
├── notebooks/
│   └── ai_search_intelligence.ipynb
├── src/
│   └── ai_search/
│       ├── __init__.py
│       ├── api.py                # HTTP routes and request/response schemas
│       ├── service.py            # Async DB, embedding, analytics, latency logging
│       └── retrieval.py          # BM25, tokenization and RRF
├── data/
│   ├── ai_search_mock_dataset.json
│   ├── advantech_retrieval_eval_questions.json
│   └── advantech_offline_reference_passages.json
├── tests/
│   ├── test_api.py
│   ├── test_retrieval.py
│   └── test_extraction.py
├── docs/
│   ├── images/               # PNGs exported when notebook is executed
│   ├── PART_1_RETRIEVAL_EVALUATION.md
│   └── PART_2_API_BACKEND.md
├── Dockerfile
├── compose.yaml              # Fresh local PostgreSQL + pgvector
├── compose.api.yaml          # API connecting to an existing local DB
├── requirements.txt          # Notebook dependencies
├── requirements-api.txt
├── requirements-dev.txt      # Backend + test dependencies
├── .env.example
├── .gitignore
└── README.md
```

## Part 1 — Retrieval Research & Evaluation

**Purpose:** Explore whether dense semantic search, lexical BM25 or RRF-based hybrid retrieval produces stronger *evidence retrieval* for the same set of technical questions.

**Technical coverage:** website ingestion, HTML boilerplate filtering, overlapping text chunks, Sentence Transformers (`all-MiniLM-L6-v2`, 384D), cosine similarity, PostgreSQL + pgvector, BM25, RRF, PCA and similarity heatmaps, corpus diagnostics, Recall@K and MRR.

**Research flow:**

```mermaid
flowchart LR
    A["Sample AI answers"] --> B["Identify visibility gaps"]
    B --> C["Questions"]
    D["Web pages"] --> E["Clean & chunk"]
    E --> F["Embed & persist"]
    C --> G["Dense + BM25"]
    F --> G
    G --> H["RRF"]
    H --> I["Review labels"]
    I --> J["Evaluate"]
```

### Run Part 1

Requirements: Docker Desktop/Engine running, Python (prefer 3.11), and a local environment. Run all commands at the **repository root**.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a **fresh database only**:

```bash
docker compose up -d
docker compose ps
```

If you already have the `ai-search-db` container and stored vectors from earlier work, **reuse it** instead:

```bash
docker start ai-search-db
```

Do not run the fresh database Compose file against a conflicting existing `ai-search-db` container. Docker exposes the development database on host **127.0.0.1:5433**; the database listens on 5432 *inside* Docker.

Open `notebooks/ai_search_intelligence.ipynb` in Jupyter/VS Code using the same interpreter. Use the **figure-export-enabled notebook** and run its initial repository-root setup cell; it supports starting from either `notebooks/` or the repository root, resolving relative `data/` and `docs/images/` paths. Execute the analysis, ingestion, retrieval and evaluation sections in order. To avoid network-dependent retrieval, enable `USE_OFFLINE_FIXTURES=True` in the notebook; these passages are explicitly curated demonstrations and **not live webpage text**.

### Evaluation protocol

`data/advantech_retrieval_eval_questions.json` contains **16 authored retrieval questions**, source hints and evidence-term hints—not verified chunk-level relevance labels. The notebook produces `data/advantech_chunk_judgments_review.csv` after data ingestion. Check every suggested question–chunk judgment against the **full text** before setting `relevant` to `1` (supports the question) or `0` (does not). Keep the review CSV local (it is Git-ignored). Set `USE_SILVER_DEMO=False` to calculate scores with reviewed judgments.

- **Recall@5:** Share of all judged relevant chunks appearing among five returned results.
- **MRR@5:** Mean inverse rank of the first relevant result within the top five.
- **Unanswerable queries:** Kept separate; Recall/MRR do **not** measure safe abstention.
- **Fair comparison:** Use one frozen corpus, consistent candidate budgets, and the same judgments for dense/BM25/RRF. Silver heuristic labels can systematically favor lexical matching and are not ground truth.

Read [Part 1 technical documentation](docs/PART_1_RETRIEVAL_EVALUATION.md) for the retrieval mechanics, evaluation workflow and limitations.

## Results and visualizations

The notebook contains an exploratory retrieval study, not a validated benchmark. The figures below are **generated locally by executing the notebook**; they are not pre-populated screenshots. If a figure appears broken in the GitHub README, run its corresponding cell and commit the exported PNG in `docs/images/`.

### 1. Chunking with overlap

![Overlapping text chunks](docs/images/chunk_overlap.png)

Documents are split into overlapping windows to preserve context around chunk boundaries. Overlap also increases storage and may produce near-duplicate results.

### 2. Dense retrieval similarity

![Dense similarity ranking](docs/images/dense_similarity_ranking.png)

A query embedding is compared with chunk embeddings using cosine similarity. The graph shows ranking scores from the current selected question and page.

### 3. Embedding exploration: PCA

![PCA projection of question and chunk embeddings](docs/images/embedding_pca.png)

PCA projects the 384-dimensional embeddings onto two axes for visualization. Distances in the projected plot **do not exactly preserve** the original cosine similarities.

### 4. Cosine similarity heatmap

![Question–chunk similarity heatmap](docs/images/similarity_heatmap.png)

Shows the original embedding cosine similarities across investigation questions and text chunks. Similarity is not itself proof that a chunk answers a question.

### 5. Dense vs. sparse vs. hybrid ranking

![Dense, BM25, and RRF rank comparison](docs/images/retrieval_comparison.png)

This plot compares rankings rather than raw scores because cosine similarity, BM25 and RRF use different scales.

### 6. Ingestion quality

![Corpus quality diagnostics](docs/images/corpus_quality.png)

Displays corpus-size and text-length diagnostics after ingestion. HTML boilerplate filtering is heuristic; inspect individual chunks when evaluating quality.

### 7. Retrieval evaluation

![Recall@5 and MRR@5](docs/images/retrieval_evaluation.png)

The notebook measures Recall@5 and MRR@5 on its selected evaluation questions. **Do not interpret scores produced with `USE_SILVER_DEMO=True` as human-reviewed evidence.** These provisional, heuristic judgments can bias the comparison. Set `USE_SILVER_DEMO=False` only after reviewing question–chunk relevance labels.

#### Exporting images for GitHub

1. Use the corrected notebook version that calls `fig.savefig(FIGURE_DIR / "name.png", ...)` before each `plt.show()`.
2. Run its repository-root setup cell first; confirm that `FIGURE_DIR` points to `<repository>/docs/images`.
3. Execute plotting cells in order, including any prerequisite ingestion, model and evaluation cells.
4. Confirm `docs/images/*.png` exists **on your Mac**. Committing only the notebook does not automatically commit PNGs.
5. From the repository root, run:

```bash
ls -lh docs/images/
git add README.md notebooks/ai_search_intelligence.ipynb docs/images/
git commit -m "Document API and publish retrieval visualizations"
git push origin main
```

If you have **not yet generated** the PNGs, GitHub will show broken images until they are committed. It's better to omit image links temporarily than to substitute illustrative screenshots for real results.

## Part 2 — API & Backend Engineering

**Purpose:** Make the algorithms reusable outside Jupyter through an HTTP interface, with validated inputs and concurrency-aware I/O.

**Technical coverage:** FastAPI and Pydantic, `async`/`await` with psycopg, `asyncio.to_thread` for synchronous embedding and BM25 computation, an async timing decorator, independent retrieval modules, pytest, Dockerfile and Docker Compose.

```mermaid
flowchart TD
    A["Client, curl or /docs"] --> B["FastAPI /search"]
    B --> C["Pydantic validates request"]
    C --> D["Async search service"]
    D --> E["Embedding inference in worker thread"]
    D --> F["Async PostgreSQL I/O"]
    F --> G["pgvector dense ranking"]
    F --> H["BM25 on loaded corpus"]
    E --> G
    G --> I["RRF if hybrid"]
    H --> I
    I --> J["JSON results with sources & scores"]
```

### Run Part 2 locally

First complete Part 1 ingestion and confirm the table exists. Install the backend/test dependencies **in your project's virtual environment, not Conda base**:

```bash
python -m pip install -r requirements-dev.txt
export DATABASE_URL='postgresql://postgres:localdev@localhost:5433/ai_search'
export PYTHONPATH=src
python -m uvicorn ai_search.api:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive API documentation. The server is a separate process; it uses the PostgreSQL data already stored by the notebook.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/db
curl http://127.0.0.1:8000/analytics/visibility
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"What connectivity does UNO-2484G support?","brand":"Advantech","method":"hybrid","top_k":5}'
```

`/search` supports `dense`, `bm25`, and `hybrid` methods. Scores are **not comparable across methods**: dense returns cosine similarity, BM25 returns its lexical score, and hybrid returns an RRF score.

### Request validation and response contract

FastAPI auto-generates Swagger/OpenAPI docs at [`/docs`](http://127.0.0.1:8000/docs) while the local service is running. The `/search` request accepts:

| Field | Type | Meaning |
|---|---|---|
| `question` | string, 3–1000 characters | What evidence to retrieve |
| `brand` | string | Brand-filter for the stored corpus (defaults to `Advantech`) |
| `method` | `dense`, `bm25`, `hybrid` | Select a retrieval strategy |
| `top_k` | integer, 1–50 | Maximum number of returned chunks |

The response follows this schema (the values below are **structural placeholders**, not actual search results):

```json
{
  "method": "hybrid",
  "results": [
    {
      "id": 123,
      "url": "https://example.org/product-page",
      "chunk_index": 2,
      "content": "Example retrieved text chunk",
      "score": 0.032
    }
  ]
}
```

Replace the placeholder example with the output from your own running service only after checking that the chunks and URLs are correct. The `/health` endpoint reports process liveness; `/health/db` independently checks PostgreSQL and can return HTTP 503 when unavailable. `/analytics/visibility` uses the **synthetic** mock provider dataset and should not be represented as live provider monitoring.

For a readable actual response, run in a second terminal while Uvicorn is running:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which industrial edge gateways support remote management?","brand":"Advantech","method":"hybrid","top_k":3}' \
  | python -m json.tool
```

### Execution path

```mermaid
sequenceDiagram
    participant User as HTTP Client
    participant API as FastAPI + Pydantic
    participant Svc as Search Service
    participant Model as MiniLM Model
    participant DB as PostgreSQL + pgvector
    User->>API: POST /search
    API->>API: Validate payload
    API->>Svc: search(question, brand, method, top_k)
    opt Dense or Hybrid
        Svc->>Model: Create embedding (worker thread)
        Model-->>Svc: 384D vector
        Svc->>DB: Async cosine-distance query
        DB-->>Svc: Dense results
    end
    opt BM25 or Hybrid
        Svc->>DB: Async fetch brand corpus
        DB-->>Svc: Candidate text chunks
        Svc->>Svc: BM25 scoring (worker thread)
    end
    opt Hybrid
        Svc->>Svc: Fuse ranks using RRF
    end
    Svc-->>API: Ranked evidence chunks
    API-->>User: JSON response
```

### Why these backend choices?

- **FastAPI and Pydantic:** callable, documented HTTP endpoints with typed request validation; the notebook remains the research environment.
- **Asynchronous psycopg:** database waits need not block the event loop. This does not inherently accelerate SQL queries.
- **`asyncio.to_thread`:** offloads blocking model inference and BM25 scoring; does not turn these operations into parallel GPU inference.
- **Decorators:** `@app.get` / `@app.post` register routes; a custom `@timed` wrapper centralizes request-duration logging.
- **Docker:** reproducible API packaging while persisting PostgreSQL data separately.
- **pytest:** isolated algorithm tests and API validation tests; live database queries still need a separate integration check.

To containerize **only the API** while reusing an existing DB on Docker Desktop for Mac:

```bash
docker start ai-search-db
docker compose -f compose.api.yaml up --build -d
```

This API container connects through `host.docker.internal:5433` and listens at `127.0.0.1:8000`. `docker compose -f compose.api.yaml down` stops the API without deleting the existing DB container or volume.

### Run tests

```bash
PYTHONPATH=src python -m pytest -q
```

Unit tests are distinct from a **live integration test**: validate `/health/db` and `/search` against your locally populated database to check model imports, vector dimensions and SQL behavior end-to-end.

Read [Part 2 technical documentation](docs/PART_2_API_BACKEND.md) for routes, code ownership, concurrency, decorators, Docker and testing.

## Technical documentation

- **[Part 1 — Retrieval Research & Evaluation](docs/PART_1_RETRIEVAL_EVALUATION.md):** ingestion strategy, embedding mathematics, BM25, RRF, evaluation setup and validity threats.
- **[Part 2 — API & Backend Engineering](docs/PART_2_API_BACKEND.md):** module responsibilities, API schemas, request sequence, concurrency, decorators, Docker and limitations.
- **[Notebook](notebooks/ai_search_intelligence.ipynb):** implementation, plots and experimental results (after local execution).

## What each part demonstrates

| | Part 1 — Retrieval Research & Evaluation | Part 2 — API & Backend Engineering |
|---|---|---|
| Primary deliverable | Notebook and reproducible retrieval experiments | Callable HTTP search service |
| Core competencies | NLP, chunking, embedding, IR, relevance judgments, metrics | Python modules, API design, Pydantic, async I/O, Docker, testing |
| Typical workflow | Run notebook cells, inspect charts, review candidate evidence | Send `POST /search`, receive structured JSON |
| Storage | PostgreSQL + pgvector | Reuses the same PostgreSQL + pgvector |
| Main limitation | Small source corpus and provisional labels until human review | Prototype-level concurrency, no authentication or production monitoring |

## Key trade-offs & limitations

- The source dataset is synthetic. Provider labels do not describe real ChatGPT/Claude/Gemini outputs.
- Dense retrieval captures learned semantic relationships; BM25 is better suited to exact identifiers. RRF fuses *ranks*, not raw scores. **RRF is not guaranteed to outperform either method**.
- Current BM25 tokenization and corpus-loading strategy are intended for a small corpus; larger deployments would need optimized tokenization, caching and/or an indexed sparse retrieval engine.
- pgvector uses exact cosine ordering here: no HNSW/IVFFlat index is configured. Approximate-nearest-neighbor indexing should be measured only when scale justifies it.
- `async` improves overlap of waiting tasks, **not the speed of CPU-heavy inference**; Python's model inference is offloaded to a worker thread.
- Re-ingestion can change database-assigned row IDs; recheck any human judgments associated with old chunk IDs.
- Boilerplate removal is heuristic. HTML changes or JavaScript-rendered content can affect extraction. Respect robots rules, attribution and website usage conditions.
- API is local-development quality. No authentication, request throttling, monitoring or operational service-level commitments are provided.
- Live search tests were not executed against your Mac from this archive; run them locally before publishing performance claims.

## Reproducibility, security & attribution

The included offline reference passages are **paraphrased fixtures** with source links, not copied website content. Never publish local `.env`, virtual environments, database volumes, reviewed labels with unintended private data, or model caches. The default `localdev` password is **for local development only**; change it for any shared deployment.

This project is independent and is not affiliated with Advantech or any AI provider.

## License

No license is granted unless you add one. Choose a license explicitly before inviting reuse; third-party documentation, data and dependencies retain their own terms.
