"""DB-backed retrieval. Async helps with I/O; model inference is offloaded."""
import asyncio
from functools import lru_cache
from time import perf_counter
import logging
import os

import psycopg
from sentence_transformers import SentenceTransformer

from .retrieval import bm25_search, rrf_fuse

log = logging.getLogger(__name__)
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
DSN = os.getenv('DATABASE_URL', 'postgresql://postgres:localdev@localhost:5433/ai_search')


def timed(func):
    """Reusable async decorator. `wraps` keeps function metadata intact."""
    from functools import wraps
    @wraps(func)
    async def wrapper(*args, **kwargs):
        started = perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            log.info('%s finished in %.3fs', func.__name__, perf_counter()-started)
    return wrapper


@lru_cache(maxsize=1)
def get_model():
    """Load the local embedding model once per API worker (first call downloads it)."""
    return SentenceTransformer(MODEL_NAME)


def embed_sync(question: str) -> str:
    vector = get_model().encode(question, normalize_embeddings=True)
    return '[' + ','.join(str(float(x)) for x in vector) + ']'


async def load_corpus(conn, brand: str) -> list[dict]:
    async with conn.cursor() as cur:
        await cur.execute("""SELECT id, url, chunk_index, content
                             FROM website_chunks WHERE brand=%s ORDER BY id""", (brand,))
        rows = await cur.fetchall()
    return [dict(id=r[0], url=r[1], chunk_index=r[2], content=r[3]) for r in rows]


async def dense_search(conn, vector: str, brand: str, limit: int) -> list[dict]:
    async with conn.cursor() as cur:
        await cur.execute("""SELECT id, url, chunk_index, content,
                                   1 - (embedding <=> %s::vector) AS score
                             FROM website_chunks
                             WHERE brand=%s AND embedding IS NOT NULL
                             ORDER BY embedding <=> %s::vector LIMIT %s""",
                          (vector, brand, vector, limit))
        rows = await cur.fetchall()
    return [dict(id=r[0], url=r[1], chunk_index=r[2], content=r[3],
                 score=float(r[4])) for r in rows]


@timed
async def search(question: str, brand: str, method: str, top_k: int,
                 candidates: int = 20) -> list[dict]:
    """Use DB cosine retrieval, BM25, or RRF; same document IDs in both lists."""
    candidate_count = max(top_k, candidates)
    # Embedding generation is sync/compute-heavy: avoid blocking the event loop.
    vector = await asyncio.to_thread(embed_sync, question) if method != 'bm25' else None
    async with await psycopg.AsyncConnection.connect(DSN) as conn:
        dense = []
        sparse = []
        if method in ('dense', 'hybrid'):
            dense = await dense_search(conn, vector, brand, candidate_count)
        if method in ('bm25', 'hybrid'):
            corpus = await load_corpus(conn, brand)
            sparse = await asyncio.to_thread(bm25_search, question, corpus, candidate_count)
    if method == 'dense':
        return dense[:top_k]
    if method == 'bm25':
        return sparse[:top_k]
    return rrf_fuse(dense, sparse, limit=top_k)


async def visibility_summary(path: str):
    """Synthetic dataset: never claim these numbers describe real providers."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    records = data['records']
    return {'dataset_type': 'synthetic', 'target_brand': data['target_brand'],
            'responses': len(records),
            'target_mentioned': sum(bool(r['expected']['target_mentioned']) for r in records),
            'target_recommended': sum(bool(r['expected']['target_recommended']) for r in records)}
