"""FastAPI interface. Inspect interactive OpenAPI docs at /docs."""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg
from .service import DSN, search, visibility_summary

app = FastAPI(title='AI Search Intelligence', version='0.2.0',
              description='Local evidence retrieval; provider data is synthetic.')


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    brand: str = Field(default='Advantech', min_length=1)
    method: str = Field(default='hybrid', pattern='^(dense|bm25|hybrid)$')
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    id: int
    url: str
    chunk_index: int
    content: str
    score: float


class SearchResponse(BaseModel):
    method: str
    results: list[SearchHit]


@app.get('/health')
def health():
    """Process liveness (does not assert database readiness)."""
    return {'status': 'ok'}


@app.get('/health/db')
async def health_db():
    try:
        async with await psycopg.AsyncConnection.connect(DSN, connect_timeout=3) as conn:
            async with conn.cursor() as cur:
                await cur.execute('SELECT 1')
                await cur.fetchone()
        return {'database': 'ok'}
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail='Database unavailable') from exc


@app.post('/search', response_model=SearchResponse)
async def search_endpoint(payload: SearchRequest):
    try:
        hits = await search(payload.question, payload.brand, payload.method, payload.top_k)
        return SearchResponse(method=payload.method, results=[SearchHit(**hit) for hit in hits])
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail='Database unavailable or table not initialized') from exc


@app.get('/analytics/visibility')
async def visibility():
    return await visibility_summary(os.getenv('MOCK_DATA_PATH', 'data/ai_search_mock_dataset.json'))
