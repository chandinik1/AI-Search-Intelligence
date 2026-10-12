"""Pure ranking algorithms shared by notebook experiments and the HTTP API."""
from collections import Counter
import math
import re


def tokenize(text: str) -> list[str]:
    """Retain technical identifiers (e.g. UNO-2484G) as well as component tokens."""
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.casefold())
    return tokens + [part for term in tokens if '-' in term or '_' in term for part in re.split('[-_]', term)]


def bm25_search(query: str, corpus: list[dict], limit: int = 10,
                k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """Score the supplied in-memory corpus; no sparse index is persisted yet."""
    if not corpus:
        return []
    docs = [tokenize(row['content']) for row in corpus]
    lengths = list(map(len, docs))
    avgdl = sum(lengths) / len(docs) or 1.0
    df = Counter()
    for tokens in docs:
        df.update(set(tokens))
    terms = set(tokenize(query))
    n = len(corpus)
    output = []
    for row, tokens, dl in zip(corpus, docs, lengths):
        frequencies = Counter(tokens)
        score = 0.0
        for term in terms:
            tf = frequencies[term]
            if not tf:
                continue
            idf = math.log1p((n - df[term] + .5) / (df[term] + .5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        output.append({**row, 'score': float(score)})
    # Predictable tie-breaking; exclude no-match docs.
    return sorted((r for r in output if r['score'] > 0), key=lambda x: (-x['score'], x['id']))[:limit]


def rrf_fuse(dense: list[dict], sparse: list[dict], k: int = 60,
             limit: int = 5) -> list[dict]:
    """Fuse rank positions, not incomparable raw cosine and BM25 scores."""
    scores, documents = {}, {}
    for result_set in (dense, sparse):
        for rank, row in enumerate(result_set, start=1):
            doc_id = row['id']
            documents[doc_id] = row
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    ids = sorted(scores, key=lambda i: (-scores[i], i))[:limit]
    return [{**documents[i], 'score': scores[i]} for i in ids]
