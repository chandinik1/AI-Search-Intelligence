from fastapi.testclient import TestClient
from ai_search.api import app

client = TestClient(app)


def test_health():
    assert client.get('/health').json() == {'status': 'ok'}


def test_validation():
    assert client.post('/search', json={'question': 'x', 'method': 'hybrid'}).status_code == 422
    assert client.post('/search', json={'question': 'valid question', 'method': 'unknown'}).status_code == 422


def test_synthetic_analytics():
    resp = client.get('/analytics/visibility')
    assert resp.status_code == 200
    assert resp.json()['dataset_type'] == 'synthetic'
    assert resp.json()['responses'] == 18
