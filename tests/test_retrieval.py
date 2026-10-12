from ai_search.retrieval import tokenize, bm25_search, rrf_fuse


def test_tokenization_keeps_device_id():
    assert 'uno-2484g' in tokenize('UNO-2484G industrial PC')


def test_bm25_returns_exact_match():
    data = [{'id': 1, 'content': 'UNO-2484G industrial computer'},
            {'id': 2, 'content': 'Office printer'}]
    assert bm25_search('UNO-2484G', data)[0]['id'] == 1


def test_rrf_merges_same_id():
    dense = [{'id': 1, 'content': 'one'}, {'id': 2, 'content': 'two'}]
    sparse = [{'id': 2, 'content': 'two'}, {'id': 1, 'content': 'one'}]
    results = rrf_fuse(dense, sparse)
    assert len(results) == 2
    assert results[0]['score'] > 0


def test_rrf_empty():
    assert rrf_fuse([], []) == []
