from fastapi.testclient import TestClient
from app.main import app


def test_dataset_explorer_returns_paginated_original_rows():
    response = TestClient(app).get('/api/inspect/NSL-KDD/records?page=1&page_size=10')
    assert response.status_code == 200
    payload = response.json()
    assert payload['dataset'] == 'NSL-KDD'
    assert payload['page'] == 1
    assert len(payload['rows']) == 10
    assert payload['total_records'] > 10
    assert payload['rows'][0]['record_number'] == 1
    assert payload['rows'][0]['ground_truth']
    assert payload['rows'][0]['original_label']
    assert payload['columns']


def test_dataset_explorer_search_and_pagination_are_stable():
    client = TestClient(app)
    first = client.get('/api/inspect/NSL-KDD/records?page=1&page_size=5&q=normal')
    second = client.get('/api/inspect/NSL-KDD/records?page=2&page_size=5&q=normal')
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    assert a['filtered_records'] > 0
    assert all('normal' in row['original_label'].lower() for row in a['rows'])
    assert {row['record_number'] for row in a['rows']}.isdisjoint({row['record_number'] for row in b['rows']})


def test_dataset_explorer_rejects_unknown_dataset():
    response = TestClient(app).get('/api/inspect/unknown/records')
    assert response.status_code == 400
