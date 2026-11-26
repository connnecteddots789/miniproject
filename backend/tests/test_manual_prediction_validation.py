from fastapi.testclient import TestClient
from app.main import app


def test_manual_prediction_schema_exposes_typed_required_fields():
    response = TestClient(app).get('/api/schema/NSL-KDD')
    assert response.status_code == 200
    payload = response.json()
    assert payload['columns']
    assert set(payload['numeric_columns']).isdisjoint(payload['categorical_columns'])
    assert set(payload['numeric_columns']) | set(payload['categorical_columns']) == set(payload['columns'])
    assert payload['categorical_options']


def test_manual_prediction_rejects_empty_record():
    response = TestClient(app).post('/api/predict', json={
        'dataset': 'NSL-KDD', 'record': {}, 'approach': 'dynamic'
    })
    assert response.status_code == 422
    assert 'Missing:' in response.json()['detail']


def test_manual_prediction_rejects_non_finite_numeric_value():
    schema = TestClient(app).get('/api/schema/NSL-KDD').json()
    record = {name: 0 for name in schema['numeric_columns']}
    for name, options in schema['categorical_options'].items():
        record[name] = options[0]
    record[schema['numeric_columns'][0]] = 'NaN'
    response = TestClient(app).post('/api/predict', json={
        'dataset': 'NSL-KDD', 'record': record, 'approach': 'dynamic'
    })
    assert response.status_code == 422
    assert 'Invalid dataset values' in response.json()['detail']


def test_manual_prediction_rejects_unknown_category():
    schema = TestClient(app).get('/api/schema/NSL-KDD').json()
    record = {name: 0 for name in schema['numeric_columns']}
    for name, options in schema['categorical_options'].items():
        record[name] = options[0]
    record[schema['categorical_columns'][0]] = '__invalid_category__'
    response = TestClient(app).post('/api/predict', json={
        'dataset': 'NSL-KDD', 'record': record, 'approach': 'dynamic'
    })
    assert response.status_code == 422
    assert 'Invalid dataset values' in response.json()['detail']
