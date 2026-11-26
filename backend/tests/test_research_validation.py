from fastapi.testclient import TestClient
from app.main import app


def test_binary_research_validation_is_separate_and_measured():
    response = TestClient(app).get('/api/research/binary/unsw')
    assert response.status_code == 200
    payload = response.json()
    assert payload['dataset'] == 'UNSW-NB15'
    assert payload['task'] == 'binary intrusion detection'
    assert payload['test_records'] == 82332
    assert 0.89 < payload['test_metrics']['accuracy'] < 0.92
    assert payload['test_metrics']['labels'] == ['Normal', 'Attack']


def test_manual_prediction_examples_are_real_input_only_records():
    response = TestClient(app).get('/api/schema/NSL-KDD/examples')
    assert response.status_code == 200
    payload = response.json()
    assert payload['dataset'] == 'NSL-KDD'
    assert payload['examples']
    for example in payload['examples']:
        assert example['source'] == 'Official training record'
        assert 'label' not in example['values']
        assert 'difficulty' not in example['values']


def test_manual_compare_runs_all_three_strategies_on_one_valid_record():
    client = TestClient(app)
    examples = client.get('/api/schema/NSL-KDD/examples').json()['examples']
    record = examples[0]['values']
    response = client.post('/api/predict/compare', json={
        'dataset': 'NSL-KDD', 'record': record, 'approach': 'dynamic'
    })
    assert response.status_code == 200
    payload = response.json()
    assert set(payload['results']) == {'all', 'fixed', 'dynamic'}
    assert isinstance(payload['agreement'], bool)
    assert 'not added to the detection history' in payload['note']
