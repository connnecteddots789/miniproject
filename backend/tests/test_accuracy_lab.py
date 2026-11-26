from fastapi.testclient import TestClient
from app.main import app, DB
from app import accuracy_lab as lab


def test_accuracy_lab_uses_separate_storage():
    assert lab.DB != DB
    assert lab.DB.parent.name == 'accuracy_lab'


def test_accuracy_lab_list_endpoint_is_available():
    response = TestClient(app).get('/api/accuracy-lab/runs')
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_accuracy_lab_rejects_unsupported_dataset():
    response = TestClient(app).post('/api/accuracy-lab/runs', json={'dataset': 'UNKNOWN'})
    assert response.status_code == 400


def test_accuracy_lab_report_exports_are_valid():
    run_id = 'test-report-format'
    labels = ['Normal', 'Attack']
    metrics = {
        'accuracy': .91, 'macro_precision': .90, 'macro_recall': .89,
        'macro_f1': .895, 'weighted_f1': .91,
        'confusion_matrix': [[90, 5], [4, 1]], 'labels': labels,
        'per_class': {'Normal': {'precision': .95}, 'Attack': {'precision': .17}},
    }
    summary = {
        'dataset': 'NSL-KDD', 'protocol': 'random_holdout', 'seed': 42,
        'model': 'extra_trees', 'train_records': 100, 'validation_records': 20,
        'test_records': 100, 'original_feature_count': 41, 'encoded_feature_count': 60,
        'selected_feature_count': 30, 'selected_features': ['duration', 'src_bytes'],
        'hybrid_top_features': [{'feature': 'duration', 'score': .9}],
        'dynamic_policy': 'Validation-tuned routing policy.',
        'protocol_note': 'Stratified random holdout.',
        'exact_duplicate_overlap_test_rows': 0, 'exact_duplicate_overlap_test_fraction': 0,
        'results': {k: metrics for k in ('all_features', 'fixed_features', 'dynamic_features')},
    }
    lab.RUNS[run_id] = {'id': run_id, 'dataset': 'NSL-KDD', 'protocol': 'random_holdout',
                        'status': 'completed', 'created_at': '2026-11-19T00:00:00',
                        'summary': summary}
    pdf = lab.report_pdf(run_id)
    csv = lab.report_csv(run_id)
    data = lab.report_json(run_id)
    assert pdf.body.startswith(b'%PDF-') and len(pdf.body) > 1500
    assert csv.body.startswith(b'approach,accuracy')
    assert data['results']['dynamic_features']['accuracy'] == .91
    lab.RUNS.pop(run_id, None)


def test_adaptive_efficiency_is_measured_and_reported(monkeypatch):
    import numpy as np
    import pandas as pd
    from app.accuracy_lab import RunRequest
    from uuid import uuid4

    rng = np.random.default_rng(17)
    n = 500
    x = pd.DataFrame({f'f{i}': rng.normal(size=n) for i in range(8)})
    x['signal'] = rng.normal(size=n)
    y = pd.Series(np.where(x['signal'] + .35*x['f0'] > 0, 'Attack', 'Normal'))
    monkeypatch.setattr(lab, 'load_protocol_data', lambda dataset, protocol: (
        x, y, {'source_files':['synthetic.csv'], 'protocol_note':'Synthetic test fixture.'}
    ))
    run_id = str(uuid4())
    lab.RUNS[run_id] = {'id':run_id,'dataset':'NSL-KDD','protocol':'random_holdout',
                        'status':'queued','created_at':'2026-11-19T00:00:00','summary':{}}
    lab.worker(run_id, RunRequest(dataset='NSL-KDD', protocol='random_holdout', seed=17,
                                  model='extra_trees', feature_fraction=.5, test_size=.2))
    result = lab.RUNS[run_id]
    assert result['status'] == 'completed', result.get('error')
    efficiency = result['summary']['efficiency']
    assert efficiency['all_features_per_record'] > 0
    assert efficiency['fixed_features_per_record'] < efficiency['all_features_per_record']
    assert efficiency['fixed_features_per_record'] <= efficiency['dynamic_average_features_per_record'] <= efficiency['all_features_per_record']
    assert efficiency['test_records_timed'] == result['summary']['test_records']
    assert efficiency['dynamic_inference_wall_seconds'] >= 0
    pdf = lab.report_pdf(run_id)
    csv = lab.report_csv(run_id)
    assert pdf.body.startswith(b'%PDF-')
    assert b'avg_features_per_record' in csv.body
    lab.RUNS.pop(run_id, None)
