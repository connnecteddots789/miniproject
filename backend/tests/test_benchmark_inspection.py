from app import main

def test_confusion_cell_inspection_returns_real_stored_samples(monkeypatch):
    artifact = {
        'confusion_records': {
            'dynamic': {
                'Normal|||DoS': [
                    {'test_record': 12, 'actual': 'Normal', 'predicted': 'DoS',
                     'record': {'duration': 0.4, 'src_bytes': 120}}
                ]
            }
        }
    }
    monkeypatch.setattr(main, 'load_model', lambda dataset: artifact)
    result = main.benchmark_cell_records('NSL-KDD', 'dynamic', 'Normal', 'DoS', 25)
    assert result['count'] == 1
    assert result['records'][0]['test_record'] == 12
    assert result['records'][0]['record']['src_bytes'] == 120
