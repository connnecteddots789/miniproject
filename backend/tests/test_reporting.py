from app.reporting import build_pdf_report


def test_pdf_report_is_valid_pdf():
    model = {
        'dataset': 'NSL-KDD', 'kind': 'Decision Tree', 'seed': 42,
        'created_at': '2026-11-18T12:00:00', 'split': 'official train/test files',
        'train_records': 100, 'test_records': 20, 'records': 120,
        'results': {
            key: {
                'accuracy': .8, 'macro_precision': .7, 'macro_recall': .6,
                'macro_f1': .65, 'weighted_f1': .78, 'training_seconds': 1.2,
                'inference_seconds': .03, 'feature_count': 12,
                'labels': ['Normal', 'DoS'], 'per_class': {
                    'Normal': {'precision': .8, 'recall': .9, 'f1-score': .84, 'support': 10},
                    'DoS': {'precision': .6, 'recall': .5, 'f1-score': .55, 'support': 10},
                }, 'confusion_matrix': [[9, 1], [5, 5]],
            } for key in ('all', 'fixed', 'dynamic')
        },
        'global_importance': {'duration': .5, 'src_bytes': .3},
        'family_features': {'DoS': ['src_bytes', 'count']},
    }
    data = build_pdf_report(model)
    assert data.startswith(b'%PDF-')
    assert len(data) > 1000
