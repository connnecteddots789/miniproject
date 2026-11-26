from fastapi.testclient import TestClient
from app.main import app
from app.ml import load_model


def test_bundled_production_artifacts_load_with_pinned_sklearn():
    for dataset, expected_labels in (("NSL-KDD", 5), ("UNSW-NB15", 10)):
        artifact = load_model(dataset)
        assert artifact is not None
        assert len(artifact["labels"]) == expected_labels
        assert artifact["preprocessor"] is not None


def test_datasets_endpoint_does_not_500_on_bundled_models():
    response = TestClient(app).get("/api/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert all(item["trained"] for item in payload)
