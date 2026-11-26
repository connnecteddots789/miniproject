from fastapi.testclient import TestClient
from app.main import app, db


def test_delete_experiment_removes_only_selected_history_row():
    conn = db()
    cur = conn.execute(
        'INSERT INTO experiments(dataset,model_type,created_at,metadata) VALUES(?,?,?,?)',
        ('NSL-KDD', 'QA Test Model', '2026-11-19T00:00:00', '{}'),
    )
    experiment_id = cur.lastrowid
    conn.commit()
    conn.close()
    try:
        response = TestClient(app).delete(f'/api/experiments/{experiment_id}')
        assert response.status_code == 200
        assert response.json()['deleted'] is True
        conn = db()
        row = conn.execute('SELECT id FROM experiments WHERE id=?', (experiment_id,)).fetchone()
        conn.close()
        assert row is None
    finally:
        conn = db()
        conn.execute('DELETE FROM experiments WHERE id=?', (experiment_id,))
        conn.commit()
        conn.close()
