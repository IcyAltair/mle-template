from pathlib import Path
import numpy as np
import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from ..api import app


def test_predict_fill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    X = np.random.random((50, 784)).astype("float32")
    y = np.array([i % 10 for i in range(50)], dtype="int64")
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X, y)

    joblib.dump(model, "artifacts/model.joblib")

    client = TestClient(app)
    r = client.post("/predict", json={"fill": 0})
    assert r.status_code == 200
    data = r.json()
    assert "class_id" in data and "proba" in data
    assert len(data["proba"]) == 10