from pathlib import Path
import numpy as np

from ..train import main as train_main


def test_train_saves_model_and_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    X = np.random.random((20, 784)).astype("float32")
    y = np.array([i % 10 for i in range(20)], dtype="int64")
    np.savez_compressed("data/processed/train.npz", X=X, y=y)
    np.savez_compressed("data/processed/val.npz", X=X, y=y)

    Path("config.ini").write_text(
        """
            [LOGREG]
            C = 1.0
            max_iter = 200
            n_jobs = -1
            
            [ARTIFACTS]
            model_path = artifacts/model.joblib
            metrics_path = artifacts/metrics.json
        """.strip(),
        encoding="utf-8",
    )

    train_main("config.ini")

    assert Path("artifacts/model.joblib").exists()
    assert Path("artifacts/metrics.json").exists()