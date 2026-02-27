import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.utils import read_config, ensure_dir


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["y"]


def main(config_path: str) -> None:
    cfg = read_config(config_path)

    train_path = Path("data/processed/train.npz")
    val_path = Path("data/processed/val.npz")

    X_train, y_train = load_npz(train_path)
    X_val, y_val = load_npz(val_path)

    C = float(cfg["LOGREG"].get("C", 1.0))
    max_iter = int(cfg["LOGREG"].get("max_iter", 200))
    n_jobs = int(cfg["LOGREG"].get("n_jobs", -1))

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_pred))
    val_f1m = float(f1_score(y_val, val_pred, average="macro"))

    model_path = Path(cfg["ARTIFACTS"]["model_path"])
    metrics_path = Path(cfg["ARTIFACTS"]["metrics_path"])
    ensure_dir(model_path.parent)

    joblib.dump(model, model_path)

    metrics = {
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1m,
        "model": "LogisticRegression(lbfgs)",
        "params": {"C": C, "max_iter": max_iter, "n_jobs": n_jobs},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"VAL accuracy={val_acc:.4f}, f1_macro={val_f1m:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.ini")
    args = parser.parse_args()
    main(args.config)