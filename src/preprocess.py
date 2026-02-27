import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import read_config, ensure_dir


def load_fashion_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in CSV")
    y = df["label"].to_numpy(dtype=np.int64)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    if X.shape[1] != 784:
        raise ValueError(f"Expected 784 pixel columns, got {X.shape[1]}")
    return X, y


def main(config_path: str) -> None:
    cfg = read_config(config_path)

    raw_train = Path(cfg["DATA"]["raw_train"])
    raw_test = Path(cfg["DATA"]["raw_test"])

    val_size = float(cfg["PREPROCESS"].get("val_size", 0.1))
    random_state = int(cfg["PREPROCESS"].get("random_state", 42))
    normalize = cfg["PREPROCESS"].getboolean("normalize", True)

    out_dir = Path("data/processed")
    ensure_dir(out_dir)

    X_train_full, y_train_full = load_fashion_csv(raw_train)
    X_test, y_test = load_fashion_csv(raw_test)

    if normalize:
        X_train_full = X_train_full / 255.0
        X_test = X_test / 255.0

    stratify = y_train_full
    unique, counts = np.unique(y_train_full, return_counts=True)
    if np.min(counts) < 2:
        stratify = None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(out_dir / "test.npz", X=X_test, y=y_test)

    print("Saved:")
    print(f"- {out_dir / 'train.npz'}: X={X_train.shape}, y={y_train.shape}")
    print(f"- {out_dir / 'val.npz'}:   X={X_val.shape}, y={y_val.shape}")
    print(f"- {out_dir / 'test.npz'}:  X={X_test.shape}, y={y_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.ini")
    args = parser.parse_args()
    main(args.config)