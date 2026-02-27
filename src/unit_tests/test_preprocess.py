from pathlib import Path
import numpy as np

from ..preprocess import main as preprocess_main


def test_preprocess_outputs_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    header = "label," + ",".join([f"pixel{i}" for i in range(784)]) + "\n"
    row0 = "0," + ",".join(["0"] * 784) + "\n"
    row1 = "1," + ",".join(["255"] * 784) + "\n"

    (raw_dir / "fashion-mnist_train.csv").write_text(header + row0 + row1, encoding="utf-8")
    (raw_dir / "fashion-mnist_test.csv").write_text(header + row0 + row1, encoding="utf-8")

    Path("config.ini").write_text(
        """
            [DATA]
            raw_train = data/raw/fashion-mnist_train.csv
            raw_test  = data/raw/fashion-mnist_test.csv
            [PREPROCESS]
            val_size = 0.5
            random_state = 42
            normalize = true
        """.strip(),
        encoding="utf-8",
    )

    preprocess_main("config.ini")

    assert Path("data/processed/train.npz").exists()
    assert Path("data/processed/val.npz").exists()
    assert Path("data/processed/test.npz").exists()

    tr = np.load("data/processed/train.npz")
    assert tr["X"].shape[1] == 784