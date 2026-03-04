import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from src.models.CatVDogModel import CatVDogModel


def calculate_test_metrics(
    tests_dir: Union[str, Path] = "tests",
    answers_filename: str = "answers.json",
    config_path: Union[str, Path] = "config.ini",
    model_key: str = "LOG_REG",
    device: str = "cpu",
    return_details: bool = False,
) -> Dict[str, Any]:
    tests_dir = Path(tests_dir)
    answers_path = tests_dir / answers_filename

    with open(answers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Dict[str, str]] = data.get("items", [])
    if not items:
        raise ValueError(f"No items found in {answers_path}")

    model = CatVDogModel(config_path=str(config_path), show_log=False)
    model.set_device(device)
    model.load_classifier(model_key)

    y_true: List[str] = []
    y_pred: List[str] = []
    details: List[Dict[str, Any]] = []

    for it in items:
        rel_img = it["image"]
        true_label = it["label"]
        img_path = (tests_dir / rel_img).resolve()

        pred = model.predict_path(img_path)
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        if isinstance(pred, list) and len(pred) == 1:
            pred = pred[0]

        if isinstance(pred, (int, np.integer)):
            pred_label = "cat" if int(pred) == 0 else "dog"
        else:
            pred_label = str(pred)

        y_true.append(true_label)
        y_pred.append(pred_label)

        if return_details:
            details.append(
                {
                    "image": rel_img,
                    "true": true_label,
                    "pred": pred_label,
                }
            )

    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)

    acc = float((y_true_arr == y_pred_arr).mean())

    labels = ["cat", "dog"]
    cm = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    out: Dict[str, Any] = {
        "count": int(len(items)),
        "accuracy": acc,
        "confusion_matrix": cm,
        "device": device,
        "model_key": model_key,
        "answers_path": str(answers_path),
    }
    if return_details:
        out["details"] = details
    return out