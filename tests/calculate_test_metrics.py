import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from src.models.CatVDogModel import CatVDogModel


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _prf_from_cm(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) != 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


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
            details.append({"image": rel_img, "true": true_label, "pred": pred_label})

    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)

    acc = float((y_true_arr == y_pred_arr).mean())

    labels = ["cat", "dog"]
    cm = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    supports: Dict[str, int] = {}

    for lab in labels:
        tp = int(cm[lab][lab])
        fp = int(sum(cm[other][lab] for other in labels if other != lab))
        fn = int(sum(cm[lab][other] for other in labels if other != lab))
        per_class[lab] = _prf_from_cm(tp=tp, fp=fp, fn=fn)
        supports[lab] = int(sum(cm[lab][other] for other in labels))

    macro_precision = float(np.mean([per_class[l]["precision"] for l in labels]))
    macro_recall = float(np.mean([per_class[l]["recall"] for l in labels]))
    macro_f1 = float(np.mean([per_class[l]["f1"] for l in labels]))

    total = float(sum(supports.values()))
    weighted_precision = float(
        sum(per_class[l]["precision"] * supports[l] for l in labels) / total
    ) if total != 0 else 0.0
    weighted_recall = float(
        sum(per_class[l]["recall"] * supports[l] for l in labels) / total
    ) if total != 0 else 0.0
    weighted_f1 = float(
        sum(per_class[l]["f1"] * supports[l] for l in labels) / total
    ) if total != 0 else 0.0

    out: Dict[str, Any] = {
        "count": int(len(items)),
        "accuracy": acc,
        "precision": float(per_class["dog"]["precision"]),
        "recall": float(per_class["dog"]["recall"]),
        "f1": float(per_class["dog"]["f1"]),
        "per_class": per_class,
        "macro_avg": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "weighted_avg": {"precision": weighted_precision, "recall": weighted_recall, "f1": weighted_f1},
        "confusion_matrix": cm,
        "device": device,
        "model_key": model_key,
        "answers_path": str(answers_path),
    }

    if return_details:
        out["details"] = details

    return out


if __name__ == "__main__":
    metrics = calculate_test_metrics()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))