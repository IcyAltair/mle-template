import argparse
import configparser
import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

from src.logger import Logger

PathLike = Union[str, Path]


class CatVDogModel:
    def __init__(self, config_path: str = "config.ini", show_log: bool = True) -> None:
        logger = Logger(show_log)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        read_ok = self.config.read(config_path)
        if not read_ok:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.base_mobile_model = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        )

        self.embed_model = nn.Sequential(
            self.base_mobile_model.conv1,
            self.base_mobile_model.maxpool,
            self.base_mobile_model.stage2,
            self.base_mobile_model.stage3,
            self.base_mobile_model.stage4,
            self.base_mobile_model.conv5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ).eval()

        self.preprocess = self.build_preprocess()

        self.device: torch.device = torch.device("cpu")
        self.classifier: Any = None

        self.log.info("Service is ready")

    def build_preprocess(self) -> transforms.Compose:
        e = self.config["EMBEDDINGS"] if "EMBEDDINGS" in self.config else {}
        img_resize = int(e.get("img_resize", 256))
        img_crop = int(e.get("img_crop", 224))
        return transforms.Compose(
            [
                transforms.Resize(img_resize),
                transforms.CenterCrop(img_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_logreg_params(self, section: str = "LOG_REG") -> Dict[str, Any]:
        if section not in self.config:
            raise KeyError(f"Config section '{section}' not found")

        cfg = self.config[section]

        params: Dict[str, Any] = {}
        if "max_iter" in cfg:
            params["max_iter"] = cfg.getint("max_iter")
        if "C" in cfg:
            params["C"] = cfg.getfloat("C")
        if "solver" in cfg:
            params["solver"] = cfg.get("solver")
        if "penalty" in cfg:
            params["penalty"] = cfg.get("penalty")
        if "class_weight" in cfg:
            cw = cfg.get("class_weight")
            params["class_weight"] = None if cw.lower() == "none" else cw
        if "random_state" in cfg:
            params["random_state"] = cfg.getint("random_state")

        n_jobs = cfg.get("n_jobs", fallback=None)
        if n_jobs is not None:
            params["n_jobs"] = int(n_jobs)

        return params

    def _get_split_params(self) -> Dict[str, Any]:
        if "SPLIT" not in self.config:
            return {"test_size": 0.2, "random_state": 42, "stratify": True}
        s = self.config["SPLIT"]
        return {
            "test_size": s.getfloat("test_size", fallback=0.2),
            "random_state": s.getint("random_state", fallback=42),
            "stratify": s.getboolean("stratify", fallback=True),
        }

    def _next_experiment_dir(self, experiments_dir: PathLike) -> Path:
        root = Path(experiments_dir)
        root.mkdir(parents=True, exist_ok=True)

        existing = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("exp_")]
        max_id = 0
        for p in existing:
            parts = p.name.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                max_id = max(max_id, int(parts[1]))

        exp_id = max_id + 1
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"exp_{exp_id:04d}_{ts}"
        out = root / name
        out.mkdir(parents=True, exist_ok=False)
        return out

    def _dump_json(self, path: PathLike, obj: Dict[str, Any]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _load_json(self, path: PathLike) -> Optional[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _is_better(self, cur: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
        if best is None:
            return True
        cur_f1 = float(cur.get("f1", -1.0))
        best_f1 = float(best.get("f1", -1.0))
        if cur_f1 > best_f1:
            return True
        if cur_f1 < best_f1:
            return False
        cur_acc = float(cur.get("accuracy", -1.0))
        best_acc = float(best.get("accuracy", -1.0))
        return cur_acc > best_acc

    def set_device(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda is not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("mps is not available")

        self.device = torch.device(device)
        self.embed_model = self.embed_model.to(self.device)
        self.embed_model.eval()

    def load_classifier(self, model_key: str) -> Any:
        if model_key not in self.config:
            raise KeyError(f"Model section '{model_key}' not found in config.ini")
        if "path" not in self.config[model_key]:
            raise KeyError(f"Missing 'path' in config section '{model_key}'")

        p = self.config[model_key]["path"]
        self.classifier = joblib.load(p)
        return self.classifier

    def save_classifier(self, path: PathLike) -> None:
        if self.classifier is None:
            raise RuntimeError("Classifier is not trained/loaded")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, str(p))

    def _ensure_ready(self, classifier: Optional[Any] = None) -> Any:
        clf = classifier if classifier is not None else self.classifier
        if clf is None:
            raise RuntimeError("Classifier is not loaded/trained. Call load_classifier() or train_classifier().")
        return clf

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image.convert("RGB")).unsqueeze(0)

    @torch.no_grad()
    def embed_tensor(self, x: torch.Tensor, device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
        dev = torch.device(device) if device is not None else self.device
        x = x.to(dev)
        emb = self.embed_model(x).squeeze(0)
        emb = torch.nn.functional.normalize(emb, dim=0)
        return emb.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_pil(self, image: Image.Image, device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
        x = self.preprocess_image(image)
        return self.embed_tensor(x, device=device)

    @torch.no_grad()
    def embed_path(self, image_path: PathLike, device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
        img = Image.open(str(image_path)).convert("RGB")
        return self.embed_pil(img, device=device)

    @torch.no_grad()
    def predict_path(
        self,
        image_path: PathLike,
        classifier: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        clf = self._ensure_ready(classifier)
        emb = self.embed_path(image_path, device=device)
        return clf.predict(emb.reshape(1, -1))

    @torch.no_grad()
    def predict_pil(
        self,
        image: Image.Image,
        classifier: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        clf = self._ensure_ready(classifier)
        emb = self.embed_pil(image, device=device)
        return clf.predict(emb.reshape(1, -1))

    @torch.no_grad()
    def predict_bytes(
        self,
        image_bytes: bytes,
        classifier: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.predict_pil(img, classifier=classifier, device=device)

    def predict_dir(
        self,
        dir_image_path: PathLike,
        classifier: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        recursive: bool = False,
        return_paths: bool = False,
        skip_errors: bool = True,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        clf = self._ensure_ready(classifier)
        dir_path = Path(dir_image_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        exts_l = tuple(e.lower() for e in exts)
        pattern = "**/*" if recursive else "*"
        files = [p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in exts_l]

        preds: List[Any] = []
        paths: List[str] = []
        errors: Dict[str, str] = {}

        for p in files:
            try:
                pred = self.predict_path(p, classifier=clf, device=device)
                preds.append(pred[0] if isinstance(pred, (list, np.ndarray)) and len(pred) == 1 else pred)
                paths.append(str(p))
            except Exception:
                if not skip_errors:
                    raise
                errors[str(p)] = traceback.format_exc()

        out_preds = np.array(preds, dtype=object)
        if return_paths:
            return {"predictions": out_preds, "paths": paths, "errors": errors}
        return out_preds

    def prepare_training_data_from_dir(
        self,
        data_dir: PathLike,
        class_map: Dict[str, int],
        device: Optional[Union[str, torch.device]] = None,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        recursive: bool = False,
        limit_per_class: Optional[Dict[int, int]] = None,
        shuffle: bool = True,
        seed: int = 42,
        skip_errors: bool = True,
        data_frac: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dir_path = Path(data_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        exts_l = tuple(e.lower() for e in exts)
        pattern = "**/*" if recursive else "*"
        all_files = [p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in exts_l]

        if not (0.0 < data_frac <= 1.0):
            raise ValueError("data_frac must be in (0, 1]")

        rng = np.random.default_rng(seed)

        files_by_class: Dict[int, List[Path]] = {}
        for p in all_files:
            prefix = p.name.split(".", 1)[0]
            if prefix not in class_map:
                continue
            y = int(class_map[prefix])
            files_by_class.setdefault(y, []).append(p)

        selected: List[Path] = []
        for y, files in files_by_class.items():
            if len(files) == 0:
                continue
            if shuffle:
                rng.shuffle(files)
            n_take = max(1, int(len(files) * float(data_frac)))
            selected.extend(files[:n_take])

        if shuffle:
            rng.shuffle(selected)

        all_files = selected

        counts: Dict[int, int] = {}
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for p in tqdm(all_files, desc="Extracting embeddings", unit="img"):
            prefix = p.name.split(".", 1)[0]
            if prefix not in class_map:
                continue

            y = int(class_map[prefix])
            if limit_per_class is not None:
                counts.setdefault(y, 0)
                if y in limit_per_class and counts[y] >= int(limit_per_class[y]):
                    continue

            try:
                emb = self.embed_path(p, device=device)
                X_list.append(emb)
                y_list.append(y)
                if limit_per_class is not None:
                    counts[y] = counts.get(y, 0) + 1
            except Exception:
                if not skip_errors:
                    raise

        if len(X_list) == 0:
            raise RuntimeError("No samples collected. Check class_map and file naming scheme.")

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None,
        model_key: str = "LOG_REG",
        **override_logreg_kwargs: Any,
    ) -> Dict[str, Any]:
        if X.ndim != 2:
            raise ValueError("X must be 2D array: (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D array: (n_samples,)")

        split_cfg = self._get_split_params()
        ts = split_cfg["test_size"] if test_size is None else float(test_size)
        rs = split_cfg["random_state"] if random_state is None else int(random_state)
        st = split_cfg["stratify"] if stratify is None else bool(stratify)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=ts,
            random_state=rs,
            stratify=y if st else None,
        )

        logreg_params = self._get_logreg_params(model_key)
        logreg_params.update(override_logreg_kwargs)

        clf = LogisticRegression(**logreg_params)
        clf.fit(X_train, y_train)
        self.classifier = clf

        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="binary", pos_label=1)),
            "recall": float(recall_score(y_test, y_pred, average="binary", pos_label=1)),
            "f1": float(f1_score(y_test, y_pred, average="binary", pos_label=1)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
        return metrics

    def build_cli(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="EmbeddingLogRegService")
        parser.add_argument("--model", type=str, default="LOG_REG", choices=["LOG_REG"])
        parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
        parser.add_argument("--mode", type=str, default="single", choices=["single", "directory", "train"])
        parser.add_argument("--path", type=str, default="")
        parser.add_argument("--recursive", action="store_true")
        parser.add_argument("--return_paths", action="store_true")
        parser.add_argument("--skip_errors", action="store_true")
        parser.add_argument("--test_size", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_frac", type=float, default=1.0)
        parser.add_argument("--experiments_dir", type=str, default="experiments")
        parser.add_argument("--best_dir", type=str, default="experiments")
        parser.add_argument("--best_metric", type=str, default="f1", choices=["f1", "accuracy", "precision", "recall"])
        return parser

    def run_from_args(self, args: argparse.Namespace) -> Any:
        self.set_device(args.device)

        if args.mode in {"single", "directory"}:
            self.load_classifier(args.model)

        if args.mode == "single":
            pred = self.predict_path(args.path)
            return {"mode": "single", "path": args.path, "prediction": pred}

        if args.mode == "directory":
            res = self.predict_dir(
                args.path,
                recursive=getattr(args, "recursive", False),
                return_paths=getattr(args, "return_paths", False),
                skip_errors=getattr(args, "skip_errors", True),
            )
            return {"mode": "directory", "path": args.path, "result": res}

        if args.mode == "train":
            class_map = {"cat": 0, "dog": 1}

            X, y = self.prepare_training_data_from_dir(
                args.path,
                class_map=class_map,
                device=args.device,
                recursive=getattr(args, "recursive", False),
                seed=getattr(args, "seed", 42),
                skip_errors=getattr(args, "skip_errors", True),
                data_frac=getattr(args, "data_frac", 1.0),
            )

            metrics = self.train_classifier(
                X,
                y,
                test_size=getattr(args, "test_size", 0.2),
                random_state=getattr(args, "seed", 42),
            )

            exp_dir = self._next_experiment_dir(getattr(args, "experiments_dir", "experiments"))
            model_path = exp_dir / "model.pkl"
            report_path = exp_dir / "report.json"

            self.save_classifier(model_path)

            split_cfg = self._get_split_params()
            logreg_cfg = self._get_logreg_params("LOG_REG")
            embeddings_cfg = dict(self.config["EMBEDDINGS"]) if "EMBEDDINGS" in self.config else {}

            report = {
                "timestamp": exp_dir.name.split("_", 2)[-1] if "_" in exp_dir.name else exp_dir.name,
                "experiment_dir": str(exp_dir),
                "data": {
                    "path": str(args.path),
                    "data_frac": float(getattr(args, "data_frac", 1.0)),
                    "n_samples": int(len(y)),
                    "class_map": class_map,
                },
                "device": str(args.device),
                "split": {
                    "test_size": float(getattr(args, "test_size", split_cfg.get("test_size", 0.2))),
                    "random_state": int(getattr(args, "seed", split_cfg.get("random_state", 42))),
                    "stratify": bool(split_cfg.get("stratify", True)),
                },
                "embeddings": embeddings_cfg,
                "log_reg": logreg_cfg,
                "metrics": metrics,
                "artifacts": {
                    "model": str(model_path),
                    "report": str(report_path),
                },
            }

            self._dump_json(report_path, report)

            best_root = Path(getattr(args, "best_dir", "experiments"))
            best_root.mkdir(parents=True, exist_ok=True)
            best_model_path = best_root / "model.pkl"
            best_metrics_path = best_root / "model_metrics.json"

            prev_best = self._load_json(best_metrics_path)
            best_prev_metrics = prev_best.get("metrics") if isinstance(prev_best, dict) else None

            metric_name = str(getattr(args, "best_metric", "f1"))
            cur_metrics = dict(metrics)
            cur_score = float(cur_metrics.get(metric_name, -1.0))
            prev_score = float(best_prev_metrics.get(metric_name, -1.0)) if isinstance(best_prev_metrics, dict) else None

            is_better = False
            if prev_score is None:
                is_better = True
            else:
                if cur_score > float(prev_score):
                    is_better = True
                elif cur_score == float(prev_score):
                    is_better = self._is_better(cur_metrics, best_prev_metrics)

            updated_best = False
            if is_better:
                self.save_classifier(best_model_path)
                best_payload = {
                    "updated_at": dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "best_metric": metric_name,
                    "metrics": cur_metrics,
                    "from_experiment": str(exp_dir),
                    "artifacts": {"model": str(best_model_path), "metrics": str(best_metrics_path)},
                }
                self._dump_json(best_metrics_path, best_payload)
                updated_best = True

            return {
                "mode": "train",
                "path": args.path,
                "metrics": metrics,
                "experiment_dir": str(exp_dir),
                "experiment_model_path": str(model_path),
                "experiment_report_path": str(report_path),
                "best_updated": bool(updated_best),
                "best_model_path": str(best_model_path),
                "best_metrics_path": str(best_metrics_path),
                "best_metric": metric_name,
            }

        raise ValueError(f"Unknown mode: {args.mode}")

    def run_cli(self) -> bool:
        parser = self.build_cli()
        args = parser.parse_args()
        try:
            result = self.run_from_args(args)
            print(result)
            return True
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    CatVDogModel().run_cli()