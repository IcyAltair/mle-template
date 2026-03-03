import argparse
import configparser
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import transforms

from logger import Logger

SHOW_LOG = True

PathLike = Union[str, Path]


class Predictor:
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
        )

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.device: torch.device = torch.device("cpu")
        self.classifier: Any = None

        self.log.info("Predictor is ready")

    def set_device(self, device: str) -> None:
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")

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
        with open(p, "rb") as f:
            self.classifier = pickle.load(f)
        return self.classifier

    def _ensure_ready(self, classifier: Optional[Any] = None) -> Any:
        clf = classifier if classifier is not None else self.classifier
        if clf is None:
            raise RuntimeError("Classifier is not loaded. Call load_classifier() or pass classifier explicitly.")
        return clf

    @torch.no_grad()
    def embed_image(self, image: Image.Image, device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
        dev = torch.device(device) if device is not None else self.device
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(dev)
        emb = self.embed_model(x).squeeze(0)
        emb = torch.nn.functional.normalize(emb, dim=0)
        return emb.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_single_image(
            self,
            classifier: Optional[Any],
            image_path: PathLike,
            device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        clf = self._ensure_ready(classifier)
        img = Image.open(str(image_path)).convert("RGB")
        emb = self.embed_image(img, device=device)
        return clf.predict(emb.reshape(1, -1))

    @torch.no_grad()
    def predict_single_pil(
            self,
            classifier: Optional[Any],
            image: Image.Image,
            device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        clf = self._ensure_ready(classifier)
        emb = self.embed_image(image, device=device)
        return clf.predict(emb.reshape(1, -1))

    @torch.no_grad()
    def predict_single_bytes(
            self,
            classifier: Optional[Any],
            image_bytes: bytes,
            device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.predict_single_pil(classifier, img, device=device)

    def predict_dir_images(
            self,
            classifier: Optional[Any],
            dir_image_path: PathLike,
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
                pred = self.predict_single_image(clf, p, device=device)
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

    def build_cli(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Predictor")
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            required=True,
            default="LOG_REG",
            const="LOG_REG",
            nargs="?",
            choices=["LOG_REG"],
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            required=True,
            default="cpu",
            const="cpu",
            nargs="?",
            choices=["cpu", "cuda", "mps"],
        )
        parser.add_argument(
            "-md",
            "--mode",
            type=str,
            required=True,
            default="single",
            const="single",
            nargs="?",
            choices=["single", "directory"],
        )
        parser.add_argument(
            "-pth",
            "--path",
            type=str,
            required=True,
            default="",
            const="",
            nargs="?",
        )
        parser.add_argument(
            "--recursive",
            action="store_true",
        )
        parser.add_argument(
            "--return_paths",
            action="store_true",
        )
        parser.add_argument(
            "--skip_errors",
            action="store_true",
        )
        return parser

    def predict_from_args(self, args: argparse.Namespace) -> Any:
        self.set_device(args.device)
        self.load_classifier(args.model)

        if args.mode == "single":
            pred = self.predict_single_image(self.classifier, args.path)
            return {"mode": "single", "path": args.path, "prediction": pred}

        if args.mode == "directory":
            res = self.predict_dir_images(
                self.classifier,
                args.path,
                recursive=getattr(args, "recursive", False),
                return_paths=getattr(args, "return_paths", False),
                skip_errors=getattr(args, "skip_errors", True),
            )
            return {"mode": "directory", "path": args.path, "result": res}

        raise ValueError(f"Unknown mode: {args.mode}")

    def predict(self) -> bool:
        parser = self.build_cli()
        args = parser.parse_args()
        try:
            result = self.predict_from_args(args)
            if args.mode == "single":
                print(result["prediction"])
            else:
                print(result["result"])
            return True
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    Predictor().predict()
