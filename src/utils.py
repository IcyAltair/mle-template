import configparser
from pathlib import Path
from typing import Union


def read_config(path: Union[str, Path]) -> configparser.ConfigParser:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    cfg = configparser.ConfigParser()
    cfg.read(p, encoding="utf-8")
    return cfg


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)