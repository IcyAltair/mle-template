import argparse
import json
import random
import shutil
from pathlib import Path


def parse_label_from_filename(filename: str) -> str:
    return filename.split(".", 1)[0]


def build_test_set(
    src_dir: Path,
    out_dir: Path,
    n: int = 30,
    seed: int = 42,
    exts=(".jpg", ".jpeg", ".png"),
) -> None:
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    candidates = []
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            label = parse_label_from_filename(p.name)
            if label in ("cat", "dog"):
                candidates.append(p)

    if len(candidates) < n:
        raise ValueError(f"Недостаточно картинок: найдено {len(candidates)}, нужно {n}")

    random.seed(seed)
    chosen = random.sample(candidates, n)

    answers = []
    for p in chosen:
        label = parse_label_from_filename(p.name)
        dst = images_out / p.name
        if dst.exists():
            dst = images_out / f"{p.stem}_{random.randint(1000,9999)}{p.suffix}"

        shutil.copy2(p, dst)
        rel_path = dst.relative_to(out_dir).as_posix()

        answers.append(
            {
                "image": rel_path,
                "label": label
            }
        )

    answers_path = out_dir / "answers.json"
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dir": str(src_dir),
                "count": len(answers),
                "seed": seed,
                "items": answers,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Done: {len(answers)} images -> {images_out}")
    print(f"Answers: {answers_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/dogs-vs-cats/train")
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_test_set(
        src_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        n=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()