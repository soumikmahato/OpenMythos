from __future__ import annotations

import argparse
import glob
from pathlib import Path

from open_mythos.metaterid_tokenizer import (
    METATERID_SPECIAL_TOKENS,
    METATERID_VOCAB_SIZE,
    train_metaterid_tokenizer,
)


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = (
            sorted(Path(path) for path in glob.glob(pattern, recursive=True))
            if any(c in pattern for c in "*?[")
            else []
        )
        if matches:
            files.extend(path for path in matches if path.is_file())
            continue

        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.txt") if p.is_file()))
        elif path.is_file():
            files.append(path)

    unique: dict[Path, None] = {}
    for path in files:
        unique[path.resolve()] = None
    return list(unique)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MetaTerid 65,536-token byte-level BPE tokenizer."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Text files, directories of .txt files, or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizers/metaterid-tokenizer-v1",
        help="Directory for the saved HuggingFace-compatible tokenizer.",
    )
    parser.add_argument("--vocab-size", type=int, default=METATERID_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No training text files found.")

    output_dir = train_metaterid_tokenizer(
        files,
        args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"Saved MetaTerid tokenizer to {output_dir}")
    print(f"Training files: {len(files)}")
    print(f"Target vocab size: {args.vocab_size:,}")
    print("Special tokens:")
    for token in METATERID_SPECIAL_TOKENS:
        print(f"  {token}")


if __name__ == "__main__":
    main()
