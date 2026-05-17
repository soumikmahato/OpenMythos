#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a model-only MetaTerid checkpoint for low-RAM Kaggle resume."
    )
    parser.add_argument("--input", default="checkpoints/metaterid_t4_2b/final.pt")
    parser.add_argument("--output", default="checkpoints/metaterid_t4_2b/model_only.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    slim = {
        "step": int(ckpt["step"]),
        "tokens_seen": int(ckpt["tokens_seen"]),
        "cfg": ckpt["cfg"],
        "model": ckpt["model"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {"step": slim["step"], "tokens_seen": slim["tokens_seen"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    print(f"tokens_seen={slim['tokens_seen']:,}")


if __name__ == "__main__":
    main()
