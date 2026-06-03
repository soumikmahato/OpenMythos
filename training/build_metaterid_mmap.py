#!/usr/bin/env python3
"""
Build pre-tokenized mmap shards for high-throughput MetaTerid training.

The output directory contains one fixed-record binary shard per source and a
manifest.json consumed by MMapTokenDataset. Each record has seq_len + 1 tokens
so the training loader can return next-token (x, y) pairs without tokenizing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos.metaterid_tokenizer import MetaTeridTokenizer
from training.metaterid_data import get_mix_sources, iter_source_text, normalize_weights


def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.dtype("uint16") if vocab_size <= 65_536 else np.dtype("uint32")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mix", default="final")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--records-per-source",
        type=int,
        default=100_000,
        help="Maximum fixed-length records to write per source.",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-sample-chars", type=int, default=131_072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = MetaTeridTokenizer(args.tokenizer)
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    record_len = args.seq_len + 1
    manifest = {
        "seq_len": args.seq_len,
        "record_len": record_len,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype.name,
        "shards": [],
    }

    for source in normalize_weights(get_mix_sources(args.mix)):
        shard_name = f"{_safe_name(source.name)}.bin"
        shard_path = out_dir / shard_name
        records = 0
        buf: list[int] = []
        with shard_path.open("wb") as handle:
            for text in iter_source_text(source, args.rank, args.world_size):
                if args.max_sample_chars > 0 and len(text) > args.max_sample_chars:
                    text = text[: args.max_sample_chars]
                buf.extend(tokenizer.encode(text))
                while len(buf) >= record_len and records < args.records_per_source:
                    row = np.asarray(buf[:record_len], dtype=dtype)
                    row.tofile(handle)
                    del buf[:record_len]
                    records += 1
                if records >= args.records_per_source:
                    break

        if records == 0:
            shard_path.unlink(missing_ok=True)
            continue
        manifest["shards"].append(
            {
                "name": source.name,
                "path": shard_name,
                "dtype": dtype.name,
                "record_len": record_len,
                "records": records,
                "weight": source.weight,
            }
        )
        print(f"wrote {records:,} records -> {shard_path}", flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
