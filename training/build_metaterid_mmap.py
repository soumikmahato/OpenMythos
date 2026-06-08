#!/usr/bin/env python3
"""
Build pre-tokenized mmap shards for high-throughput MetaTerid training.

The output directory contains one fixed-record binary shard per source and a
manifest.json consumed by MMapTokenDataset. Each record has seq_len + 1 tokens
so the training loader can return next-token (x, y) pairs without tokenizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser.add_argument(
        "--mixes",
        default="",
        help=(
            "Optional comma-separated mixes to build into subdirectories under "
            "--out-dir, e.g. main_base_first_v1,main_base_middle_v1,main_base_final_v1."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--records-per-source",
        type=int,
        default=100_000,
        help=(
            "Fallback fixed-length record budget per source. Ignored when "
            "--target-tokens is positive."
        ),
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=0,
        help=(
            "Total training-token budget this mmap corpus should cover. Source "
            "record budgets are allocated proportionally to normalized mix weights."
        ),
    )
    parser.add_argument(
        "--corpus-multiplier",
        type=float,
        default=1.25,
        help=(
            "Build this multiple of --target-tokens to reduce record repetition. "
            "Only used with --target-tokens."
        ),
    )
    parser.add_argument(
        "--dedup",
        choices=["none", "exact"],
        default="exact",
        help="Deduplicate cleaned documents across all sources in this mix build.",
    )
    parser.add_argument(
        "--dedup-against",
        action="append",
        default=[],
        help=(
            "Prior mmap directory or document_hashes.uint64 file whose document "
            "hashes should be excluded. Repeat for multiple prior corpora."
        ),
    )
    parser.add_argument(
        "--source-shuffle-seed",
        type=int,
        default=0,
        help=(
            "Non-zero seed used to shuffle source shards/stream buffers. Use a "
            "different value for each continuation corpus to avoid rereading dataset heads."
        ),
    )
    parser.add_argument(
        "--source-shuffle-buffer",
        type=int,
        default=10_000,
        help="Streaming dataset shuffle buffer size when --source-shuffle-seed is non-zero.",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-sample-chars", type=int, default=131_072)
    return parser.parse_args()


def _requested_records(args: argparse.Namespace, source_weight: float, record_len: int) -> int:
    if args.target_tokens > 0:
        requested_tokens = args.target_tokens * args.corpus_multiplier * source_weight
        return max(1, math.ceil(requested_tokens / record_len))
    return args.records_per_source


def _document_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _load_prior_hashes(paths: list[str]) -> set[int]:
    hashes: set[int] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            path = path / "document_hashes.uint64"
        if not path.exists():
            raise FileNotFoundError(f"Missing prior dedup hash index: {path}")
        if path.stat().st_size % np.dtype("uint64").itemsize:
            raise ValueError(f"Invalid uint64 dedup hash index: {path}")
        hashes.update(int(value) for value in np.fromfile(path, dtype=np.uint64))
    return hashes


def _build_one(args: argparse.Namespace, tokenizer: MetaTeridTokenizer, mix_name: str, out_dir: Path) -> None:
    if args.target_tokens < 0:
        raise ValueError("--target-tokens must be non-negative")
    if args.records_per_source <= 0:
        raise ValueError("--records-per-source must be positive")
    if args.corpus_multiplier <= 0:
        raise ValueError("--corpus-multiplier must be positive")
    if args.source_shuffle_buffer <= 0:
        raise ValueError("--source-shuffle-buffer must be positive")
    if args.dedup == "none" and args.dedup_against:
        raise ValueError("--dedup-against requires --dedup exact")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_tmp_path = out_dir / "manifest.json.building"
    manifest_path.unlink(missing_ok=True)
    manifest_tmp_path.unlink(missing_ok=True)
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    record_len = args.seq_len + 1
    manifest = {
        "format_version": 2,
        "document_boundary": "bos_eos",
        "mix": mix_name,
        "seq_len": args.seq_len,
        "record_len": record_len,
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype.name,
        "target_tokens": args.target_tokens,
        "corpus_multiplier": args.corpus_multiplier if args.target_tokens > 0 else None,
        "dedup": args.dedup,
        "dedup_against": args.dedup_against,
        "source_shuffle_seed": args.source_shuffle_seed,
        "source_shuffle_buffer": args.source_shuffle_buffer,
        "rank": args.rank,
        "world_size": args.world_size,
        "shards": [],
    }
    seen_documents: set[int] | None = (
        _load_prior_hashes(args.dedup_against) if args.dedup == "exact" else None
    )
    prior_document_count = len(seen_documents) if seen_documents is not None else 0
    manifest["prior_document_hashes"] = prior_document_count
    total_requested_records = 0

    for source_idx, source in enumerate(normalize_weights(get_mix_sources(mix_name))):
        shard_name = f"{_safe_name(source.name)}.bin"
        shard_path = out_dir / shard_name
        shard_tmp_path = shard_path.with_suffix(shard_path.suffix + ".building")
        shard_path.unlink(missing_ok=True)
        shard_tmp_path.unlink(missing_ok=True)
        requested_records = _requested_records(args, source.weight, record_len)
        total_requested_records += requested_records
        records = 0
        accepted_documents = 0
        duplicate_documents = 0
        buf: list[int] = []
        with shard_tmp_path.open("wb") as handle:
            shuffle_seed = (
                args.source_shuffle_seed + 1009 * source_idx
                if args.source_shuffle_seed
                else None
            )
            for text in iter_source_text(
                source,
                args.rank,
                args.world_size,
                shuffle_seed=shuffle_seed,
                shuffle_buffer_size=args.source_shuffle_buffer,
                repeat_local=False,
            ):
                if args.max_sample_chars > 0 and len(text) > args.max_sample_chars:
                    text = text[: args.max_sample_chars]
                if seen_documents is not None:
                    document_hash = _document_hash(text)
                    if document_hash in seen_documents:
                        duplicate_documents += 1
                        continue
                    seen_documents.add(document_hash)
                accepted_documents += 1
                buf.extend(tokenizer.encode_document(text))
                while len(buf) >= record_len and records < requested_records:
                    row = np.asarray(buf[:record_len], dtype=dtype)
                    row.tofile(handle)
                    del buf[:record_len]
                    records += 1
                if records >= requested_records:
                    break

        if records == 0:
            shard_tmp_path.unlink(missing_ok=True)
            print(
                f"warning: source {source.name} produced no complete records "
                f"(requested {requested_records:,})",
                flush=True,
            )
            continue
        shard_tmp_path.replace(shard_path)
        coverage = records / requested_records
        expected_training_records = (
            args.target_tokens * source.weight / record_len
            if args.target_tokens > 0
            else None
        )
        expected_passes = (
            expected_training_records / records
            if expected_training_records is not None
            else None
        )
        manifest["shards"].append(
            {
                "name": source.name,
                "path": shard_name,
                "dtype": dtype.name,
                "record_len": record_len,
                "records": records,
                "requested_records": requested_records,
                "coverage": coverage,
                "accepted_documents": accepted_documents,
                "duplicate_documents": duplicate_documents,
                "expected_training_records": expected_training_records,
                "expected_passes": expected_passes,
                "weight": source.weight,
            }
        )
        print(
            f"wrote {records:,}/{requested_records:,} records "
            f"({coverage:.1%} coverage, {duplicate_documents:,} duplicates skipped) "
            f"-> {shard_path}",
            flush=True,
        )
        if coverage < 0.95:
            print(
                f"warning: source {source.name} only reached {coverage:.1%} of its "
                "weighted corpus budget; this source will repeat more often than intended",
                flush=True,
            )
        if expected_passes is not None and expected_passes > 1.0:
            print(
                f"warning: source {source.name} is expected to cycle "
                f"{expected_passes:.2f} times during training",
                flush=True,
            )

    total_records = sum(int(shard["records"]) for shard in manifest["shards"])
    manifest["total_records"] = total_records
    manifest["total_packed_tokens"] = total_records * record_len
    manifest["requested_records"] = total_requested_records
    manifest["record_coverage"] = (
        total_records / total_requested_records if total_requested_records else 0.0
    )
    if total_records == 0:
        raise RuntimeError(f"No mmap records were produced for mix {mix_name}")
    if args.target_tokens > 0 and manifest["total_packed_tokens"] < args.target_tokens:
        raise RuntimeError(
            f"Built only {manifest['total_packed_tokens']:,} packed tokens for a "
            f"{args.target_tokens:,}-token run. Increase source coverage or lower "
            "--target-tokens; refusing to publish an undersized corpus manifest."
        )

    if seen_documents is not None:
        hash_path = out_dir / "document_hashes.uint64"
        hash_tmp_path = out_dir / "document_hashes.uint64.building"
        hash_tmp_path.unlink(missing_ok=True)
        np.fromiter(seen_documents, dtype=np.uint64, count=len(seen_documents)).tofile(
            hash_tmp_path
        )
        hash_tmp_path.replace(hash_path)
        manifest["document_hashes"] = hash_path.name
        manifest["document_hash_count"] = len(seen_documents)
        manifest["new_document_hashes"] = len(seen_documents) - prior_document_count

    manifest_tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_tmp_path.replace(manifest_path)
    print(f"manifest -> {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    tokenizer = MetaTeridTokenizer(args.tokenizer)
    mixes = [part.strip() for part in args.mixes.split(",") if part.strip()]
    if not mixes:
        _build_one(args, tokenizer, args.mix, Path(args.out_dir))
        return
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for mix_name in mixes:
        _build_one(args, tokenizer, mix_name, root / mix_name)


if __name__ == "__main__":
    main()
