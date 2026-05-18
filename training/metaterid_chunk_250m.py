#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def _latest_tokens(ckpt_dir: Path) -> int:
    final_meta = ckpt_dir / "final.json"
    if final_meta.exists():
        return int(json.loads(final_meta.read_text(encoding="utf-8")).get("tokens_seen", 0))

    final = ckpt_dir / "final.pt"
    if final.exists():
        ckpt = torch.load(final, map_location="cpu", weights_only=False)
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        del ckpt
        return tokens_seen

    token_ckpts = sorted(ckpt_dir.glob("tokens_*.pt"))
    if not token_ckpts:
        return 0

    meta = token_ckpts[-1].with_suffix(".json")
    if meta.exists():
        return int(json.loads(meta.read_text(encoding="utf-8")).get("tokens_seen", 0))

    ckpt = torch.load(token_ckpts[-1], map_location="cpu", weights_only=False)
    tokens_seen = int(ckpt.get("tokens_seen", 0))
    del ckpt
    return tokens_seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one 250M-token MetaTerid Kaggle chunk on 2x T4."
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--ckpt-dir", default="checkpoints/metaterid_t4_2b")
    parser.add_argument("--chunk-tokens", type=int, default=250_000_000)
    parser.add_argument("--max-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument(
        "--max-sample-chars",
        type=int,
        default=65_536,
        help="Maximum characters from one source document to tokenize at once. Use 0 to disable.",
    )
    parser.add_argument(
        "--current-tokens",
        type=int,
        default=None,
        help="Known current token count. Use this on Kaggle to avoid loading final.pt in the launcher.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Forward --resume-optimizer to the training script. Off by default for Kaggle RAM safety.",
    )
    parser.add_argument(
        "--mix",
        default="kaggle_chunk",
        choices=[
            "pilot",
            "kaggle_chunk",
            "kaggle_fineweb_only",
            "kaggle_no_math",
            "kaggle_fineweb_math",
            "kaggle_fineweb_code_instruct",
        ],
        help="Dataset mix preset forwarded to metaterid_pilot_2b.py.",
    )
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    tokens_seen = args.current_tokens if args.current_tokens is not None else _latest_tokens(ckpt_dir)
    if tokens_seen >= args.max_tokens:
        print(f"Run already reached {tokens_seen:,} tokens >= {args.max_tokens:,}.")
        return

    target_tokens = min(tokens_seen + args.chunk_tokens, args.max_tokens)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "training/metaterid_pilot_2b.py",
        "--tokenizer",
        args.tokenizer,
        "--ckpt-dir",
        str(ckpt_dir),
        "--target-tokens",
        str(target_tokens),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch",
        str(args.micro_batch),
        "--grad-accum",
        str(args.grad_accum),
        "--num-workers",
        str(args.num_workers),
        "--log-every",
        str(args.log_every),
        "--max-sample-chars",
        str(args.max_sample_chars),
        "--mix",
        args.mix,
    ]
    if tokens_seen > 0:
        command.append("--resume")
    if args.resume_optimizer:
        command.append("--resume-optimizer")
    if args.log_memory:
        command.append("--log-memory")

    print(f"Current tokens: {tokens_seen:,}")
    print(f"Target tokens after this chunk: {target_tokens:,}")
    print("Command:")
    print(" ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
