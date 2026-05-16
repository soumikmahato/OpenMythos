#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def _latest_tokens(ckpt_dir: Path) -> int:
    final = ckpt_dir / "final.pt"
    if final.exists():
        ckpt = torch.load(final, map_location="cpu", weights_only=False)
        return int(ckpt.get("tokens_seen", 0))

    token_ckpts = sorted(ckpt_dir.glob("tokens_*.pt"))
    if not token_ckpts:
        return 0

    ckpt = torch.load(token_ckpts[-1], map_location="cpu", weights_only=False)
    return int(ckpt.get("tokens_seen", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one 250M-token MetaTerid Kaggle chunk on 2x T4."
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--ckpt-dir", default="checkpoints/metaterid_t4_2b")
    parser.add_argument("--chunk-tokens", type=int, default=250_000_000)
    parser.add_argument("--max-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    tokens_seen = _latest_tokens(ckpt_dir)
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
    ]
    if tokens_seen > 0:
        command.append("--resume")

    print(f"Current tokens: {tokens_seen:,}")
    print(f"Target tokens after this chunk: {target_tokens:,}")
    print("Command:")
    print(" ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
