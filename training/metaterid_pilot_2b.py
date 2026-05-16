#!/usr/bin/env python3
"""
MetaTerid 2B-token T4 pilot training script.

This run validates the training pipeline before H100 scaling. It saves
checkpoints at 10M, 100M, 500M, 1B, and 2B tokens.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer
from training.metaterid_data import METATERID_T4_PILOT_MIX, MixedTokenDataset
from training.metaterid_optim import build_optimizer

TOKEN_MILESTONES = (10_000_000, 100_000_000, 500_000_000, 1_000_000_000, 2_000_000_000)


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg,
    step: int,
    tokens_seen: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "cfg": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)


def _latest_checkpoint(path: Path) -> Path | None:
    ckpts = sorted(path.glob("tokens_*.pt"))
    return ckpts[-1] if ckpts else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Path to MetaTerid tokenizer.")
    parser.add_argument("--ckpt-dir", default="checkpoints/metaterid_t4_2b")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "muon"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = MetaTeridTokenizer(args.tokenizer)
    cfg = metaterid_t4_pilot()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = args.seq_len

    model = MetaTeridForCausalLM(cfg).to(device)
    optimizer = build_optimizer(
        model,
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    tokens_seen = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume:
        latest = _latest_checkpoint(ckpt_dir)
        if latest is not None:
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt["step"])
            tokens_seen = int(ckpt["tokens_seen"])
            logger.info(f"Resumed from {latest} at {tokens_seen:,} tokens")

    dataset = MixedTokenDataset(
        tokenizer,
        args.seq_len,
        METATERID_T4_PILOT_MIX,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    global_batch_tokens = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = math.ceil(args.target_tokens / global_batch_tokens)
    amp_dtype = torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if device == "cuda"
        else nullcontext()
    )

    logger.info(
        f"device={device} seq_len={args.seq_len} micro_batch={args.micro_batch} "
        f"grad_accum={args.grad_accum} global_batch_tokens={global_batch_tokens:,} "
        f"total_steps={total_steps:,}"
    )

    milestone_index = 0
    while milestone_index < len(TOKEN_MILESTONES) and tokens_seen >= TOKEN_MILESTONES[milestone_index]:
        milestone_index += 1

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()

    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            n_loops = random.randint(cfg.train_min_loops, cfg.train_max_loops)

            with amp_ctx:
                logits = model(x, n_loops=n_loops)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                loss = loss / args.grad_accum

            loss.backward()
            loss_accum += float(loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += global_batch_tokens

        if (step + 1) % args.log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tokens * args.log_every / max(dt, 1e-6)
            logger.info(
                f"step={step + 1:,}/{total_steps:,} tokens={tokens_seen:,} "
                f"loss={loss_accum:.4f} grad_norm={float(grad_norm):.2f} "
                f"lr={lr:.2e} tok/s={tok_per_sec:,.0f}"
            )
            t0 = time.perf_counter()

        while milestone_index < len(TOKEN_MILESTONES) and tokens_seen >= TOKEN_MILESTONES[milestone_index]:
            milestone = TOKEN_MILESTONES[milestone_index]
            path = ckpt_dir / f"tokens_{milestone:013d}.pt"
            _save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                step=step + 1,
                tokens_seen=tokens_seen,
            )
            logger.success(f"Saved milestone checkpoint {path}")
            milestone_index += 1

        if tokens_seen >= args.target_tokens:
            break

    final_path = ckpt_dir / "final.pt"
    _save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        step=step + 1,
        tokens_seen=tokens_seen,
    )
    logger.success(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
