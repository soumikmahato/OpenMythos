#!/usr/bin/env python3
"""
Main MetaTerid pretraining entrypoint.

This script is for the serious post-Kaggle path: H100-class runs, larger token
budgets, explicit checkpoint milestones, optional AdamW -> Muon-style optimizer
switching, BF16/FP16 autocast, and richer dataset mix presets.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_1b, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer
from training.metaterid_data import MIX_PRESETS, MixedTokenDataset, get_mix_sources
from training.metaterid_optim import build_optimizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("metaterid_main_train")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _memory_summary() -> str:
    if os.name != "posix":
        return "memory=unavailable"
    try:
        rss_kb = 0
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
        available_kb = 0
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    break
        cuda = ""
        if torch.cuda.is_available():
            cuda = (
                f" cuda_alloc={torch.cuda.memory_allocated() / 1024 ** 3:.2f}GiB"
                f" cuda_reserved={torch.cuda.memory_reserved() / 1024 ** 3:.2f}GiB"
            )
        return f"rss={rss_kb / 1024 ** 2:.2f}GiB mem_avail={available_kb / 1024 ** 2:.2f}GiB{cuda}"
    except Exception as exc:
        return f"memory=unavailable:{type(exc).__name__}"


def _parse_csv_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.replace("_", "")) for part in value.split(",") if part.strip()]


def _latest_checkpoint(path: Path, *, prefer_final: bool = False) -> Path | None:
    if prefer_final:
        final = path / "final.pt"
        if final.exists():
            return final
    model_only = path / "model_only.pt"
    if model_only.exists():
        return model_only
    final = path / "final.pt"
    if final.exists():
        return final
    ckpts = sorted(path.glob("tokens_*.pt"))
    return ckpts[-1] if ckpts else None


def _load_variant(name: str):
    if name == "metaterid_1b":
        return metaterid_1b()
    if name == "t4_pilot":
        return metaterid_t4_pilot()
    raise ValueError(f"Unknown variant: {name}")


def _dtype_from_args(value: str) -> torch.dtype:
    lowered = value.lower()
    if lowered == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if lowered == "bf16":
        return torch.bfloat16
    if lowered == "fp16":
        return torch.float16
    if lowered == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {value}")


def _lr_by_tokens(
    tokens_seen: int,
    *,
    warmup_tokens: int,
    target_tokens: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if warmup_tokens > 0 and tokens_seen < warmup_tokens:
        return max_lr * (tokens_seen + 1) / warmup_tokens
    if tokens_seen >= target_tokens:
        return min_lr
    denom = max(1, target_tokens - warmup_tokens)
    progress = (tokens_seen - warmup_tokens) / denom
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer,
    cfg,
    step: int,
    tokens_seen: int,
    save_optimizer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "cfg": cfg,
        "model": model.state_dict(),
    }
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(
        json.dumps({"step": step, "tokens_seen": tokens_seen}, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--ckpt-dir", default="checkpoints/metaterid_main")
    parser.add_argument("--variant", default="metaterid_1b", choices=["metaterid_1b", "t4_pilot"])
    parser.add_argument("--mix", default="final", choices=sorted(MIX_PRESETS))
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--target-tokens", type=int, default=100_000_000_000)
    parser.add_argument(
        "--additional-tokens",
        type=int,
        default=0,
        help="Train this many more tokens from the resumed checkpoint. Overrides --target-tokens after resume.",
    )
    parser.add_argument(
        "--checkpoint-tokens",
        default="1_000_000_000,5_000_000_000,10_000_000_000,30_000_000_000,50_000_000_000,100_000_000_000",
        help="Comma-separated absolute token milestones.",
    )
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-sample-chars", type=int, default=131_072)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "adamw_muon", "muon"])
    parser.add_argument(
        "--optimizer-after-switch",
        default="adamw_muon",
        choices=["adamw", "adam", "adamw_muon", "muon"],
    )
    parser.add_argument(
        "--muon-switch-ratio",
        type=float,
        default=0.20,
        help="Switch optimizer after this fraction of target tokens. Use >=1 to disable.",
    )
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-path",
        default="",
        help="Explicit checkpoint path to resume from. Overrides automatic checkpoint selection.",
    )
    parser.add_argument(
        "--resume-from-final",
        action="store_true",
        help="Prefer final.pt over model_only.pt when resuming, useful for optimizer-state resume.",
    )
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true", default=True)
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA devices.")
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    if "cuda" in device:
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = MetaTeridTokenizer(args.tokenizer)
    cfg = _load_variant(args.variant)
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = args.seq_len

    ckpt_dir = Path(args.ckpt_dir)
    start_step = 0
    tokens_seen = 0
    resume_path: Path | None = None

    base_model = MetaTeridForCausalLM(cfg).to(device)
    if args.resume:
        latest = (
            Path(args.resume_path)
            if args.resume_path
            else _latest_checkpoint(ckpt_dir, prefer_final=args.resume_from_final)
        )
        if latest is not None:
            resume_path = latest
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            base_model.load_state_dict(ckpt["model"])
            start_step = int(ckpt.get("step", 0))
            tokens_seen = int(ckpt.get("tokens_seen", 0))
            del ckpt
            gc.collect()
            if master:
                logger.info(f"Resumed model from {latest} at {tokens_seen:,} tokens")

    target_tokens = args.target_tokens
    if args.additional_tokens > 0:
        target_tokens = tokens_seen + args.additional_tokens
        if master:
            logger.info(
                f"Using additional token target: {tokens_seen:,} + "
                f"{args.additional_tokens:,} = {target_tokens:,}"
            )

    model = base_model
    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )

    switch_tokens = int(target_tokens * args.muon_switch_ratio)
    optimizer_name = (
        args.optimizer_after_switch
        if args.muon_switch_ratio < 1.0 and tokens_seen >= switch_tokens
        else args.optimizer
    )
    optimizer = build_optimizer(
        model,
        name=optimizer_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    switched = optimizer_name == args.optimizer_after_switch

    if args.resume and args.resume_optimizer:
        latest = (
            Path(args.resume_path)
            if args.resume_path
            else resume_path
            if resume_path is not None
            else _latest_checkpoint(ckpt_dir, prefer_final=True)
        )
        if latest is not None:
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                if master:
                    logger.info("Resumed optimizer state")
            elif master:
                logger.warning("Checkpoint has no optimizer state; optimizer reset")
            del ckpt
            gc.collect()

    dataset = MixedTokenDataset(
        tokenizer,
        args.seq_len,
        get_mix_sources(args.mix),
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        max_sample_chars=args.max_sample_chars,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.num_workers,
        pin_memory=("cuda" in device),
    )

    global_batch_tokens = world_size * args.micro_batch * args.grad_accum * args.seq_len
    precision_dtype = _dtype_from_args(args.precision)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=precision_dtype)
        if "cuda" in device and precision_dtype != torch.float32
        else nullcontext()
    )
    milestones = sorted(_parse_csv_ints(args.checkpoint_tokens))
    milestone_index = 0
    while milestone_index < len(milestones) and tokens_seen >= milestones[milestone_index]:
        milestone_index += 1

    if master:
        logger.info(
            f"variant={args.variant} mix={args.mix} ddp={ddp} world_size={world_size} "
            f"seq_len={args.seq_len} micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
            f"global_batch_tokens={global_batch_tokens:,} target_tokens={target_tokens:,} "
            f"optimizer={optimizer_name} precision={precision_dtype}"
        )
        if args.log_memory:
            logger.info(f"memory before training loop: {_memory_summary()}")

    model.train()
    data_iter = iter(loader)
    step = start_step
    t0 = time.perf_counter()
    last_log_step = step

    while tokens_seen < target_tokens:
        if (
            not switched
            and args.muon_switch_ratio < 1.0
            and tokens_seen >= switch_tokens
        ):
            optimizer = build_optimizer(
                model,
                name=args.optimizer_after_switch,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            switched = True
            if master:
                logger.info(
                    f"Switched optimizer to {args.optimizer_after_switch} at {tokens_seen:,} tokens"
                )

        lr = _lr_by_tokens(
            tokens_seen,
            warmup_tokens=args.warmup_tokens,
            target_tokens=target_tokens,
            max_lr=args.lr,
            min_lr=args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(args.grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            n_loops = random.randint(cfg.train_min_loops, cfg.train_max_loops)
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < args.grad_accum - 1
                else nullcontext()
            )
            with sync_ctx, amp_ctx:
                logits = model(x, n_loops=n_loops)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                loss = loss / args.grad_accum
            loss.backward()
            loss_accum += float(loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        tokens_seen += global_batch_tokens

        if master and step % args.log_every == 0:
            dt = time.perf_counter() - t0
            steps_delta = max(1, step - last_log_step)
            tok_per_sec = global_batch_tokens * steps_delta / max(dt, 1e-6)
            logger.info(
                f"step={step:,} tokens={tokens_seen:,}/{target_tokens:,} "
                f"loss={loss_accum:.4f} grad_norm={float(grad_norm):.2f} "
                f"lr={lr:.2e} optimizer={'post_switch' if switched else 'pre_switch'} "
                f"tok/s={tok_per_sec:,.0f}"
            )
            if args.log_memory:
                logger.info(f"memory: {_memory_summary()}")
            t0 = time.perf_counter()
            last_log_step = step

        while milestone_index < len(milestones) and tokens_seen >= milestones[milestone_index]:
            milestone = milestones[milestone_index]
            if master:
                path = ckpt_dir / f"tokens_{milestone:013d}.pt"
                _save_checkpoint(
                    path,
                    model=base_model,
                    optimizer=optimizer,
                    cfg=cfg,
                    step=step,
                    tokens_seen=tokens_seen,
                    save_optimizer=args.save_optimizer,
                )
                logger.info(f"Saved milestone checkpoint {path}")
            if ddp:
                dist.barrier()
            milestone_index += 1

        if args.save_every_steps > 0 and step % args.save_every_steps == 0 and master:
            _save_checkpoint(
                ckpt_dir / "latest.pt",
                model=base_model,
                optimizer=optimizer,
                cfg=cfg,
                step=step,
                tokens_seen=tokens_seen,
                save_optimizer=args.save_optimizer,
            )

    if master:
        _save_checkpoint(
            ckpt_dir / "final.pt",
            model=base_model,
            optimizer=optimizer,
            cfg=cfg,
            step=step,
            tokens_seen=tokens_seen,
            save_optimizer=args.save_optimizer,
        )
        _save_checkpoint(
            ckpt_dir / "model_only.pt",
            model=base_model,
            optimizer=optimizer,
            cfg=cfg,
            step=step,
            tokens_seen=tokens_seen,
            save_optimizer=False,
        )
        logger.info(f"Training complete. Final checkpoint: {ckpt_dir / 'final.pt'}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
