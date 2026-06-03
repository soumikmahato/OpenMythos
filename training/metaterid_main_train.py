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
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_1b, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer
from training.metaterid_data import (
    MIX_PRESETS,
    MMapTokenDataset,
    MixedTokenDataset,
    get_mix_sources,
)
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


def _parse_loop_buckets(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    if "-" in value and "," not in value:
        start_s, end_s = value.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start > end:
            raise ValueError("--loop-buckets start cannot exceed end")
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def _collect_moe_aux_loss(model: torch.nn.Module) -> torch.Tensor | None:
    losses = []
    for module in model.modules():
        if not hasattr(module, "impl"):
            continue
        aux = getattr(module, "last_aux_loss", None)
        if aux is not None:
            losses.append(aux)
    if not losses:
        return None
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total


@torch.no_grad()
def _update_moe_router_biases(model: torch.nn.Module) -> None:
    for module in model.modules():
        if getattr(module, "impl", None) != "packed":
            continue
        update = getattr(module, "update_router_bias", None)
        if not callable(update):
            continue
        counts = getattr(getattr(module, "router", None), "last_metrics", {}).get(
            "tokens_per_expert"
        )
        if counts is not None and dist.is_available() and dist.is_initialized():
            counts = counts.detach().to(next(module.parameters()).device).float()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            update(counts)
        else:
            update()


def _moe_metrics_summary(model: torch.nn.Module) -> str:
    parts = []
    for module in model.modules():
        if not hasattr(module, "impl"):
            continue
        metrics = getattr(module, "last_metrics", {})
        if not metrics:
            continue
        entropy = metrics.get("router_entropy")
        load_max = metrics.get("expert_load_max")
        load_mean = metrics.get("expert_load_mean")
        if entropy is not None and load_max is not None and load_mean is not None:
            parts.append(
                f"router_entropy={float(entropy):.2f} "
                f"expert_load_max={float(load_max):.0f} "
                f"expert_load_mean={float(load_mean):.1f}"
            )
    return " ".join(parts)


def _compile_callable(fn: Callable[[torch.Tensor], torch.Tensor], mode: str):
    kwargs = {"dynamic": False}
    if mode != "default":
        kwargs["mode"] = mode
    return torch.compile(fn, **kwargs)


def _build_loop_bucket_fns(model: torch.nn.Module, buckets: list[int], compile_mode: str):
    hidden_fns: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    logits_fns: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for loops in buckets:
        def hidden_fn(input_ids, *, _loops=loops):
            return model(input_ids, n_loops=_loops, return_hidden=True)

        def logits_fn(input_ids, *, _loops=loops):
            return model(input_ids, n_loops=_loops)

        hidden_fns[loops] = _compile_callable(hidden_fn, compile_mode)
        logits_fns[loops] = _compile_callable(logits_fn, compile_mode)
    return hidden_fns, logits_fns


def _autotune_micro_batch(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    cfg,
    device: str,
    precision_dtype: torch.dtype,
) -> int:
    if "cuda" not in device:
        return args.micro_batch
    candidates = []
    value = max(1, args.micro_batch)
    while value >= 1:
        candidates.append(value)
        value //= 2
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=precision_dtype)
        if precision_dtype != torch.float32
        else nullcontext()
    )
    for candidate in candidates:
        try:
            model.zero_grad(set_to_none=True)
            x = torch.randint(0, cfg.vocab_size, (candidate, args.seq_len), device=device)
            y = torch.randint(0, cfg.vocab_size, (candidate, args.seq_len), device=device)
            with amp_ctx:
                loss = _language_model_loss(
                    model=model,
                    lm_head=base_model.head,
                    input_ids=x,
                    labels=y,
                    vocab_size=cfg.vocab_size,
                    n_loops=cfg.train_min_loops,
                    loss_chunk_tokens=args.loss_chunk_tokens,
                )
            loss.backward()
            model.zero_grad(set_to_none=True)
            del x, y, loss
            torch.cuda.empty_cache()
            return candidate
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    return 1


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


def _param_dtype_from_args(value: str) -> torch.dtype | None:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return None


def _convert_packed_moe_params_(model: torch.nn.Module, dtype: torch.dtype) -> None:
    for module in model.modules():
        if getattr(module, "impl", None) != "packed":
            continue
        for name in ("routed_gate_weight", "routed_up_weight", "routed_down_weight"):
            param = getattr(module, name, None)
            if param is not None:
                param.data = param.data.to(dtype)
        for name in ("shared_gate", "shared_up", "shared_down"):
            layer = getattr(module, name, None)
            if layer is not None:
                layer.to(dtype)


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


def _language_model_loss(
    *,
    model,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    n_loops: int,
    loss_chunk_tokens: int,
    hidden_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    if loss_chunk_tokens <= 0:
        logits = logits_fn(input_ids) if logits_fn is not None else model(input_ids, n_loops=n_loops)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        aux_loss = _collect_moe_aux_loss(model)
        return loss if aux_loss is None else loss + aux_loss

    hidden = (
        hidden_fn(input_ids)
        if hidden_fn is not None
        else model(input_ids, n_loops=n_loops, return_hidden=True)
    )
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    total_loss = hidden.new_zeros(())
    total_tokens = labels.numel()

    for start in range(0, total_tokens, loss_chunk_tokens):
        end = min(start + loss_chunk_tokens, total_tokens)
        logits = lm_head(hidden[start:end])
        total_loss = total_loss + F.cross_entropy(
            logits, labels[start:end], reduction="sum"
        )
    loss = total_loss / total_tokens
    aux_loss = _collect_moe_aux_loss(model)
    return loss if aux_loss is None else loss + aux_loss


def _backward_language_model_loss(
    *,
    model,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    n_loops: int,
    loss_chunk_tokens: int,
    loss_scale: float,
    hidden_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    if loss_chunk_tokens <= 0:
        logits = logits_fn(input_ids) if logits_fn is not None else model(input_ids, n_loops=n_loops)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        aux_loss = _collect_moe_aux_loss(model)
        if aux_loss is not None:
            loss = loss + aux_loss
        scaled_loss = loss * loss_scale
        scaled_loss.backward()
        return float(scaled_loss.detach())

    hidden = (
        hidden_fn(input_ids)
        if hidden_fn is not None
        else model(input_ids, n_loops=n_loops, return_hidden=True)
    )
    hidden_shape = hidden.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    total_tokens = labels.numel()
    hidden_grad = torch.zeros_like(hidden)
    head_params = [p for p in lm_head.parameters() if p.requires_grad]
    loss_accum = 0.0

    for start in range(0, total_tokens, loss_chunk_tokens):
        end = min(start + loss_chunk_tokens, total_tokens)
        hidden_chunk = hidden[start:end]
        logits = lm_head(hidden_chunk)
        chunk_loss_sum = F.cross_entropy(
            logits, labels[start:end], reduction="sum"
        )
        scaled_chunk_loss = chunk_loss_sum * (loss_scale / total_tokens)
        grads = torch.autograd.grad(
            scaled_chunk_loss,
            (hidden_chunk, *head_params),
            retain_graph=True,
            allow_unused=False,
        )
        hidden_grad[start:end].add_(grads[0])
        for param, grad in zip(head_params, grads[1:]):
            grad = grad.detach()
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.add_(grad)
        loss_accum += float(scaled_chunk_loss.detach())

    aux_loss = _collect_moe_aux_loss(model)
    if aux_loss is not None:
        scaled_aux_loss = aux_loss * loss_scale
        scaled_aux_loss.backward(retain_graph=True)
        loss_accum += float(scaled_aux_loss.detach())

    hidden.backward(hidden_grad.reshape(hidden_shape))
    return loss_accum


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
    parser.add_argument("--data-backend", default="streaming", choices=["streaming", "mmap"])
    parser.add_argument("--mmap-dir", default="")
    parser.add_argument(
        "--loss-chunk-tokens",
        type=int,
        default=8192,
        help=(
            "If >0, compute the tied LM head and cross-entropy in chunks of this "
            "many flattened tokens. Reduces peak memory for large vocabularies."
        ),
    )
    parser.add_argument(
        "--train-min-loops",
        type=int,
        default=0,
        help="Override cfg.train_min_loops. Use for speed/memory smoke tests.",
    )
    parser.add_argument(
        "--train-max-loops",
        type=int,
        default=0,
        help="Override cfg.train_max_loops. Use for speed/memory smoke tests.",
    )
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
    parser.add_argument(
        "--model-param-dtype",
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help=(
            "Convert model parameters before training. bf16 enables the "
            "torch grouped_mm MoE backend on H100; fp32 keeps the mixed-precision default."
        ),
    )
    parser.add_argument(
        "--moe-param-dtype",
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help=(
            "Convert only packed MoE expert/shared weights before training. "
            "bf16 enables grouped_mm while keeping router/norm/embed/head params in fp32."
        ),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--compile-loop-buckets", action="store_true")
    parser.add_argument("--loop-buckets", default="4-16")
    parser.add_argument("--loop-schedule", default="random", choices=["random", "bucketed"])
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--moe-pad-for-cuda-graphs", action="store_true")
    parser.add_argument("--moe-graph-capacity-factor", type=float, default=1.25)
    parser.add_argument("--moe-static-expert-capacity", type=int, default=0)
    parser.add_argument("--moe-impl", default="", choices=["", "legacy", "packed"])
    parser.add_argument(
        "--moe-backend",
        default="",
        choices=["", "auto", "grouped_mm", "padded", "sorted"],
    )
    parser.add_argument("--auto-micro-batch", action="store_true")
    parser.add_argument("--router-score-function", default="", choices=["", "softmax", "sigmoid"])
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
    parser.add_argument("--find-unused-parameters", action="store_true", default=None)
    parser.add_argument("--no-find-unused-parameters", dest="find_unused_parameters", action="store_false")
    parser.add_argument("--ddp-static-graph", action="store_true")
    parser.add_argument("--ddp-gradient-as-bucket-view", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wants_compile = args.compile or args.compile_loop_buckets
    if wants_compile and args.moe_pad_for_cuda_graphs and args.moe_static_expert_capacity <= 0:
        raise ValueError(
            "Compiled graph-safe MoE padding requires "
            "--moe-static-expert-capacity. Run a warmup benchmark to choose it."
        )

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
    if args.train_min_loops > 0:
        cfg.train_min_loops = args.train_min_loops
    if args.train_max_loops > 0:
        cfg.train_max_loops = args.train_max_loops
    if cfg.train_min_loops > cfg.train_max_loops:
        raise ValueError("--train-min-loops cannot exceed --train-max-loops")
    if args.router_score_function:
        cfg.router_score_function = args.router_score_function
    if args.moe_impl:
        cfg.moe_impl = args.moe_impl
    if args.moe_backend:
        cfg.moe_backend = args.moe_backend
    if args.moe_pad_for_cuda_graphs:
        cfg.moe_pad_for_cuda_graphs = True
    cfg.moe_graph_capacity_factor = args.moe_graph_capacity_factor
    cfg.moe_static_expert_capacity = args.moe_static_expert_capacity
    if args.find_unused_parameters is None:
        args.find_unused_parameters = getattr(cfg, "moe_impl", "legacy") != "packed"
    if args.cuda_graphs and master:
        logger.warning(
            "--cuda-graphs currently enables graph-safe MoE padding and static loop "
            "buckets, but full CUDA graph replay of the train step is not captured yet."
        )
    precision_dtype = _dtype_from_args(args.precision)

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

    model_dtype = _param_dtype_from_args(args.model_param_dtype)
    if model_dtype is not None:
        base_model = base_model.to(model_dtype)
    moe_dtype = _param_dtype_from_args(args.moe_param_dtype)
    if moe_dtype is not None:
        _convert_packed_moe_params_(base_model, moe_dtype)

    target_tokens = args.target_tokens
    if args.additional_tokens > 0:
        target_tokens = tokens_seen + args.additional_tokens
        if master:
            logger.info(
                f"Using additional token target: {tokens_seen:,} + "
                f"{args.additional_tokens:,} = {target_tokens:,}"
            )

    if args.auto_micro_batch:
        tuned = _autotune_micro_batch(
            args=args,
            model=base_model,
            base_model=base_model,
            cfg=cfg,
            device=device,
            precision_dtype=precision_dtype,
        )
        if master:
            logger.info(f"auto_micro_batch selected micro_batch={tuned}")
        args.micro_batch = tuned

    model = base_model
    if args.compile and not args.compile_loop_buckets:
        model = _compile_callable(model, args.compile_mode)

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
            static_graph=args.ddp_static_graph,
            gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        )

    loop_hidden_fns = {}
    loop_logits_fns = {}
    loop_buckets = _parse_loop_buckets(args.loop_buckets)
    if args.compile_loop_buckets:
        loop_hidden_fns, loop_logits_fns = _build_loop_bucket_fns(
            model, loop_buckets, args.compile_mode
        )
        if master:
            logger.info(f"Compiled loop buckets: {loop_buckets} mode={args.compile_mode}")

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

    if args.data_backend == "mmap":
        if not args.mmap_dir:
            raise ValueError("--mmap-dir is required when --data-backend=mmap")
        dataset = MMapTokenDataset(
            args.mmap_dir,
            args.seq_len,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
    else:
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
        persistent_workers=args.num_workers > 0,
    )

    global_batch_tokens = world_size * args.micro_batch * args.grad_accum * args.seq_len
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
            f"data_backend={args.data_backend} moe_impl={getattr(cfg, 'moe_impl', 'legacy')} "
            f"moe_backend={getattr(cfg, 'moe_backend', 'auto')} "
            f"seq_len={args.seq_len} micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
            f"global_batch_tokens={global_batch_tokens:,} target_tokens={target_tokens:,} "
            f"optimizer={optimizer_name} precision={precision_dtype} "
            f"model_param_dtype={args.model_param_dtype} moe_param_dtype={args.moe_param_dtype} "
            f"find_unused_parameters={args.find_unused_parameters}"
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
            if args.loop_schedule == "bucketed" and loop_buckets:
                n_loops = loop_buckets[(step * args.grad_accum + micro_step) % len(loop_buckets)]
            else:
                n_loops = random.randint(cfg.train_min_loops, cfg.train_max_loops)
            hidden_fn = loop_hidden_fns.get(n_loops)
            logits_fn = loop_logits_fns.get(n_loops)
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < args.grad_accum - 1
                else nullcontext()
            )
            with sync_ctx, amp_ctx:
                loss_value = _backward_language_model_loss(
                    model=model,
                    lm_head=base_model.head,
                    input_ids=x,
                    labels=y,
                    vocab_size=cfg.vocab_size,
                    n_loops=n_loops,
                    loss_chunk_tokens=args.loss_chunk_tokens,
                    loss_scale=1.0 / args.grad_accum,
                    hidden_fn=hidden_fn,
                    logits_fn=logits_fn,
                )
            loss_accum += loss_value

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _update_moe_router_biases(model)
        step += 1
        tokens_seen += global_batch_tokens

        if master and step % args.log_every == 0:
            dt = time.perf_counter() - t0
            steps_delta = max(1, step - last_log_step)
            tok_per_sec = global_batch_tokens * steps_delta / max(dt, 1e-6)
            moe_metrics = _moe_metrics_summary(model)
            logger.info(
                f"step={step:,} tokens={tokens_seen:,}/{target_tokens:,} "
                f"loss={loss_accum:.4f} grad_norm={float(grad_norm):.2f} "
                f"lr={lr:.2e} optimizer={'post_switch' if switched else 'pre_switch'} "
                f"tok/s={tok_per_sec:,.0f}"
                + (f" {moe_metrics}" if moe_metrics else "")
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
