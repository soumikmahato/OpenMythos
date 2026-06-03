#!/usr/bin/env python3
"""
Synthetic H100 throughput benchmark for MetaTerid training.

This isolates model/MoE/loss/optimizer throughput from dataset/tokenizer cost.
Use the main training script with --data-backend=mmap for full pipeline runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_1b, metaterid_t4_pilot
from training.metaterid_optim import build_optimizer


def load_variant(name: str):
    if name == "metaterid_1b":
        return metaterid_1b()
    if name == "t4_pilot":
        return metaterid_t4_pilot()
    raise ValueError(f"Unknown variant: {name}")


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def param_dtype_from_arg(value: str) -> torch.dtype | None:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return None


def convert_packed_moe_params_(model: torch.nn.Module, dtype: torch.dtype) -> None:
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


def prepare_cuda_graph_modules_(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "record_metrics"):
            module.record_metrics = False
        if hasattr(module, "graph_safe_counts"):
            module.graph_safe_counts = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="metaterid_1b", choices=["metaterid_1b", "t4_pilot"])
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--loss-chunk-tokens", type=int, default=8192)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--model-param-dtype",
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help=(
            "Optional parameter dtype conversion before benchmarking. "
            "Use bf16 to enable torch grouped_mm MoE on H100."
        ),
    )
    parser.add_argument(
        "--moe-param-dtype",
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help=(
            "Convert only packed MoE expert/shared weights. bf16 enables "
            "grouped_mm while keeping router/norm/embed/head params in fp32."
        ),
    )
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "adamw_muon", "muon"])
    parser.add_argument("--moe-impl", default="packed", choices=["legacy", "packed"])
    parser.add_argument(
        "--moe-backend",
        default="auto",
        choices=["auto", "grouped_mm", "padded", "sorted"],
    )
    parser.add_argument("--moe-pad-for-cuda-graphs", action="store_true")
    parser.add_argument("--moe-graph-capacity-factor", type=float, default=1.25)
    parser.add_argument("--moe-static-expert-capacity", type=int, default=0)
    parser.add_argument("--router-score-function", default="sigmoid", choices=["softmax", "sigmoid"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help=(
            "Capture forward + chunked loss + backward for the static synthetic "
            "benchmark. Optimizer step and grad clipping stay outside the graph."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def chunked_loss(model, head, x, y, vocab_size: int, loops: int, loss_chunk_tokens: int):
    if loss_chunk_tokens <= 0:
        logits = model(x, n_loops=loops)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        return loss + moe_aux_loss(model)

    hidden = model(x, n_loops=loops, return_hidden=True)
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = y.reshape(-1)
    total = hidden.new_zeros(())
    for start in range(0, labels.numel(), loss_chunk_tokens):
        end = min(start + loss_chunk_tokens, labels.numel())
        logits = head(hidden[start:end])
        total = total + F.cross_entropy(logits, labels[start:end], reduction="sum")
    return total / labels.numel() + moe_aux_loss(model)


def moe_aux_loss(model) -> torch.Tensor:
    total = None
    for module in model.modules():
        if not hasattr(module, "impl"):
            continue
        aux = getattr(module, "last_aux_loss", None)
        if aux is not None:
            total = aux if total is None else total + aux
    if total is None:
        return next(model.parameters()).new_zeros(())
    return total


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark is intended for CUDA/H100 runs.")
    if args.compile and args.moe_pad_for_cuda_graphs and args.moe_static_expert_capacity <= 0:
        raise ValueError(
            "--compile with --moe-pad-for-cuda-graphs requires "
            "--moe-static-expert-capacity. Run a warmup benchmark to choose it."
        )
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    precision = dtype_from_arg(args.precision)

    cfg = load_variant(args.variant)
    cfg.vocab_size = args.vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.moe_impl = args.moe_impl
    cfg.moe_backend = args.moe_backend
    cfg.moe_pad_for_cuda_graphs = args.moe_pad_for_cuda_graphs
    cfg.moe_graph_capacity_factor = args.moe_graph_capacity_factor
    cfg.moe_static_expert_capacity = args.moe_static_expert_capacity
    cfg.router_score_function = args.router_score_function

    model = MetaTeridForCausalLM(cfg).to(device)
    model_dtype = param_dtype_from_arg(args.model_param_dtype)
    if model_dtype is not None:
        model = model.to(model_dtype)
    moe_dtype = param_dtype_from_arg(args.moe_param_dtype)
    if moe_dtype is not None:
        convert_packed_moe_params_(model, moe_dtype)
    if args.cuda_graphs:
        prepare_cuda_graph_modules_(model)
    if args.compile:
        compile_kwargs = {"dynamic": False}
        if args.compile_mode != "default":
            compile_kwargs["mode"] = args.compile_mode
        model = torch.compile(model, **compile_kwargs)
    optimizer = build_optimizer(model, name=args.optimizer, lr=3e-4, weight_decay=0.1)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=precision)
        if precision != torch.float32
        else nullcontext()
    )

    x = torch.randint(
        0, cfg.vocab_size, (args.micro_batch, args.seq_len), device=device
    )
    y = torch.randint(
        0, cfg.vocab_size, (args.micro_batch, args.seq_len), device=device
    )
    timings = []

    def backward_pass() -> torch.Tensor:
        with amp_ctx:
            loss = chunked_loss(
                model,
                model.head,
                x,
                y,
                cfg.vocab_size,
                args.loops,
                args.loss_chunk_tokens,
            )
        loss.backward()
        return loss

    if args.cuda_graphs:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(args.warmup_steps):
                optimizer.zero_grad(set_to_none=False)
                backward_pass()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=False)
        torch.cuda.current_stream().wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            backward_pass()
        torch.cuda.synchronize()

        for _ in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=False)
            graph.replay()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    else:
        total_steps = args.warmup_steps + args.steps
        for step in range(total_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            backward_pass()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            if step >= args.warmup_steps:
                timings.append(time.perf_counter() - t0)

    tokens = args.micro_batch * args.seq_len
    mean_step = sum(timings) / len(timings)
    result = {
        "variant": args.variant,
        "moe_impl": args.moe_impl,
        "moe_backend": args.moe_backend,
        "router_score_function": args.router_score_function,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "loops": args.loops,
        "precision": str(precision),
        "model_param_dtype": args.model_param_dtype,
        "moe_param_dtype": args.moe_param_dtype,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "cuda_graphs": args.cuda_graphs,
        "mean_step_s": mean_step,
        "tok_per_s": tokens / mean_step,
        "cuda_max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
