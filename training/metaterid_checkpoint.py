from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer


def dtype_from_arg(value: str) -> torch.dtype | None:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if value in {"auto", "checkpoint", "none", ""}:
        return None
    raise ValueError(f"Unsupported dtype argument: {value}")


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


def load_metaterid_model(
    *,
    checkpoint_path: str | Path,
    tokenizer: MetaTeridTokenizer,
    device: str | torch.device,
    seq_len: int | None = None,
    model_param_dtype: str = "checkpoint",
    moe_param_dtype: str = "auto",
    moe_backend: str = "checkpoint",
) -> tuple[MetaTeridForCausalLM, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg") or metaterid_t4_pilot()
    cfg.vocab_size = tokenizer.vocab_size
    if seq_len is not None:
        cfg.max_seq_len = seq_len

    requested_backend = moe_backend.strip().lower()
    if requested_backend not in {"checkpoint", "auto", "grouped_mm", "padded", "sorted"}:
        raise ValueError(f"Unsupported --moe-backend: {moe_backend}")
    if requested_backend != "checkpoint":
        cfg.moe_backend = requested_backend

    device = torch.device(device)
    if device.type != "cuda" and getattr(cfg, "moe_backend", "auto") == "grouped_mm":
        cfg.moe_backend = "padded"

    model = MetaTeridForCausalLM(cfg)
    model.load_state_dict(checkpoint["model"])

    model_dtype = dtype_from_arg(model_param_dtype)
    if model_dtype is not None:
        model = model.to(model_dtype)

    moe_dtype = dtype_from_arg(moe_param_dtype)
    if moe_dtype is None and device.type == "cuda":
        backend = getattr(cfg, "moe_backend", "auto")
        if backend in {"grouped_mm", "auto"}:
            moe_dtype = torch.bfloat16
    if moe_dtype is not None:
        convert_packed_moe_params_(model, moe_dtype)

    model = model.to(device)
    model.eval()
    return model, checkpoint
