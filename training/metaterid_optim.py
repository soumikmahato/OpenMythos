from __future__ import annotations

import torch


def build_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Build the optimizer for MetaTerid training.

    Muon is part of the project plan, but this repository does not yet carry a
    validated Muon implementation. The pilot defaults to AdamW and can fail
    explicitly if Muon is requested before implementation.
    """
    lowered = name.lower()
    if lowered in {"adam", "adamw"}:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )
    if lowered in {"muon", "moonlight"}:
        raise NotImplementedError(
            "Muon is planned for MetaTerid but is not implemented in this repo yet. "
            "Use --optimizer adamw for the T4 pilot."
        )
    raise ValueError(f"Unknown optimizer: {name}")
