from __future__ import annotations

import torch


def _zeropower_via_newton_schulz(update: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize a 2D update matrix with the Newton-Schulz iteration used by
    Muon-style optimizers. This is intentionally conservative and only applied
    to matrix-shaped parameters selected for Muon.
    """
    if update.ndim != 2:
        return update

    orig_dtype = update.dtype
    x = update.float()
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """
    Minimal Muon-style optimizer for matrix parameters.

    Use through `build_optimizer(..., name="adamw_muon")` so embeddings, norms,
    routers, heads, attention-sensitive projections, and scalar/bias parameters
    stay on AdamW. This implementation is deliberately small and explicit; it is
    meant for MetaTerid experiments, not as a drop-in replacement for mature
    distributed Muon implementations.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                grad = param.grad
                if grad.ndim != 2:
                    param.add_(grad, alpha=-lr)
                    continue

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum)
                update = _zeropower_via_newton_schulz(update, steps=ns_steps)
                scale = max(1.0, (param.shape[0] / max(1, param.shape[1])) ** 0.5)
                param.add_(update, alpha=-lr * scale)
        return loss


class OptimizerList:
    """Small wrapper so hybrid optimizers behave like one optimizer."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        states = state_dict.get("optimizers")
        if states is None:
            raise ValueError("Hybrid optimizer checkpoint missing 'optimizers' key")
        if len(states) != len(self.optimizers):
            raise ValueError("Hybrid optimizer state count does not match optimizer count")
        for optimizer, state in zip(self.optimizers, states):
            optimizer.load_state_dict(state)


def _is_adam_sensitive_param(name: str, param: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    if param.ndim < 2:
        return True
    sensitive_terms = (
        "embed",
        "head",
        "norm",
        "router",
        "gate",
        "act",
        "halt",
        "injection",
        "attn_res",
        "block_attn_res",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    return any(term in lowered for term in sensitive_terms)


def build_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Build the optimizer for MetaTerid training.

    AdamW is the quality-safe default. `adamw_muon` uses AdamW for sensitive
    parameters and Muon for eligible matrix weights, matching the MetaTerid
    plan without forcing Muon onto numerically sensitive tensors.
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
    if lowered in {"muon"}:
        params = [param for param in model.parameters() if param.requires_grad and param.ndim == 2]
        if not params:
            raise ValueError("No eligible 2D parameters found for Muon")
        return Muon(params, lr=lr, weight_decay=weight_decay)
    if lowered in {"adamw_muon", "moonlight"}:
        adam_params = []
        muon_params = []
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_adam_sensitive_param(param_name, param):
                adam_params.append(param)
            else:
                muon_params.append(param)

        optimizers: list[torch.optim.Optimizer] = []
        if adam_params:
            optimizers.append(
                torch.optim.AdamW(
                    adam_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95),
                    fused=torch.cuda.is_available(),
                )
            )
        if muon_params:
            optimizers.append(Muon(muon_params, lr=lr, weight_decay=weight_decay))
        if not optimizers:
            raise ValueError("No trainable parameters found")
        return OptimizerList(optimizers)
    if lowered in {"muon_full"}:
        return Muon(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")
