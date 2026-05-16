from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.main import RMSNorm


class BlockAttentionResidual(nn.Module):
    """
    Content-dependent residual aggregation over block/source outputs.

    The Attention Residuals paper distinguishes Full AttnRes from Block
    AttnRes. This module implements the practical block-level primitive: the
    caller provides a small list of previous block outputs, and the module
    learns to attend over those sources for each token before adding a gated
    residual update to the current hidden state.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_heads: int = 4,
        max_sources: int = 8,
        dropout: float = 0.0,
        gate_init: float = 0.05,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if max_sources < 1:
            raise ValueError("max_sources must be at least 1")

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_sources = max_sources

        self.current_norm = RMSNorm(dim)
        self.source_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self, current: torch.Tensor, sources: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Args:
            current: Tensor of shape `(B, T, D)`.
            sources: Prior block outputs, each shape `(B, T, D)`.

        Returns:
            Tensor of shape `(B, T, D)`.
        """
        if not sources:
            return current

        selected = list(sources[-self.max_sources :])
        B, T, D = current.shape
        src = torch.stack(selected, dim=2)

        q = self.q_proj(self.current_norm(current)).view(
            B, T, self.n_heads, self.head_dim
        )
        k = self.k_proj(self.source_norm(src)).view(
            B, T, len(selected), self.n_heads, self.head_dim
        )
        v = self.v_proj(src).view(B, T, len(selected), self.n_heads, self.head_dim)

        scores = torch.einsum("bthd,btshd->bths", q, k) * (self.head_dim**-0.5)
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = self.dropout(weights)

        out = torch.einsum("bths,btshd->bthd", weights, v)
        out = out.contiguous().view(B, T, D)
        out = self.o_proj(out)
        return current + self.gate.to(current.dtype) * out
