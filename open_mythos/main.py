import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def _is_cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    moe_impl: str = "legacy"  # "legacy" | "packed"
    moe_backend: str = "auto"  # "auto" | "grouped_mm" | "padded" | "sorted"
    moe_dispatcher: str = "local_packed"
    router_score_function: str = "softmax"  # "softmax" | "sigmoid"
    normalize_topk: bool = True
    enable_router_bias: bool = True
    router_bias_update_rate: float = 1e-3
    route_scale: float = 1.0
    seq_aux_loss_coeff: float = 0.0
    moe_pad_for_cuda_graphs: bool = False
    moe_graph_capacity_factor: float = 1.25
    moe_static_expert_capacity: int = 0
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_rope_cos_sin(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute real-valued RoPE cos/sin pairs for compiler-friendly training.

    The public precompute_rope_freqs API keeps returning complex phasors for
    compatibility with older tests and utilities. Model forward passes use this
    real layout to avoid complex operators in torch.compile/Inductor graphs.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed complex frequencies of shape (T, head_dim//2)
                     or real cos/sin pairs of shape (T, head_dim//2, 2),
                     already sliced to exactly the positions being processed
                     (caller is responsible for correct start_pos offset)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    seq_len = x.shape[1]
    if freqs_cis.size(0) < seq_len:
        raise RuntimeError(
            f"RoPE frequency table length {freqs_cis.size(0)} is shorter than sequence length {seq_len}"
        )
    if freqs_cis.size(0) != seq_len:
        freqs_cis = freqs_cis[:seq_len]
    x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
    if freqs_cis.is_complex():
        cos = freqs_cis.real
        sin = freqs_cis.imag
    else:
        cos = freqs_cis[..., 0]
        sin = freqs_cis[..., 1]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    x_even = x_pair[..., 0]
    x_odd = x_pair[..., 1]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    )
    return rotated.flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-bound-optimal. Inputs are cast to
    bfloat16 for flash_attn and restored to the original dtype afterward.
    Falls back to manual scaled dot-product attention when flash-attn is absent.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor | bool] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        if _HAS_FLASH_ATTN:
            # flash_attn_func expects (B, T, H, head_dim) — GQA is handled natively
            # (n_kv_heads < n_heads is supported without repeat_interleave).
            # causal=True when mask is present (full-sequence prefill/training);
            # causal=False for single-token decode where T=1 and mask is None.
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=bool(mask))
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Fallback: PyTorch SDPA with explicit KV head expansion. On H100,
            # this routes training prefill shapes to Flash-SDPA when available.
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            dropout_p = self.dropout_p if self.training else 0.0
            use_causal = mask is True or (mask is not None and q.size(-2) == k.size(-2))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if use_causal or mask is True else mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor | bool] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        dropout_p = self.attn_drop.p if self.training else 0.0
        use_causal = mask is True or (mask is not None and q.size(-2) == k.size(-2))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if use_causal or mask is True else mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MetaTeridMoERouter(nn.Module):
    """
    DeepSeek-V3-style top-k router for the packed MetaTerid MoE path.

    The optional expert bias affects only the routing decision. Gate weights are
    gathered from unbiased scores, preserving the aux-loss-free balancing shape.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int,
        topk: int,
        *,
        score_function: str = "sigmoid",
        normalize_topk: bool = True,
        route_scale: float = 1.0,
        enable_router_bias: bool = True,
        seq_aux_loss_coeff: float = 0.0,
        router_bias_update_rate: float = 1e-3,
    ):
        super().__init__()
        if score_function not in {"softmax", "sigmoid"}:
            raise ValueError("router_score_function must be 'softmax' or 'sigmoid'")
        self.n_experts = n_experts
        self.topk = topk
        self.score_function = score_function
        self.normalize_topk = normalize_topk
        self.route_scale = route_scale
        self.enable_router_bias = enable_router_bias
        self.seq_aux_loss_coeff = seq_aux_loss_coeff
        self.router_bias_update_rate = router_bias_update_rate
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("router_bias", torch.zeros(n_experts))
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.record_metrics = True

    def forward(
        self,
        x: torch.Tensor,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        logits = F.linear(x.float(), self.weight.float())
        if self.score_function == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = logits.sigmoid().to(torch.float32)

        routing_scores = scores
        if self.enable_router_bias:
            routing_scores = routing_scores + self.router_bias.float()

        _, topk_idx = routing_scores.topk(self.topk, dim=-1)
        topk_weight = scores.gather(-1, topk_idx)
        if self.normalize_topk:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp(
                min=1e-9
            )
        topk_weight = (topk_weight * self.route_scale).to(x.dtype)

        aux_loss = None
        if self.training and self.seq_aux_loss_coeff > 0.0:
            aux_loss = self._sequence_aux_loss(scores, topk_idx, batch_size, seq_len)
        self.last_aux_loss = aux_loss
        self._record_metrics(scores, topk_idx)
        return topk_weight, topk_idx, aux_loss

    def _sequence_aux_loss(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        batch_size: Optional[int],
        seq_len: Optional[int],
    ) -> torch.Tensor:
        if batch_size is None or seq_len is None or batch_size * seq_len != scores.size(0):
            selected = F.one_hot(topk_idx, self.n_experts).float().mean(dim=(0, 1))
            prob = scores.mean(dim=0)
            return self.seq_aux_loss_coeff * self.n_experts * (selected * prob).sum()

        selected = F.one_hot(topk_idx, self.n_experts).float().mean(dim=1)
        selected = selected.view(batch_size, seq_len, self.n_experts).mean(dim=1)
        prob = scores.view(batch_size, seq_len, self.n_experts).mean(dim=1)
        return (
            self.seq_aux_loss_coeff
            * self.n_experts
            * (selected * prob).sum(dim=1).mean()
        )

    def _record_metrics(self, scores: torch.Tensor, topk_idx: torch.Tensor) -> None:
        if not self.record_metrics or (scores.is_cuda and _is_cuda_graph_capturing()):
            return
        with torch.no_grad():
            counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_experts)
            probs = scores.mean(dim=0)
            probs = probs / probs.sum().clamp(min=1e-9)
            entropy = -(probs * probs.clamp(min=1e-9).log()).sum()
            self.last_metrics = {
                "tokens_per_expert": counts.detach(),
                "router_entropy": entropy.detach(),
                "expert_load_max": counts.max().detach(),
                "expert_load_mean": counts.float().mean().detach(),
            }

    @torch.no_grad()
    def update_router_bias(self, counts: Optional[torch.Tensor] = None) -> None:
        if not self.enable_router_bias or not self.last_metrics:
            return
        if counts is None:
            counts = self.last_metrics["tokens_per_expert"].float()
        else:
            counts = counts.float()
        if counts.numel() == 0 or counts.sum() <= 0:
            return
        target = counts.mean()
        direction = torch.sign(target - counts).to(self.router_bias.device)
        self.router_bias.add_(self.router_bias_update_rate * direction)


class MetaTeridMoEDispatcher(nn.Module):
    """
    Local packed dispatcher: route -> permute by expert -> grouped experts ->
    unpermute/combine. The padded backend uses batched GEMMs across experts.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.topk = cfg.n_experts_per_tok
        self.dim = cfg.dim
        self.backend = cfg.moe_backend
        self.pad_for_cuda_graphs = cfg.moe_pad_for_cuda_graphs
        self.capacity_factor = cfg.moe_graph_capacity_factor
        self.static_expert_capacity = cfg.moe_static_expert_capacity
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.record_metrics = True
        self.graph_safe_counts = False
        self.graph_counts: Optional[torch.Tensor] = None

    @staticmethod
    def _has_grouped_mm() -> bool:
        return hasattr(F, "grouped_mm")

    def _can_grouped_mm(
        self,
        sorted_x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> bool:
        return (
            self._has_grouped_mm()
            and sorted_x.is_cuda
            and gate_weight.dtype == torch.bfloat16
            and up_weight.dtype == torch.bfloat16
            and down_weight.dtype == torch.bfloat16
        )

    def forward(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, dim = x.shape
        flat_expert = topk_idx.reshape(-1)
        flat_weight = topk_weight.reshape(-1)

        order = flat_expert.argsort()
        sorted_expert = flat_expert[order]
        sorted_token = torch.div(order, self.topk, rounding_mode="floor")
        sorted_weight = flat_weight[order]
        sorted_x = x.index_select(0, sorted_token)
        if self.graph_safe_counts:
            expert_ids = torch.arange(
                self.n_experts, device=sorted_expert.device, dtype=sorted_expert.dtype
            )
            boundaries = torch.searchsorted(sorted_expert, expert_ids, right=True)
            counts = boundaries - torch.cat((boundaries.new_zeros(1), boundaries[:-1]))
        else:
            counts = torch.bincount(sorted_expert, minlength=self.n_experts)
        if self.graph_counts is not None:
            self.graph_counts.add_(counts.to(dtype=self.graph_counts.dtype))

        backend = self.backend
        if backend == "auto":
            backend = (
                "grouped_mm"
                if self._can_grouped_mm(sorted_x, gate_weight, up_weight, down_weight)
                else "padded"
            )
        if backend == "grouped_mm":
            if not self._can_grouped_mm(sorted_x, gate_weight, up_weight, down_weight):
                raise RuntimeError(
                    "moe_backend='grouped_mm' requires CUDA and BF16 packed expert "
                    "weights. Use --model-param-dtype bf16 or "
                    "moe_backend='padded'."
                )
            sorted_out = self._torch_grouped_mm(
                sorted_x, counts, gate_weight, up_weight, down_weight
            )
        elif backend == "padded":
            sorted_out = self._padded_grouped(
                sorted_x, sorted_expert, counts, gate_weight, up_weight, down_weight
            )
        elif backend == "sorted":
            sorted_out = self._sorted_loop(
                sorted_x, counts, gate_weight, up_weight, down_weight
            )
        else:
            raise ValueError(f"Unsupported MoE backend: {self.backend}")

        sorted_out = sorted_out * sorted_weight.unsqueeze(-1)
        out = x.new_zeros(n_tokens, dim)
        out.index_add_(0, sorted_token, sorted_out)
        if self.record_metrics and not (x.is_cuda and _is_cuda_graph_capturing()):
            with torch.no_grad():
                self.last_metrics = {
                    "tokens_per_expert": counts.detach(),
                    "expert_load_max": counts.max().detach(),
                    "expert_load_mean": counts.float().mean().detach(),
                }
        return out

    def _torch_grouped_mm(
        self,
        sorted_x: torch.Tensor,
        counts: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = sorted_x.dtype
        sorted_x = sorted_x if sorted_x.dtype == torch.bfloat16 else sorted_x.to(torch.bfloat16)
        offs = counts.cumsum(0).to(torch.int32)
        gate = F.grouped_mm(sorted_x, gate_weight.transpose(1, 2), offs=offs)
        up = F.grouped_mm(sorted_x, up_weight.transpose(1, 2), offs=offs)
        hidden = F.silu(gate) * up
        return F.grouped_mm(hidden, down_weight.transpose(1, 2), offs=offs).to(orig_dtype)

    def _padded_grouped(
        self,
        sorted_x: torch.Tensor,
        sorted_expert: torch.Tensor,
        counts: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        total = sorted_x.size(0)
        if total == 0:
            return sorted_x.new_zeros(0, self.dim)
        if self.static_expert_capacity > 0:
            max_count = self.static_expert_capacity
        elif self.pad_for_cuda_graphs:
            if torch._dynamo.is_compiling():
                raise RuntimeError(
                    "Compiled graph-safe MoE padding requires "
                    "moe_static_expert_capacity > 0. Run a warmup benchmark first "
                    "and pass --moe-static-expert-capacity."
                )
            observed = int(counts.max().item())
            max_count = math.ceil(observed * self.capacity_factor)
            max_count = int(((max_count + 15) // 16) * 16)
        else:
            max_count = int(counts.max().item())
        if not torch._dynamo.is_compiling() and int(counts.max().item()) > max_count:
            raise RuntimeError(
                "MoE static expert capacity overflow: "
                f"max_count={int(counts.max().item())} capacity={max_count}. "
                "Increase --moe-static-expert-capacity or --moe-graph-capacity-factor."
            )
        offsets = counts.cumsum(0) - counts
        positions = torch.arange(total, device=sorted_x.device) - torch.repeat_interleave(
            offsets, counts
        )
        padded = sorted_x.new_zeros(self.n_experts, max_count, self.dim)
        padded[sorted_expert, positions] = sorted_x
        gate = torch.bmm(padded, gate_weight.transpose(1, 2))
        up = torch.bmm(padded, up_weight.transpose(1, 2))
        hidden = F.silu(gate) * up
        expert_out = torch.bmm(hidden, down_weight.transpose(1, 2))
        return expert_out[sorted_expert, positions]

    def _sorted_loop(
        self,
        sorted_x: torch.Tensor,
        counts: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        out = sorted_x.new_empty(sorted_x.shape[0], self.dim)
        offsets = counts.cumsum(0) - counts
        active = torch.nonzero(counts > 0, as_tuple=False).flatten()
        for eid_tensor in active:
            eid = int(eid_tensor)
            start = int(offsets[eid])
            end = start + int(counts[eid])
            chunk = sorted_x[start:end]
            hidden = F.silu(F.linear(chunk, gate_weight[eid])) * F.linear(
                chunk, up_weight[eid]
            )
            out[start:end] = F.linear(hidden, down_weight[eid])
        zero_use = (gate_weight.sum() + up_weight.sum() + down_weight.sum()) * 0.0
        return out + zero_use


class MoEFFN(nn.Module):
    """
    Fine-grained MoE FFN. The legacy implementation preserves the original
    Python dispatch path; the packed implementation uses a separated router and
    Megatron-style local token dispatcher for MetaTerid H100 training.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.impl = getattr(cfg, "moe_impl", "legacy")
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_metrics: dict[str, torch.Tensor] = {}

        if self.impl == "legacy":
            self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
            self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
            self.routed_experts = nn.ModuleList(
                [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
            )
            self.shared_experts = nn.ModuleList(
                [
                    Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                    for _ in range(self.n_shared)
                ]
            )
        elif self.impl == "packed":
            self.router = MetaTeridMoERouter(
                cfg.dim,
                cfg.n_experts,
                cfg.n_experts_per_tok,
                score_function=getattr(cfg, "router_score_function", "sigmoid"),
                normalize_topk=getattr(cfg, "normalize_topk", True),
                route_scale=getattr(cfg, "route_scale", 1.0),
                enable_router_bias=getattr(cfg, "enable_router_bias", True),
                seq_aux_loss_coeff=getattr(cfg, "seq_aux_loss_coeff", 0.0),
                router_bias_update_rate=getattr(cfg, "router_bias_update_rate", 1e-3),
            )
            self.dispatcher = MetaTeridMoEDispatcher(cfg)
            self.routed_gate_weight = nn.Parameter(
                torch.empty(cfg.n_experts, cfg.expert_dim, cfg.dim)
            )
            self.routed_up_weight = nn.Parameter(
                torch.empty(cfg.n_experts, cfg.expert_dim, cfg.dim)
            )
            self.routed_down_weight = nn.Parameter(
                torch.empty(cfg.n_experts, cfg.dim, cfg.expert_dim)
            )
            for param in (
                self.routed_gate_weight,
                self.routed_up_weight,
                self.routed_down_weight,
            ):
                nn.init.normal_(param, std=0.02)
            shared_hidden = cfg.n_shared_experts * cfg.n_experts_per_tok * cfg.expert_dim
            self.shared_gate = nn.Linear(cfg.dim, shared_hidden, bias=False)
            self.shared_up = nn.Linear(cfg.dim, shared_hidden, bias=False)
            self.shared_down = nn.Linear(shared_hidden, cfg.dim, bias=False)
        else:
            raise ValueError("moe_impl must be 'legacy' or 'packed'")

    @torch.no_grad()
    def update_router_bias(self, counts: Optional[torch.Tensor] = None) -> None:
        if self.impl == "packed":
            self.router.update_router_bias(counts)

    def _forward_legacy(self, flat: torch.Tensor) -> torch.Tensor:
        logits = self.router(flat)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])

        for shared in self.shared_experts:
            out = out + shared(flat)
        return out

    def _forward_packed(self, flat: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        topk_scores, topk_idx, aux_loss = self.router(flat, batch_size, seq_len)
        routed = self.dispatcher(
            flat,
            topk_idx,
            topk_scores,
            self.routed_gate_weight,
            self.routed_up_weight,
            self.routed_down_weight,
        )
        shared = self.shared_down(F.silu(self.shared_gate(flat)) * self.shared_up(flat))
        self.last_aux_loss = aux_loss
        self.last_metrics = {
            **self.router.last_metrics,
            **{f"dispatcher_{k}": v for k, v in self.dispatcher.last_metrics.items()},
        }
        return routed + shared

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.impl == "packed":
            old_bias_key = prefix + "router_bias"
            new_bias_key = prefix + "router.router_bias"
            if old_bias_key in state_dict and new_bias_key not in state_dict:
                state_dict[new_bias_key] = state_dict[old_bias_key]
                state_dict.pop(old_bias_key, None)

        if self.impl == "packed" and prefix + "routed_gate_weight" not in state_dict:
            routed_gate, routed_up, routed_down = [], [], []
            for eid in range(self.n_experts):
                gate_key = prefix + f"routed_experts.{eid}.gate.weight"
                up_key = prefix + f"routed_experts.{eid}.up.weight"
                down_key = prefix + f"routed_experts.{eid}.down.weight"
                if gate_key not in state_dict:
                    break
                routed_gate.append(state_dict[gate_key])
                routed_up.append(state_dict[up_key])
                routed_down.append(state_dict[down_key])
            if len(routed_gate) == self.n_experts:
                state_dict[prefix + "routed_gate_weight"] = torch.stack(routed_gate, 0)
                state_dict[prefix + "routed_up_weight"] = torch.stack(routed_up, 0)
                state_dict[prefix + "routed_down_weight"] = torch.stack(routed_down, 0)
                for eid in range(self.n_experts):
                    for name in ("gate", "up", "down"):
                        state_dict.pop(prefix + f"routed_experts.{eid}.{name}.weight", None)

        if self.impl == "packed" and prefix + "shared_gate.weight" not in state_dict:
            shared_gate, shared_up, shared_down = [], [], []
            for sid in range(self.n_shared):
                gate_key = prefix + f"shared_experts.{sid}.gate.weight"
                up_key = prefix + f"shared_experts.{sid}.up.weight"
                down_key = prefix + f"shared_experts.{sid}.down.weight"
                if gate_key not in state_dict:
                    break
                shared_gate.append(state_dict[gate_key])
                shared_up.append(state_dict[up_key])
                shared_down.append(state_dict[down_key])
            if len(shared_gate) == self.n_shared:
                state_dict[prefix + "shared_gate.weight"] = torch.cat(shared_gate, dim=0)
                state_dict[prefix + "shared_up.weight"] = torch.cat(shared_up, dim=0)
                state_dict[prefix + "shared_down.weight"] = torch.cat(shared_down, dim=1)
                for sid in range(self.n_shared):
                    for name in ("gate", "up", "down"):
                        state_dict.pop(prefix + f"shared_experts.{sid}.{name}.weight", None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.view(B * T, D)
        if self.impl == "legacy":
            out = self._forward_legacy(flat)
        else:
            out = self._forward_packed(flat, B, T)
        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


def precompute_loop_index_embeddings(
    model_dim: int, loop_dim: int, max_loops: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute loop-index embeddings as a real table for compiled training."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, dtype=torch.float32) / loop_dim)
    )
    loop_idx = torch.arange(max_loops, dtype=torch.float32).unsqueeze(1)
    angles = loop_idx * freqs.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:, :loop_dim]
    table = torch.zeros(max_loops, model_dim, dtype=torch.float32)
    table[:, :loop_dim] = emb
    return table


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        # Clamp for depth extrapolation: at inference n_loops can exceed the
        # training max_loop_iters. Iterations beyond the trained range reuse
        # the last learned per-loop scale rather than indexing out of range.
        t_idx = min(loop_t, self.scale.num_embeddings - 1)
        s = self.scale.weight[t_idx]  # (rank,)
        return self.forward_with_scale(x, s)

    def forward_with_scale(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply the adapter with an already-selected per-loop scale tensor."""
        down = self.down(x) * scale  # (B, T, rank)
        return down @ self.B  # (B, T, dim)


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor | bool] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Clamp keeps the product finite in float32 for any gradient step size.
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of OpenMythos — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, lora_rank, max_loop_iters, act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding
        self.register_buffer(
            "loop_index_table",
            precompute_loop_index_embeddings(cfg.dim, self.loop_dim, cfg.max_loop_iters),
        )

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor | bool] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        Can be increased at inference for deeper reasoning (depth extrapolation).
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key

        Returns:
            ACT-weighted sum of hidden states across iterations, shape (B, T, dim)
        """
        fixed_loop_count = n_loops is not None
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            if t < self.loop_index_table.size(0):
                loop_emb = self.loop_index_table[t].to(dtype=h.dtype)
                h_loop = h + loop_emb.view(1, 1, D)
                lora_scale = self.lora.scale.weight[t]
            else:
                h_loop = loop_index_embedding(h, t, self.loop_dim)
                lora_scale = self.lora.scale.weight[-1]
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora.forward_with_scale(trans_out, lora_scale)
            h = self.injection(h, e, trans_out)

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick: once cumulative_p + p crosses threshold,
            # assign the remaining probability mass as the final weight.
            # Gate by still_running so halted positions contribute exactly
            # once (on the halting step) and zero thereafter — otherwise
            # threshold<1 leaves a non-zero remainder that leaks every step.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # Only short-circuit when there is no KV cache to keep consistent.
            # With a cache, every loop depth must run on every forward pass so
            # later decode steps find populated keys at every cache_key.
            if not fixed_loop_count and halted.all() and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled)
        freqs = precompute_rope_cos_sin(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_cos_sin(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below the diagonal, -inf above.

        Args:
            seq_len -- sequence length
            device  -- target device
            dtype   -- tensor dtype (must match activation dtype so the additive
                       mask doesn't upcast the attention logits in the fallback
                       attention path — e.g. bf16 weights with an fp32 mask
                       promotes attn to fp32 and then breaks the fp32-vs-bf16
                       matmul against V)

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        mask = torch.full(
            (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids -- token indices of shape (B, T)
            n_loops   -- recurrent loop depth; defaults to cfg.max_loop_iters.
                         Increase at inference to extrapolate to harder problems.
            kv_cache  -- dict mutated in-place for autoregressive KV caching;
                         pass an empty dict {} and reuse across decode steps
            start_pos -- index of the first token in input_ids within the full
                         sequence; used to select the correct RoPE frequencies
                         during incremental decoding (0 for prefill, prompt_len
                         for each subsequent decode step)

        Returns:
            Logits of shape (B, T, vocab_size)
        """
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        mask = True if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # encoded input frozen for injection every loop
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops        -- recurrent loop depth for each decode step
            temperature    -- softmax temperature; lower = more greedy
            top_k          -- restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            logits = self.forward(
                cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
