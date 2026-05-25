from __future__ import annotations

from typing import Optional

import torch

from open_mythos.main import OpenMythos
from open_mythos.metaterid.attn_res import BlockAttentionResidual
from open_mythos.metaterid.config import MetaTeridConfig


class MetaTeridForCausalLM(OpenMythos):
    """
    MetaTerid language model.

    The first implementation keeps the proven OpenMythos recurrent backbone and
    adds Block Attention Residuals across major block outputs. This gives a
    measurable MetaTerid baseline without destabilizing the whole codebase.
    """

    cfg: MetaTeridConfig

    def __init__(self, cfg: MetaTeridConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.block_attn_res = (
            BlockAttentionResidual(
                cfg.dim,
                n_heads=cfg.attn_res_heads,
                max_sources=cfg.attn_res_max_sources,
                dropout=cfg.attn_res_dropout,
                gate_init=cfg.attn_res_gate_init,
            )
            if cfg.use_block_attn_res
            else None
        )

    def _apply_attn_res(
        self, x: torch.Tensor, sources: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.block_attn_res is None:
            return x
        return self.block_attn_res(x, sources)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        sources: list[torch.Tensor] = [x]

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")
            x = self._apply_attn_res(x, sources)
            sources.append(x)

        e = x
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
        x = self._apply_attn_res(x, sources)
        sources.append(x)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")
            x = self._apply_attn_res(x, sources)
            sources.append(x)

        x = self.norm(x)
        if return_hidden:
            return x
        return self.head(x)
