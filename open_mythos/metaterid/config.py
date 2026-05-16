from __future__ import annotations

from dataclasses import dataclass

from open_mythos.main import MythosConfig
from open_mythos.metaterid_tokenizer import METATERID_VOCAB_SIZE


@dataclass
class MetaTeridConfig(MythosConfig):
    """
    MetaTerid architecture configuration.

    This extends the working OpenMythos config instead of replacing it so the
    new model can reuse the tested attention, MoE, recurrence, generation, and
    checkpointing paths while adding MetaTerid-specific controls.
    """

    vocab_size: int = METATERID_VOCAB_SIZE
    model_name: str = "MetaTerid"
    use_block_attn_res: bool = True
    attn_res_max_sources: int = 8
    attn_res_heads: int = 4
    attn_res_dropout: float = 0.0
    attn_res_gate_init: float = 0.05
    target_active_moe_params: int = 151_000_000
    train_min_loops: int = 2
    train_max_loops: int = 8
