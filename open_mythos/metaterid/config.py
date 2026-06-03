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
    moe_impl: str = "packed"
    moe_backend: str = "auto"
    moe_dispatcher: str = "local_packed"
    router_score_function: str = "sigmoid"
    normalize_topk: bool = True
    enable_router_bias: bool = True
    router_bias_update_rate: float = 1e-3
    route_scale: float = 1.0
    seq_aux_loss_coeff: float = 1e-4
    moe_pad_for_cuda_graphs: bool = False
    moe_graph_capacity_factor: float = 1.25
    moe_static_expert_capacity: int = 0
