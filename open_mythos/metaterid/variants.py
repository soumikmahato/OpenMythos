from __future__ import annotations

from open_mythos.metaterid.config import MetaTeridConfig
from open_mythos.metaterid_tokenizer import METATERID_VOCAB_SIZE


def metaterid_1b() -> MetaTeridConfig:
    """
    Target MetaTerid 1B-shape config around the 151M active MoE budget.

    This intentionally mirrors the OpenMythos 1B MoE active shape while using
    the MetaTerid tokenizer and Block AttnRes controls.
    """
    return MetaTeridConfig(
        vocab_size=METATERID_VOCAB_SIZE,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
        use_block_attn_res=True,
        attn_res_max_sources=8,
        attn_res_heads=4,
        attn_res_dropout=0.0,
        attn_res_gate_init=0.05,
        train_min_loops=4,
        train_max_loops=16,
    )


def metaterid_t4_pilot() -> MetaTeridConfig:
    """
    T4-friendly pilot config for the 2B-token training pipeline test.

    This is deliberately smaller than the target 1B config so a single T4 can
    validate data flow, checkpointing, loss behavior, recurrence, routing, and
    Block AttnRes before H100 scaling.
    """
    return MetaTeridConfig(
        vocab_size=METATERID_VOCAB_SIZE,
        dim=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        max_loop_iters=8,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_experts=16,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=512,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=4,
        dropout=0.0,
        use_block_attn_res=True,
        attn_res_max_sources=4,
        attn_res_heads=4,
        attn_res_dropout=0.0,
        attn_res_gate_init=0.05,
        train_min_loops=2,
        train_max_loops=8,
    )
