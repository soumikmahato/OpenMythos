import pytest

torch = pytest.importorskip("torch")

from open_mythos.metaterid import (
    BlockAttentionResidual,
    MetaTeridConfig,
    MetaTeridForCausalLM,
    metaterid_1b,
    metaterid_t4_pilot,
)


def tiny_cfg(**overrides) -> MetaTeridConfig:
    cfg = MetaTeridConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        lora_rank=4,
        use_block_attn_res=True,
        attn_res_heads=4,
        attn_res_max_sources=4,
        train_min_loops=1,
        train_max_loops=3,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_block_attention_residual_shape_and_grad():
    module = BlockAttentionResidual(dim=64, n_heads=4, max_sources=3)
    current = torch.randn(2, 8, 64, requires_grad=True)
    sources = [torch.randn(2, 8, 64) for _ in range(5)]

    out = module(current, sources)
    assert out.shape == current.shape

    out.mean().backward()
    assert current.grad is not None
    assert module.gate.grad is not None


def test_block_attention_residual_no_sources_is_identity():
    module = BlockAttentionResidual(dim=64, n_heads=4, max_sources=3)
    current = torch.randn(2, 8, 64)
    assert torch.equal(module(current, []), current)


def test_metaterid_forward_shape():
    cfg = tiny_cfg()
    model = MetaTeridForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

    logits = model(input_ids, n_loops=2)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_metaterid_without_attn_res_forward_shape():
    cfg = tiny_cfg(use_block_attn_res=False)
    model = MetaTeridForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))

    logits = model(input_ids, n_loops=1)
    assert logits.shape == (1, 6, cfg.vocab_size)


def test_metaterid_variants():
    target = metaterid_1b()
    pilot = metaterid_t4_pilot()

    assert target.vocab_size == 65_536
    assert target.n_experts == 64
    assert target.n_experts_per_tok == 4
    assert target.n_shared_experts == 2
    assert target.use_block_attn_res

    assert pilot.dim < target.dim
    assert pilot.max_seq_len <= 1024
    assert pilot.train_min_loops <= pilot.train_max_loops <= pilot.max_loop_iters
