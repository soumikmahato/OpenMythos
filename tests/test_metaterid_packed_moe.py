import pytest

torch = pytest.importorskip("torch")

from open_mythos.main import MetaTeridMoERouter, MoEFFN, MythosConfig


def moe_cfg(**overrides):
    defaults = dict(
        vocab_size=128,
        dim=16,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=16,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=2,
        n_experts_per_tok=2,
        expert_dim=8,
        moe_backend="padded",
        router_score_function="softmax",
        seq_aux_loss_coeff=0.0,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


def copy_legacy_to_packed(legacy: MoEFFN, packed: MoEFFN) -> None:
    with torch.no_grad():
        packed.router.weight.copy_(legacy.router.weight)
        for eid, expert in enumerate(legacy.routed_experts):
            packed.routed_gate_weight[eid].copy_(expert.gate.weight)
            packed.routed_up_weight[eid].copy_(expert.up.weight)
            packed.routed_down_weight[eid].copy_(expert.down.weight)
        packed.shared_gate.weight.copy_(
            torch.cat([expert.gate.weight for expert in legacy.shared_experts], dim=0)
        )
        packed.shared_up.weight.copy_(
            torch.cat([expert.up.weight for expert in legacy.shared_experts], dim=0)
        )
        packed.shared_down.weight.copy_(
            torch.cat([expert.down.weight for expert in legacy.shared_experts], dim=1)
        )


def test_packed_softmax_matches_legacy_forward_and_backward():
    torch.manual_seed(0)
    legacy = MoEFFN(moe_cfg(moe_impl="legacy"))
    packed = MoEFFN(moe_cfg(moe_impl="packed"))
    copy_legacy_to_packed(legacy, packed)
    x_legacy = torch.randn(2, 5, 16, requires_grad=True)
    x_packed = x_legacy.detach().clone().requires_grad_(True)

    out_legacy = legacy(x_legacy)
    out_packed = packed(x_packed)
    assert torch.allclose(out_packed, out_legacy, atol=1e-5, rtol=1e-5)

    out_legacy.sum().backward()
    out_packed.sum().backward()
    assert torch.allclose(x_packed.grad, x_legacy.grad, atol=1e-4, rtol=1e-4)


def test_packed_moe_loads_legacy_state_dict_layout():
    torch.manual_seed(1)
    cfg = moe_cfg(moe_impl="packed")
    legacy = MoEFFN(moe_cfg(moe_impl="legacy"))
    packed = MoEFFN(cfg)
    packed.load_state_dict(legacy.state_dict(), strict=False)
    x = torch.randn(2, 4, cfg.dim)

    assert torch.allclose(packed(x), legacy(x), atol=1e-5, rtol=1e-5)


def test_sigmoid_router_weights_are_normalized_and_bias_is_buffer():
    router = MetaTeridMoERouter(
        dim=16,
        n_experts=4,
        topk=2,
        score_function="sigmoid",
        normalize_topk=True,
        enable_router_bias=True,
    )
    x = torch.randn(6, 16)
    weights, indices, _ = router(x, batch_size=2, seq_len=3)

    assert weights.shape == (6, 2)
    assert indices.shape == (6, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-6)
    assert "router_bias" not in {name for name, _ in router.named_parameters()}
    assert "router_bias" in {name for name, _ in router.named_buffers()}


def test_sequence_aux_loss_handles_batched_tokens():
    router = MetaTeridMoERouter(
        dim=16,
        n_experts=4,
        topk=2,
        score_function="sigmoid",
        seq_aux_loss_coeff=1e-4,
    )
    router.train()
    weights, indices, aux = router(torch.randn(6, 16), batch_size=2, seq_len=3)

    assert weights.shape == (6, 2)
    assert indices.shape == (6, 2)
    assert aux is not None
    assert aux.ndim == 0


def test_router_bias_update_moves_underloaded_experts_up():
    router = MetaTeridMoERouter(
        dim=4,
        n_experts=4,
        topk=1,
        enable_router_bias=True,
        router_bias_update_rate=0.1,
    )
    router.last_metrics = {"tokens_per_expert": torch.tensor([10, 0, 0, 0])}
    before = router.router_bias.clone()
    router.update_router_bias()
    assert router.router_bias[0] < before[0]
    assert torch.all(router.router_bias[1:] > before[1:])


def test_router_bias_update_accepts_global_counts_override():
    router = MetaTeridMoERouter(
        dim=4,
        n_experts=4,
        topk=1,
        enable_router_bias=True,
        router_bias_update_rate=0.1,
    )
    router.last_metrics = {"tokens_per_expert": torch.tensor([0, 0, 0, 1])}
    router.update_router_bias(torch.tensor([0, 8, 0, 0]))

    assert router.router_bias[1] < 0
    assert router.router_bias[0] > 0


def test_packed_moe_zero_token_experts_still_get_grad_slots():
    cfg = moe_cfg(moe_impl="packed", n_experts=4, n_experts_per_tok=1)
    moe = MoEFFN(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.router_bias.copy_(torch.tensor([10.0, -10.0, -10.0, -10.0]))
    x = torch.randn(1, 3, cfg.dim, requires_grad=True)
    out = moe(x).sum()
    out.backward()

    assert moe.routed_gate_weight.grad is not None
    assert moe.routed_up_weight.grad is not None
    assert moe.routed_down_weight.grad is not None


def test_padded_moe_static_capacity_overflow_is_explicit():
    cfg = moe_cfg(
        moe_impl="packed",
        n_experts=4,
        n_experts_per_tok=1,
        moe_pad_for_cuda_graphs=True,
        moe_static_expert_capacity=1,
    )
    moe = MoEFFN(cfg)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.router_bias.copy_(torch.tensor([10.0, -10.0, -10.0, -10.0]))

    with pytest.raises(RuntimeError, match="capacity overflow"):
        moe(torch.randn(1, 3, cfg.dim))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.nn.functional, "grouped_mm"),
    reason="torch grouped_mm MoE backend requires CUDA and PyTorch grouped_mm",
)
def test_grouped_mm_backend_matches_padded_cuda_bf16():
    torch.manual_seed(2)
    cfg_padded = moe_cfg(moe_impl="packed", moe_backend="padded")
    cfg_grouped = moe_cfg(moe_impl="packed", moe_backend="grouped_mm")
    padded = MoEFFN(cfg_padded).cuda().to(torch.bfloat16)
    grouped = MoEFFN(cfg_grouped).cuda().to(torch.bfloat16)
    grouped.load_state_dict(padded.state_dict())
    x = torch.randn(2, 8, cfg_padded.dim, device="cuda", dtype=torch.bfloat16)

    out_padded = padded(x)
    out_grouped = grouped(x)

    assert torch.allclose(out_grouped.float(), out_padded.float(), atol=2e-2, rtol=2e-2)
