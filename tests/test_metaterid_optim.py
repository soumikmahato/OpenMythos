import pytest

torch = pytest.importorskip("torch")

from training.metaterid_optim import Muon, OptimizerList, build_optimizer


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.linear = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.linear(self.norm(x)).sum()


def test_build_adamw_muon_hybrid_steps():
    model = TinyModule()
    optimizer = build_optimizer(model, name="adamw_muon", lr=1e-3, weight_decay=0.01)
    assert isinstance(optimizer, OptimizerList)

    loss = model(torch.randn(2, 8))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_muon_matrix_step_updates_param():
    param = torch.nn.Parameter(torch.randn(8, 4))
    optimizer = Muon([param], lr=1e-3)
    before = param.detach().clone()
    param.grad = torch.randn_like(param)
    optimizer.step()
    assert not torch.equal(before, param.detach())
