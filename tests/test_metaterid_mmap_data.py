import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import training.metaterid_data as metaterid_data
from training.metaterid_data import DataSource, MMapTokenDataset, MixedTokenDataset, get_mix_sources


def write_shard(root, name, rows, dtype="uint16"):
    arr = np.asarray(rows, dtype=np.dtype(dtype))
    path = root / name
    arr.tofile(path)
    return {
        "path": name,
        "dtype": dtype,
        "record_len": arr.shape[1],
        "records": arr.shape[0],
        "weight": 1.0,
    }


def test_mmap_dataset_next_token_alignment(tmp_path):
    shard = write_shard(tmp_path, "tokens.bin", [[1, 2, 3, 4], [5, 6, 7, 8]])
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": [shard]}),
        encoding="utf-8",
    )

    dataset = MMapTokenDataset(tmp_path, seq_len=3, seed=123)
    x, y = next(iter(dataset))

    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.shape == (3,)
    assert y.shape == (3,)
    assert torch.equal(y, x + 1)


def test_mmap_dataset_rank_shards_deterministically(tmp_path):
    shard = write_shard(
        tmp_path,
        "tokens.bin",
        [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": [shard]}),
        encoding="utf-8",
    )

    rank0 = MMapTokenDataset(tmp_path, seq_len=2, rank=0, world_size=2, seed=7)
    rank1 = MMapTokenDataset(tmp_path, seq_len=2, rank=1, world_size=2, seed=7)

    x0, _ = next(iter(rank0))
    x1, _ = next(iter(rank1))
    assert not torch.equal(x0, x1)


def test_streaming_dataset_adds_document_boundaries(monkeypatch):
    class DummyTokenizer:
        def encode(self, text):
            return [ord(ch) - 96 for ch in text]

        def encode_document(self, text):
            return [101, *self.encode(text), 102]

    def fake_iter_source_text(source, rank, world_size):
        yield "abc"

    monkeypatch.setattr(metaterid_data, "iter_source_text", fake_iter_source_text)
    dataset = MixedTokenDataset(
        DummyTokenizer(),
        seq_len=4,
        sources=[DataSource(name="dummy", weight=1.0)],
    )

    x, y = next(iter(dataset))

    assert x.tolist() == [101, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 102]


def test_web_heavy_mix_rebalances_l3_down_and_general_web_up():
    sources = {source.name: source.weight for source in get_mix_sources("main_base_web_heavy_v1")}

    assert sum(sources.values()) == pytest.approx(1.0)
    assert sources["ultrafineweb_en"] + sources["filtered_fineweb_edu"] == pytest.approx(0.48)
    assert sources["ultrafineweb_l3_multistyle_en"] + sources["ultrafineweb_l3_qa_en"] == pytest.approx(0.08)
