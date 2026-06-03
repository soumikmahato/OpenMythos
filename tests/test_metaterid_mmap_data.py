import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.metaterid_data import MMapTokenDataset


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
