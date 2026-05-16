from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from open_mythos.metaterid_tokenizer import (
    METATERID_SPECIAL_TOKENS,
    METATERID_TOKENIZER_NAME,
    METATERID_VOCAB_SIZE,
    MetaTeridTokenizer,
    train_metaterid_tokenizer,
)
from training.train_metaterid_tokenizer import expand_inputs


def test_metaterid_tokenizer_constants():
    assert METATERID_TOKENIZER_NAME == "metaterid-tokenizer-v1"
    assert METATERID_VOCAB_SIZE == 65_536
    assert len(METATERID_SPECIAL_TOKENS) == len(set(METATERID_SPECIAL_TOKENS))

    for token in (
        "<|think|>",
        "<|end_think|>",
        "<|tool|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ):
        assert token in METATERID_SPECIAL_TOKENS


def test_train_and_load_metaterid_tokenizer(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "<|system|>You are MetaTerid.",
                "<|user|>Solve 2 + 2 and explain briefly.",
                "<|assistant|><|think|>2 + 2 = 4<|end_think|><|answer|>4",
                "<|tool_call|>{\"name\":\"python\",\"arguments\":\"2+2\"}",
                "<|tool_result|>4<|eot|>",
                "def add(a, b):\n    return a + b",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = train_metaterid_tokenizer(
        [corpus],
        tmp_path / "tokenizer",
        vocab_size=512,
        min_frequency=1,
    )

    tokenizer = MetaTeridTokenizer(output_dir)
    token_ids = tokenizer.special_token_ids

    assert output_dir.joinpath("tokenizer.json").is_file()
    assert set(METATERID_SPECIAL_TOKENS) <= set(token_ids)
    assert all(isinstance(token_id, int) for token_id in token_ids.values())

    text = "<|user|>What is 3 + 5?<|assistant|><|think|>8<|end_think|>"
    ids = tokenizer.encode(text)
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)

    decoded = tokenizer.decode(tokenizer.encode("MetaTerid keeps code: x += 1"))
    assert "MetaTerid" in decoded
    assert "x" in decoded


def test_expand_inputs_accepts_directories_files_and_globs(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    a = data_dir / "a.txt"
    b = data_dir / "b.txt"
    ignored = data_dir / "ignored.md"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    ignored.write_text("ignored", encoding="utf-8")

    expanded = expand_inputs(
        [
            str(data_dir),
            str(a),
            str(data_dir / "*.txt"),
        ]
    )

    assert expanded == [a.resolve(), b.resolve()]
