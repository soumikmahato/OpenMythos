from __future__ import annotations

from pathlib import Path
from typing import Iterable

from transformers import PreTrainedTokenizerFast

METATERID_TOKENIZER_NAME = "metaterid-tokenizer-v1"
METATERID_VOCAB_SIZE = 65_536

METATERID_SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|think|>",
    "<|end_think|>",
    "<|answer|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|eot|>",
]


def _ensure_tokenizers_available() -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Training the MetaTerid tokenizer requires the `tokenizers` package. "
            "Install project dependencies or run `pip install tokenizers`."
        ) from exc


class MetaTeridTokenizer:
    """
    Fast tokenizer wrapper for MetaTerid.

    The intended production tokenizer is a custom byte-level BPE tokenizer with
    65,536 entries, including reserved chat, thinking, tool, and FIM tokens.
    """

    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)
        self._validate_special_tokens()

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in METATERID_SPECIAL_TOKENS
        }

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _validate_special_tokens(self) -> None:
        missing = [
            token
            for token in METATERID_SPECIAL_TOKENS
            if token not in self.tokenizer.get_vocab()
        ]
        if missing:
            raise ValueError(
                "MetaTerid tokenizer is missing required special tokens: "
                + ", ".join(missing)
            )


def train_metaterid_tokenizer(
    files: Iterable[str | Path],
    output_dir: str | Path,
    *,
    vocab_size: int = METATERID_VOCAB_SIZE,
    min_frequency: int = 2,
) -> Path:
    """
    Train and save the MetaTerid byte-level BPE tokenizer.

    Args:
        files: Text corpus files used to train the tokenizer.
        output_dir: Directory that will receive a HuggingFace-compatible tokenizer.
        vocab_size: Target vocabulary size including special tokens.
        min_frequency: Minimum BPE pair frequency.

    Returns:
        Path to the saved tokenizer directory.
    """

    _ensure_tokenizers_available()
    from tokenizers import ByteLevelBPETokenizer

    file_paths = [str(Path(path)) for path in files]
    if not file_paths:
        raise ValueError("At least one training text file is required.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train(
        files=file_paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=METATERID_SPECIAL_TOKENS,
    )

    tokenizer_json = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=[
            token
            for token in METATERID_SPECIAL_TOKENS
            if token not in {"<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>"}
        ],
    )
    fast.save_pretrained(output_path)
    return output_path
