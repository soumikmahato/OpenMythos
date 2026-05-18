from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class DataSource:
    name: str
    weight: float
    dataset: str | None = None
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    local_jsonl: str | None = None
    formatter: str = "auto"
    fallback: "DataSource | None" = None


METATERID_T4_PILOT_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.60,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="small_starcoder_code",
        weight=0.10,
        dataset="bigcode/starcoderdata",
        split="train",
        text_field="content",
        fallback=DataSource(
            name="small_stack_smol_code_fallback",
            weight=0.10,
            dataset="bigcode/the-stack-smol",
            split="train",
            text_field="content",
            fallback=DataSource(
                name="codeparrot_clean_code_fallback",
                weight=0.10,
                dataset="codeparrot/codeparrot-clean",
                split="train",
                text_field="content",
            ),
        ),
    ),
    DataSource(
        name="math_stem",
        weight=0.10,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="reference_wiki",
        weight=0.06,
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="instruction_openhermes_25",
        weight=0.06,
        dataset="teknium/OpenHermes-2.5",
        split="train",
        text_field="conversations",
        formatter="messages",
    ),
    DataSource(
        name="instruction_tulu3_personas_if",
        weight=0.03,
        dataset="allenai/tulu-3-sft-personas-instruction-following",
        split="train",
        text_field="messages",
        formatter="messages",
    ),
    DataSource(
        name="tool_chat_hermes_function_calling",
        weight=0.03,
        dataset="NousResearch/hermes-function-calling-v1",
        split="train",
        formatter="auto",
    ),
    DataSource(
        name="multilingual_fineweb2_hq",
        weight=0.02,
        dataset="epfml/FineWeb2-HQ",
        config="deu_Latn",
        split="train",
        text_field="text",
        formatter="text",
    ),
]

METATERID_T4_LOCAL_PRIVATE_MIX = [
    source
    for source in METATERID_T4_PILOT_MIX
    if source.name
    not in {
        "instruction_tulu3_personas_if",
        "tool_chat_hermes_function_calling",
        "multilingual_fineweb2_hq",
    }
] + [
    DataSource(
        name="instruction_local",
        weight=0.05,
        local_jsonl="data/instruction.jsonl",
        formatter="auto",
    ),
    DataSource(
        name="tool_chat_local",
        weight=0.03,
        local_jsonl="data/tool_chat.jsonl",
        formatter="auto",
    ),
    DataSource(
        name="multilingual_local",
        weight=0.02,
        local_jsonl="data/multilingual.jsonl",
        formatter="auto",
    ),
]


METATERID_T4_KAGGLE_CHUNK_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.70,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="codeparrot_clean_code",
        weight=0.10,
        dataset="codeparrot/codeparrot-clean",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.12,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="instruction_openhermes_25",
        weight=0.08,
        dataset="teknium/OpenHermes-2.5",
        split="train",
        text_field="conversations",
        formatter="messages",
    ),
]


METATERID_T4_KAGGLE_FINEWEB_ONLY_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=1.0,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
]


METATERID_T4_KAGGLE_NO_MATH_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.80,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="codeparrot_clean_code",
        weight=0.10,
        dataset="codeparrot/codeparrot-clean",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="instruction_openhermes_25",
        weight=0.10,
        dataset="teknium/OpenHermes-2.5",
        split="train",
        text_field="conversations",
        formatter="messages",
    ),
]


METATERID_T4_KAGGLE_FINEWEB_MATH_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.75,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.25,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
]


METATERID_T4_KAGGLE_FINEWEB_CODE_INSTRUCT_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.80,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="codeparrot_clean_code",
        weight=0.10,
        dataset="codeparrot/codeparrot-clean",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="instruction_openhermes_25",
        weight=0.10,
        dataset="teknium/OpenHermes-2.5",
        split="train",
        text_field="conversations",
        formatter="messages",
    ),
]


METATERID_T4_KAGGLE_FACTUAL_REFERENCE_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.75,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="reference_wiki",
        weight=0.25,
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
    ),
]


METATERID_T4_KAGGLE_INSTRUCT_TOOL_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.70,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="instruction_openhermes_25",
        weight=0.15,
        dataset="teknium/OpenHermes-2.5",
        split="train",
        text_field="conversations",
        formatter="messages",
    ),
    DataSource(
        name="instruction_tulu3_personas_if",
        weight=0.10,
        dataset="allenai/tulu-3-sft-personas-instruction-following",
        split="train",
        text_field="messages",
        formatter="messages",
    ),
    DataSource(
        name="tool_chat_hermes_function_calling",
        weight=0.05,
        dataset="NousResearch/hermes-function-calling-v1",
        split="train",
        formatter="auto",
    ),
]


METATERID_T4_KAGGLE_MULTILINGUAL_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.80,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="multilingual_fineweb2_hq",
        weight=0.20,
        dataset="epfml/FineWeb2-HQ",
        config="deu_Latn",
        split="train",
        text_field="text",
        formatter="text",
    ),
]


METATERID_T4_KAGGLE_CONSOLIDATE_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.70,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.10,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="codeparrot_clean_code",
        weight=0.08,
        dataset="codeparrot/codeparrot-clean",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="reference_wiki",
        weight=0.07,
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="instruction_openhermes_25",
        weight=0.05,
        dataset="teknium/OpenHermes-2.5",
        split="train",
        text_field="conversations",
        formatter="messages",
    ),
]


def normalize_weights(sources: list[DataSource]) -> list[DataSource]:
    total = sum(source.weight for source in sources)
    if total <= 0:
        raise ValueError("dataset source weights must sum to a positive value")
    return [
        DataSource(
            name=source.name,
            weight=source.weight / total,
            dataset=source.dataset,
            config=source.config,
            split=source.split,
            text_field=source.text_field,
            local_jsonl=source.local_jsonl,
            formatter=source.formatter,
            fallback=source.fallback,
        )
        for source in sources
    ]


def _rank_worker_shard(rank: int, world_size: int) -> tuple[int, int]:
    worker = get_worker_info()
    num_workers = worker.num_workers if worker else 1
    worker_id = worker.id if worker else 0
    total_shards = world_size * num_workers
    shard_index = rank * num_workers + worker_id
    return total_shards, shard_index


def _format_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    role_tokens = {
        "system": "<|system|>",
        "user": "<|user|>",
        "human": "<|user|>",
        "assistant": "<|assistant|>",
        "gpt": "<|assistant|>",
        "tool": "<|tool_result|>",
        "tool_call": "<|tool_call|>",
        "function": "<|tool_result|>",
    }
    for message in messages:
        role = str(message.get("role") or message.get("from") or "user").lower()
        content = message.get("content") or message.get("value") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        parts.append(f"{role_tokens.get(role, '<|user|>')}{content.strip()}")
    return "".join(parts) + "<|eot|>" if parts else ""


def _format_sample(sample: dict, source: DataSource) -> str:
    value = sample.get(source.text_field)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _format_messages(value)

    for key in ("messages", "conversations"):
        messages = sample.get(key)
        if isinstance(messages, list):
            return _format_messages(messages)

    inputs = sample.get("inputs") or sample.get("instruction") or sample.get("prompt")
    targets = sample.get("targets") or sample.get("output") or sample.get("response")
    if isinstance(inputs, str) and isinstance(targets, str):
        return f"<|user|>{inputs.strip()}<|assistant|>{targets.strip()}<|eot|>"

    tools = sample.get("tools") or sample.get("tool")
    query = sample.get("query") or sample.get("question")
    answer = sample.get("answer") or sample.get("completion")
    if isinstance(query, str) and isinstance(answer, str):
        tool_text = f"<|tool|>{tools}" if tools else ""
        return f"{tool_text}<|user|>{query.strip()}<|assistant|>{answer.strip()}<|eot|>"

    content = sample.get("content") or sample.get("text")
    return content if isinstance(content, str) else ""


def _iter_local_jsonl(path: Path, source: DataSource) -> Iterator[str]:
    if not path.exists():
        return iter(())

    def _reader() -> Iterator[str]:
        while True:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = _format_sample(row, source)
                    if text.strip():
                        yield text

    return _reader()


def _iter_hf_stream(source: DataSource, rank: int, world_size: int) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config:
        kwargs["name"] = source.config

    ds = load_dataset(**kwargs)
    total_shards, shard_index = _rank_worker_shard(rank, world_size)
    manual_shard = False
    try:
        ds = ds.shard(num_shards=total_shards, index=shard_index)
    except IndexError:
        # Some streaming datasets expose fewer physical data sources than the
        # number of DDP ranks x DataLoader workers, and a few even fail
        # rank-level sharding when num_shards > dataset.num_shards. Keep the
        # stream alive by sharding records manually. This is less network
        # efficient but reliable for Kaggle smoke/pilot runs.
        manual_shard = True

    for row_idx, sample in enumerate(ds):
        if manual_shard and row_idx % total_shards != shard_index:
            continue
        text = _format_sample(sample, source)
        if text.strip():
            yield text


def iter_source_text(source: DataSource, rank: int, world_size: int) -> Iterator[str]:
    if source.local_jsonl is not None:
        return _iter_local_jsonl(Path(source.local_jsonl), source)
    if source.dataset is None:
        return iter(())
    if source.fallback is None:
        return _iter_hf_stream(source, rank, world_size)

    def _with_fallback() -> Iterator[str]:
        try:
            yield from _iter_hf_stream(source, rank, world_size)
        except Exception as exc:
            print(
                f"[metaterid_data] Source {source.name} failed with {type(exc).__name__}: {exc}. "
                f"Falling back to {source.fallback.name}.",
                flush=True,
            )
            yield from iter_source_text(source.fallback, rank, world_size)

    return _with_fallback()


class MixedTokenDataset(IterableDataset):
    """
    Weighted streaming dataset that packs text into fixed-length token chunks.

    The sampler chooses a source by weight, pulls one document from that source,
    appends it to a rolling token buffer, and yields next-token prediction pairs
    of fixed length. Missing local optional files are skipped rather than
    failing the pilot run.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        sources: list[DataSource],
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1337,
        max_sample_chars: int = 131_072,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sources = normalize_weights(sources)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.max_sample_chars = max_sample_chars

    def _bounded_text(self, text: str, rng: random.Random) -> str:
        if self.max_sample_chars <= 0 or len(text) <= self.max_sample_chars:
            return text
        start = rng.randint(0, len(text) - self.max_sample_chars)
        return text[start : start + self.max_sample_chars]

    def __iter__(self):
        rng = random.Random(self.seed + self.rank)
        source_iters = {
            source.name: iter_source_text(source, self.rank, self.world_size)
            for source in self.sources
        }
        names = [source.name for source in self.sources]
        weights = [source.weight for source in self.sources]
        buf: list[int] = []

        while True:
            name = rng.choices(names, weights=weights, k=1)[0]
            iterator = source_iters[name]
            try:
                text = next(iterator)
            except StopIteration:
                source_iters[name] = itertools.cycle(())
                continue

            text = self._bounded_text(text, rng)
            buf.extend(self.tokenizer.encode(text))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
