#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class CorpusSource:
    name: str
    weight: float
    dataset: str | None = None
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    formatter: str = "auto"


HF_SOURCES = {
    "fineweb_edu": CorpusSource(
        name="fineweb_edu",
        weight=0.50,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    "openwebmath": CorpusSource(
        name="openwebmath",
        weight=0.18,
        dataset="open-web-math/open-web-math",
        text_field="text",
    ),
    "codeparrot": CorpusSource(
        name="codeparrot",
        weight=0.14,
        dataset="codeparrot/codeparrot-clean",
        text_field="content",
    ),
    "openhermes": CorpusSource(
        name="openhermes",
        weight=0.08,
        dataset="teknium/OpenHermes-2.5",
        text_field="conversations",
        formatter="messages",
    ),
    "tulu": CorpusSource(
        name="tulu",
        weight=0.04,
        dataset="allenai/tulu-3-sft-personas-instruction-following",
        text_field="messages",
        formatter="messages",
    ),
    "hermes_tools": CorpusSource(
        name="hermes_tools",
        weight=0.03,
        dataset="NousResearch/hermes-function-calling-v1",
        formatter="auto",
    ),
    "fineweb2_de": CorpusSource(
        name="fineweb2_de",
        weight=0.03,
        dataset="epfml/FineWeb2-HQ",
        config="deu_Latn",
        text_field="text",
    ),
}


SYNTHETIC_SOURCE_NAMES = {
    "synthetic_tools",
    "synthetic_code",
    "synthetic_math",
    "synthetic_chat",
    "synthetic_fim",
}


DEFAULT_SOURCES = [
    "fineweb_edu",
    "openwebmath",
    "codeparrot",
    "openhermes",
    "hermes_tools",
    "synthetic_tools",
    "synthetic_code",
    "synthetic_math",
    "synthetic_chat",
    "synthetic_fim",
]


def _format_messages(messages: list[dict]) -> str:
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
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or message.get("from") or "user").lower()
        content = message.get("content") or message.get("value") or ""
        if isinstance(content, str) and content.strip():
            parts.append(f"{role_tokens.get(role, '<|user|>')}{content.strip()}")
    return "".join(parts) + "<|eot|>" if parts else ""


def _format_sample(sample: dict, source: CorpusSource) -> str:
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


def _bounded_text(text: str, rng: random.Random, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    start = rng.randint(0, len(text) - max_chars)
    return text[start : start + max_chars]


def _iter_hf_source(source: CorpusSource, *, max_chars: int, seed: int) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config:
        kwargs["name"] = source.config

    stable_offset = int(hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_offset)
    dataset = load_dataset(**kwargs)
    for sample in dataset:
        text = _format_sample(sample, source)
        if text.strip():
            yield _bounded_text(text, rng, max_chars)


def _synthetic_math_examples() -> Iterator[str]:
    templates = [
        r"Let $a,b \in \mathbb{R}$. Then $(a+b)^2 = a^2 + 2ab + b^2$.",
        r"The derivative is \[\frac{d}{dx}x^n = n x^{n-1}.\]",
        r"Solve: \(2x + 5 = 13\). Therefore \(x = 4\).",
        r"\begin{align} y &= mx + b \\ \Delta &= b^2 - 4ac \end{align}",
        r"Euler's identity is $e^{i\pi} + 1 = 0$.",
        r"<|user|>What is 10 * 10?<|assistant|><|think|>10 groups of 10 equals 100.<|end_think|><|answer|>100<|eot|>",
    ]
    while True:
        for item in templates:
            yield item


def _synthetic_code_examples() -> Iterator[str]:
    templates = [
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
        "const http = require('node:http');\nhttp.createServer((req, res) => res.end('hello')).listen(3000);",
        "SELECT user_id, COUNT(*) AS n FROM events GROUP BY user_id ORDER BY n DESC;",
        "<html>\n  <body>\n    <h1>Hello world</h1>\n  </body>\n</html>",
        "class TokenBucket:\n    def __init__(self, rate: float):\n        self.rate = rate",
    ]
    while True:
        for item in templates:
            yield item


def _synthetic_tool_examples() -> Iterator[str]:
    templates = [
        '<|tool_call|>{"name":"web_search","arguments":{"query":"latest LLM tokenizer research"}}<|eot|>',
        '<|tool_result|>{"title":"Result","url":"https://example.com","snippet":"Tokenizer fertility matters."}<|eot|>',
        '<|tool|>python<|tool_call|>{"code":"print(2 + 2)"}<|tool_result|>4<|eot|>',
        '<|user|>Search for MetaTerid checkpoints.<|assistant|><|tool_call|>{"name":"web_search","arguments":{"query":"MetaTerid checkpoint"}}<|eot|>',
    ]
    while True:
        for item in templates:
            yield item


def _synthetic_chat_examples() -> Iterator[str]:
    templates = [
        "<|system|>You are MetaTerid, a concise assistant.<|user|>Explain photosynthesis.<|assistant|><|answer|>Photosynthesis lets plants use light to make sugars from carbon dioxide and water.<|eot|>",
        "<|user|>Give one short sentence about the ocean.<|assistant|>The ocean moves heat, water, and life around Earth.<|eot|>",
        "<|user|>Think briefly, then answer: 3 + 5.<|assistant|><|think|>3 + 5 = 8.<|end_think|><|answer|>8<|eot|>",
    ]
    while True:
        for item in templates:
            yield item


def _synthetic_fim_examples() -> Iterator[str]:
    templates = [
        "<|fim_prefix|>def multiply(a, b):\n    <|fim_suffix|>\nprint(multiply(3, 4))<|fim_middle|>return a * b",
        "<|fim_prefix|>The capital of France is <|fim_suffix|>.<|fim_middle|>Paris",
        "<|fim_prefix|>let area = <|fim_suffix|>; console.log(area);<|fim_middle|>width * height",
    ]
    while True:
        for item in templates:
            yield item


def _synthetic_iter(name: str) -> Iterator[str]:
    if name == "synthetic_tools":
        return _synthetic_tool_examples()
    if name == "synthetic_code":
        return _synthetic_code_examples()
    if name == "synthetic_math":
        return _synthetic_math_examples()
    if name == "synthetic_chat":
        return _synthetic_chat_examples()
    if name == "synthetic_fim":
        return _synthetic_fim_examples()
    raise ValueError(f"Unknown synthetic source: {name}")


def _build_iters(source_names: list[str], *, max_chars: int, seed: int) -> dict[str, Iterator[str]]:
    iters: dict[str, Iterator[str]] = {}
    for name in source_names:
        if name in HF_SOURCES:
            iters[name] = _iter_hf_source(HF_SOURCES[name], max_chars=max_chars, seed=seed)
        elif name in SYNTHETIC_SOURCE_NAMES:
            iters[name] = _synthetic_iter(name)
        else:
            raise ValueError(f"Unknown source: {name}")
    return iters


def _source_weights(source_names: list[str]) -> list[float]:
    weights = []
    for name in source_names:
        if name in HF_SOURCES:
            weights.append(HF_SOURCES[name].weight)
        else:
            weights.append(0.03)
    total = sum(weights)
    return [weight / total for weight in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a diverse corpus for MetaTerid tokenizer training.")
    parser.add_argument("--output-dir", default="data/tokenizer_corpus")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--total-docs", type=int, default=200_000)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument(
        "--hard-exit",
        action="store_true",
        help=(
            "Flush files and exit with os._exit(0) after success. Useful in some "
            "notebook runtimes where datasets/pyarrow finalizers abort at shutdown."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_names = [name.strip() for name in args.sources.split(",") if name.strip()]
    requested_sources = list(source_names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_iters = _build_iters(source_names, max_chars=args.max_chars, seed=args.seed)
    weights = _source_weights(source_names)
    handles = [
        (output_dir / f"tokenizer_corpus_{idx:03d}.txt").open("w", encoding="utf-8")
        for idx in range(args.shards)
    ]
    counts = {name: 0 for name in source_names}
    exhausted_sources: list[str] = []
    failed_sources: dict[str, str] = {}

    try:
        for idx in range(args.total_docs):
            name = rng.choices(source_names, weights=weights, k=1)[0]
            iterator = source_iters[name]
            try:
                text = next(iterator)
            except StopIteration:
                print(f"[tokenizer_corpus] source {name} exhausted", flush=True)
                source_names.remove(name)
                exhausted_sources.append(name)
                if not source_names:
                    break
                weights = _source_weights(source_names)
                continue
            except Exception as exc:
                print(f"[tokenizer_corpus] source {name} failed with {type(exc).__name__}: {exc}", flush=True)
                source_names.remove(name)
                failed_sources[name] = f"{type(exc).__name__}: {exc}"
                if not source_names:
                    break
                weights = _source_weights(source_names)
                continue

            text = text.strip()
            if not text:
                continue
            counts[name] = counts.get(name, 0) + 1
            handle = handles[idx % len(handles)]
            handle.write(text)
            handle.write("\n\n<|eot|>\n\n")
            if args.report_every > 0 and (idx + 1) % args.report_every == 0:
                print(f"[tokenizer_corpus] wrote {idx + 1:,}/{args.total_docs:,} docs", flush=True)
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "total_docs_requested": args.total_docs,
        "requested_sources": requested_sources,
        "active_sources": source_names,
        "exhausted_sources": exhausted_sources,
        "failed_sources": failed_sources,
        "counts": counts,
        "max_chars": args.max_chars,
        "shards": args.shards,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
