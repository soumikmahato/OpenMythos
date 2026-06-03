#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class CorpusSource:
    name: str
    weight: float
    dataset: str | None = None
    config: str | None = None
    data_dir: str | None = None
    split: str = "train"
    text_field: str = "text"
    formatter: str = "auto"
    min_chars: int = 200


HF_SOURCES = {
    "fineweb_edu": CorpusSource(
        name="fineweb_edu",
        weight=0.25,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    "ultrafineweb_l3_qa_en": CorpusSource(
        name="ultrafineweb_l3_qa_en",
        weight=0.18,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-QA-Synthetic",
        text_field="text",
    ),
    "ultrafineweb_l3_multistyle_en": CorpusSource(
        name="ultrafineweb_l3_multistyle_en",
        weight=0.12,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        text_field="text",
    ),
    "openwebmath": CorpusSource(
        name="openwebmath",
        weight=0.10,
        dataset="open-web-math/open-web-math",
        text_field="text",
    ),
    "ultradata_math_l3_qa": CorpusSource(
        name="ultradata_math_l3_qa",
        weight=0.08,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-QA-Synthetic",
        text_field="text",
        formatter="auto",
    ),
    "ultradata_math_l3_textbook": CorpusSource(
        name="ultradata_math_l3_textbook",
        weight=0.06,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-Textbook-Exercise-Synthetic",
        text_field="text",
        formatter="auto",
    ),
    "ultradata_sft_math": CorpusSource(
        name="ultradata_sft_math",
        weight=0.04,
        dataset="openbmb/UltraData-SFT-2605",
        config="Math",
        split="no_think",
        formatter="auto",
        min_chars=80,
    ),
    "ultradata_sft_code": CorpusSource(
        name="ultradata_sft_code",
        weight=0.04,
        dataset="openbmb/UltraData-SFT-2605",
        config="Code",
        split="no_think",
        formatter="auto",
        min_chars=80,
    ),
    "ultradata_sft_if": CorpusSource(
        name="ultradata_sft_if",
        weight=0.03,
        dataset="openbmb/UltraData-SFT-2605",
        config="IF",
        split="no_think",
        formatter="auto",
        min_chars=80,
    ),
    "codeparrot": CorpusSource(
        name="codeparrot",
        weight=0.06,
        dataset="codeparrot/codeparrot-clean",
        text_field="content",
        min_chars=80,
    ),
    "the_stack_python": CorpusSource(
        name="the_stack_python",
        weight=0.04,
        dataset="bigcode/the-stack",
        data_dir="data/python",
        text_field="content",
        min_chars=80,
    ),
    "the_stack_markdown": CorpusSource(
        name="the_stack_markdown",
        weight=0.02,
        dataset="bigcode/the-stack",
        data_dir="data/markdown",
        text_field="content",
        min_chars=80,
    ),
    "the_stack_tex": CorpusSource(
        name="the_stack_tex",
        weight=0.02,
        dataset="bigcode/the-stack",
        data_dir="data/tex",
        text_field="content",
        min_chars=80,
    ),
    "the_stack_javascript": CorpusSource(
        name="the_stack_javascript",
        weight=0.02,
        dataset="bigcode/the-stack",
        data_dir="data/javascript",
        text_field="content",
        min_chars=80,
    ),
    "the_stack_sql": CorpusSource(
        name="the_stack_sql",
        weight=0.01,
        dataset="bigcode/the-stack",
        data_dir="data/sql",
        text_field="content",
        min_chars=80,
    ),
    "openhermes": CorpusSource(
        name="openhermes",
        weight=0.02,
        dataset="teknium/OpenHermes-2.5",
        text_field="conversations",
        formatter="messages",
        min_chars=80,
    ),
    "tulu": CorpusSource(
        name="tulu",
        weight=0.01,
        dataset="allenai/tulu-3-sft-personas-instruction-following",
        text_field="messages",
        formatter="messages",
        min_chars=80,
    ),
    "hermes_tools": CorpusSource(
        name="hermes_tools",
        weight=0.02,
        dataset="NousResearch/hermes-function-calling-v1",
        formatter="auto",
        min_chars=80,
    ),
}


SYNTHETIC_SOURCE_NAMES = {
    "synthetic_tools",
    "synthetic_code",
    "synthetic_math",
    "synthetic_chat",
    "synthetic_fim",
}

SYNTHETIC_WEIGHTS = {
    "synthetic_tools": 0.012,
    "synthetic_code": 0.012,
    "synthetic_math": 0.012,
    "synthetic_chat": 0.008,
    "synthetic_fim": 0.008,
}


DEFAULT_SOURCES = [
    "fineweb_edu",
    "ultrafineweb_l3_qa_en",
    "ultrafineweb_l3_multistyle_en",
    "openwebmath",
    "ultradata_math_l3_qa",
    "ultradata_math_l3_textbook",
    "ultradata_sft_math",
    "ultradata_sft_code",
    "ultradata_sft_if",
    "codeparrot",
    "the_stack_python",
    "the_stack_markdown",
    "the_stack_tex",
    "the_stack_javascript",
    "the_stack_sql",
    "openhermes",
    "hermes_tools",
    "synthetic_tools",
    "synthetic_code",
    "synthetic_math",
    "synthetic_chat",
    "synthetic_fim",
]

SEPARATOR_LINE_RE = re.compile(r"^\s*[\*\|/#\\]*(?:[-_=*#~]{12,})[\*\|/#\\\s]*$")
LONG_REPEAT_RE = re.compile(r"([ \t\-_=*#~])\1{15,}")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.ASCII)
MATH_OR_CODE_HINT_RE = re.compile(
    r"(\\begin|\\frac|\\sum|\\mathbb|```|def |class |SELECT |<html|</|#include|"
    r"function\s+\w+|=>|<\|tool_call\|>|<\|fim_|[$][^$]+[$])"
)
SAFE_NON_ASCII = set("πΠθΘλΛμΜσΣΩωαβγδε∑∏∞≈≠≤≥±×÷√∂∆∇∈∉∫")


def _latin_or_math_char(ch: str) -> bool:
    if ch in SAFE_NON_ASCII:
        return True
    if ch.isascii():
        return True
    name = unicodedata.name(ch, "")
    return "LATIN" in name


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
    if isinstance(content, str):
        return content

    string_values = [
        value
        for value in sample.values()
        if isinstance(value, str) and len(value.strip()) >= source.min_chars
    ]
    if string_values:
        return max(string_values, key=len)
    return ""


def _sample_allowed(sample: dict, source: CorpusSource) -> bool:
    language = sample.get("language") or sample.get("lang")
    if isinstance(language, str):
        language = language.lower()
        if language not in {"en", "eng", "english", "code"} and not source.name.startswith("the_stack"):
            return False
    language_score = sample.get("language_score") or sample.get("lang_score")
    if isinstance(language_score, (int, float)) and language_score < 0.65:
        return False
    alphanum_fraction = sample.get("alphanum_fraction")
    if isinstance(alphanum_fraction, (int, float)) and alphanum_fraction < 0.12:
        return False
    return True


def _bounded_text(text: str, rng: random.Random, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    start = rng.randint(0, len(text) - max_chars)
    return text[start : start + max_chars]


def _clean_candidate_text(text: str) -> tuple[str, Counter]:
    reasons: Counter = Counter()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    separator_lines = 0
    blank_run = 0
    max_blank_run = 0

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            max_blank_run = max(max_blank_run, blank_run)
            if blank_run <= 2:
                cleaned_lines.append("")
            continue

        blank_run = 0
        if SEPARATOR_LINE_RE.match(line):
            separator_lines += 1
            continue

        line = re.sub(r"[ \t]{5,}", "    ", line.rstrip())
        line = re.sub(r"([\-_=*#~])\1{7,}", lambda m: m.group(1) * 4, line)
        cleaned_lines.append(line)

    if separator_lines:
        reasons["separator_lines_removed"] = separator_lines
    if max_blank_run > 3:
        reasons["long_blank_runs_collapsed"] = max_blank_run

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned, reasons


def _text_quality_reason(text: str, source: CorpusSource) -> str | None:
    if len(text) < source.min_chars:
        return "too_short"

    printable = [ch for ch in text if ch.isprintable() or ch in "\n\t"]
    if not printable:
        return "no_printable_text"

    same_char_match = LONG_REPEAT_RE.search(text)
    if same_char_match:
        return "long_repeated_character_run"

    hint = MATH_OR_CODE_HINT_RE.search(text) is not None
    alnum = sum(ch.isalnum() for ch in text)
    separator_chars = sum(ch in "-_=*#~" for ch in text)
    ascii_or_latin = sum(_latin_or_math_char(ch) for ch in printable)
    punctuation_only_lines = 0
    nonempty_lines = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        nonempty_lines += 1
        if len(stripped) >= 16 and PUNCT_ONLY_RE.match(stripped):
            punctuation_only_lines += 1

    if separator_chars / max(1, len(text)) > 0.18 and not hint:
        return "separator_dominated"
    if punctuation_only_lines >= 3 and punctuation_only_lines / max(1, nonempty_lines) > 0.25:
        return "punctuation_line_dominated"
    if alnum / max(1, len(text)) < 0.12 and not hint:
        return "low_alnum"
    if ascii_or_latin / max(1, len(printable)) < 0.82 and not hint:
        return "non_english_or_non_latin"
    return None


def clean_tokenizer_text(text: str, source: CorpusSource) -> tuple[str, str | None, Counter]:
    cleaned, notes = _clean_candidate_text(text)
    reason = _text_quality_reason(cleaned, source)
    return cleaned, reason, notes


def _iter_hf_source(source: CorpusSource, *, max_chars: int, seed: int) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config:
        kwargs["name"] = source.config
    if source.data_dir:
        kwargs["data_dir"] = source.data_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        kwargs["token"] = token

    stable_offset = int(hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_offset)
    dataset = load_dataset(**kwargs)
    for sample in dataset:
        if not _sample_allowed(sample, source):
            continue
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
            weights.append(SYNTHETIC_WEIGHTS.get(name, 0.01))
    total = sum(weights)
    return [weight / total for weight in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a diverse corpus for MetaTerid tokenizer training.")
    parser.add_argument("--output-dir", default="data/tokenizer_corpus")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--total-docs", type=int, default=200_000)
    parser.add_argument(
        "--target-chars",
        type=int,
        default=0,
        help=(
            "Stop after this many accepted UTF-8 characters. For the production "
            "3B-token tokenizer corpus, use roughly 12B-15B characters as a "
            "token-count proxy before the final tokenizer exists."
        ),
    )
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
    accepted_docs = 0
    accepted_chars = 0
    raw_samples = 0
    skipped = {name: Counter() for name in source_names}
    clean_notes = {name: Counter() for name in source_names}
    seen_hashes: set[str] = set()
    exhausted_sources: list[str] = []
    failed_sources: dict[str, str] = {}

    try:
        while accepted_docs < args.total_docs:
            if args.target_chars > 0 and accepted_chars >= args.target_chars:
                break
            name = rng.choices(source_names, weights=weights, k=1)[0]
            iterator = source_iters[name]
            try:
                text = next(iterator)
                raw_samples += 1
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

            source = HF_SOURCES.get(name, CorpusSource(name=name, weight=0.0, min_chars=1))
            text, skip_reason, notes = clean_tokenizer_text(text, source)
            clean_notes.setdefault(name, Counter()).update(notes)
            if skip_reason:
                skipped.setdefault(name, Counter())[skip_reason] += 1
                continue
            if not text:
                skipped.setdefault(name, Counter())["empty_after_cleaning"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                skipped.setdefault(name, Counter())["duplicate"] += 1
                continue
            seen_hashes.add(digest)

            counts[name] = counts.get(name, 0) + 1
            accepted_docs += 1
            accepted_chars += len(text)
            handle = handles[(accepted_docs - 1) % len(handles)]
            handle.write(text)
            handle.write("\n\n<|eot|>\n\n")
            if args.report_every > 0 and accepted_docs % args.report_every == 0:
                print(
                    "[tokenizer_corpus] "
                    f"accepted={accepted_docs:,}/{args.total_docs:,} "
                    f"chars={accepted_chars:,} raw={raw_samples:,}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "total_docs_requested": args.total_docs,
        "target_chars": args.target_chars,
        "accepted_docs": accepted_docs,
        "accepted_chars": accepted_chars,
        "raw_samples": raw_samples,
        "requested_sources": requested_sources,
        "active_sources": source_names,
        "exhausted_sources": exhausted_sources,
        "failed_sources": failed_sources,
        "counts": counts,
        "skipped": {name: dict(counter) for name, counter in skipped.items()},
        "cleaning_notes": {name: dict(counter) for name, counter in clean_notes.items()},
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
