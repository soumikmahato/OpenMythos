#!/usr/bin/env python3
"""
Inspect candidate MetaTerid training datasets before expensive mmap/H100 runs.

The script prints configs/splits/features where available, samples head and
shuffled rows through HF streaming, and flags common tokenizer/training
pollution patterns such as SAP-style banner headers, markdown table separator
rows, long punctuation banners, unicode-space artifacts, and non-English-heavy
text. It never stores or prints HF tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import (  # noqa: E402
    Dataset,
    get_dataset_config_names,
    get_dataset_infos,
    get_dataset_split_names,
    load_dataset,
)

BANNER_RE = re.compile(r"^[ \t]*(?:[/*#;!%-]|\*&|\*)?[ \t]*(?:[-_=*#~]){16,}[ \t]*(?:\*/)?[ \t]*$", re.MULTILINE)
SAP_REPORT_RE = re.compile(r"(?im)^\s*\*&[-*]{10,}|^\s*\*&\s+Report\s+\w+")
MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"(?m)^\s*\|?(?:\s*:?-{3,}:?\s*\|){2,}\s*:?-{3,}:?\s*\|?\s*$")
LONG_SPACE_RE = re.compile(r"[ \t]{8,}")
UNICODE_SPACE_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
REPEATED_PUNCT_RE = re.compile(r"([-_=*#~])\1{15,}")
SEPARATOR_LINE_RE = re.compile(r"^\s*[\*\|/#\\]*(?:[-_=*#~]{12,})[\*\|/#\\\s]*$")
LONG_REPEAT_RE = re.compile(r"([ \t\-_=*#~])\1{15,}")
LATIN_RE = re.compile(r"[A-Za-z]")
NON_LATIN_RE = re.compile(r"[^\W\d_A-Za-z]", re.UNICODE)
COMMENTISH_RE = re.compile(r"^\s*(?:#|//|/\*|\*|\*/|;|--|%|REM\b|\*&)")
HEADER_KEYWORD_RE = re.compile(
    r"(?i)\b(?:copyright|license|licence|author|created by|generated|auto-generated|"
    r"report\s+\w+|filter empty values|local interface|importing)\b"
)


@dataclass(frozen=True)
class InspectSource:
    name: str
    dataset: str
    config: str | None = None
    data_dir: str | None = None
    split: str = "train"
    text_field: str = "text"
    formatter: str = "auto"


SOURCES = [
    InspectSource("ultrafineweb_en", "openbmb/Ultra-FineWeb", split="en", text_field="content"),
    InspectSource(
        "ultrafineweb_l3_qa_en",
        "openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-QA-Synthetic",
    ),
    InspectSource(
        "ultrafineweb_l3_multistyle_en",
        "openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
    ),
    InspectSource("fineweb_edu", "HuggingFaceFW/fineweb-edu", config="sample-10BT"),
    InspectSource("openwebmath", "open-web-math/open-web-math"),
    InspectSource(
        "ultradata_math_l3_qa",
        "openbmb/UltraData-Math",
        config="UltraData-Math-L3-QA-Synthetic",
    ),
    InspectSource(
        "ultradata_math_l3_textbook",
        "openbmb/UltraData-Math",
        config="UltraData-Math-L3-Textbook-Exercise-Synthetic",
    ),
    InspectSource(
        "ultradata_sft_code_no_think",
        "openbmb/UltraData-SFT-2605",
        config="Code",
        split="no_think",
        formatter="auto",
    ),
    InspectSource(
        "ultradata_sft_math_no_think",
        "openbmb/UltraData-SFT-2605",
        config="Math",
        split="no_think",
        formatter="auto",
    ),
    InspectSource(
        "ultradata_sft_if_no_think",
        "openbmb/UltraData-SFT-2605",
        config="IF",
        split="no_think",
        formatter="auto",
    ),
    InspectSource("the_stack_python", "bigcode/the-stack", data_dir="data/python", text_field="content"),
    InspectSource("the_stack_javascript", "bigcode/the-stack", data_dir="data/javascript", text_field="content"),
    InspectSource("the_stack_markdown", "bigcode/the-stack", data_dir="data/markdown", text_field="content"),
    InspectSource("the_stack_tex", "bigcode/the-stack", data_dir="data/tex", text_field="content"),
    InspectSource("the_stack_sql", "bigcode/the-stack", data_dir="data/sql", text_field="content"),
    InspectSource("starcoderdata_unavailable_audit", "bigcode/starcoderdata", data_dir="python", text_field="content"),
    InspectSource("the_stack_smol_unavailable_audit", "bigcode/the-stack-smol", data_dir="data/python", text_field="content"),
    InspectSource("the_stack_v2_train_smol_ids_audit", "bigcode/the-stack-v2-train-smol-ids", text_field="blob_id"),
    InspectSource("hermes_tools", "NousResearch/hermes-function-calling-v1", formatter="auto"),
]


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


def _short(text: str, limit: int = 700) -> str:
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def _dataset_kwargs(source: InspectSource, *, streaming: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": streaming,
        "token": _token(),
    }
    if source.config:
        kwargs["name"] = source.config
    if source.data_dir:
        kwargs["data_dir"] = source.data_dir
    return kwargs


def _source_to_data_source(source: InspectSource) -> DataSource:
    return source


def _strip_leading_polluting_header(lines: list[str]) -> list[str]:
    window = lines[:80]
    if len(window) < 4:
        return lines
    end = 0
    commentish = 0
    separators = 0
    keyword_hit = False
    for idx, line in enumerate(window):
        stripped = line.strip()
        if not stripped:
            if end:
                end = idx + 1
            continue
        is_separator = bool(SEPARATOR_LINE_RE.match(line) or MARKDOWN_TABLE_SEPARATOR_RE.match(line))
        is_commentish = bool(COMMENTISH_RE.match(line))
        if not (is_separator or is_commentish):
            break
        end = idx + 1
        commentish += int(is_commentish)
        separators += int(is_separator)
        keyword_hit = keyword_hit or bool(HEADER_KEYWORD_RE.search(line))
    if end >= 4 and (keyword_hit or separators >= 2) and commentish >= max(2, end // 2):
        return lines[end:]
    return lines


def clean_training_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = UNICODE_SPACE_RE.sub(" ", text)
    latin = len(LATIN_RE.findall(text))
    non_latin = len(NON_LATIN_RE.findall(text))
    if non_latin > 200 and non_latin > latin * 0.5:
        return ""
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in _strip_leading_polluting_header(text.split("\n")):
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")
            continue
        blank_run = 0
        if SEPARATOR_LINE_RE.match(line) or MARKDOWN_TABLE_SEPARATOR_RE.match(line):
            continue
        line = LONG_REPEAT_RE.sub(lambda match: match.group(1) * 4, line)
        line = re.sub(r"[ \t]{5,}", "    ", line)
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


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


def _format_sample(sample: dict, source: InspectSource) -> str:
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


def _extract_text(row: dict[str, Any], source: InspectSource) -> str:
    return _format_sample(row, _source_to_data_source(source))


def _artifact_counts(text: str) -> Counter:
    counts: Counter = Counter()
    if BANNER_RE.search(text):
        counts["banner_line"] += 1
    if SAP_REPORT_RE.search(text):
        counts["sap_report_header"] += 1
    if MARKDOWN_TABLE_SEPARATOR_RE.search(text):
        counts["markdown_table_separator"] += 1
    if LONG_SPACE_RE.search(text):
        counts["long_space_run"] += 1
    if UNICODE_SPACE_RE.search(text):
        counts["unicode_space"] += 1
    if REPEATED_PUNCT_RE.search(text):
        counts["repeated_punct"] += 1
    latin = len(LATIN_RE.findall(text))
    non_latin = len(NON_LATIN_RE.findall(text))
    if non_latin > 200 and non_latin > latin * 0.5:
        counts["non_latin_heavy"] += 1
    return counts


def _sample_rows(source: InspectSource, *, head: int, shuffled: int, seed: int) -> tuple[list[dict], list[dict]]:
    ds = load_dataset(**_dataset_kwargs(source, streaming=True))
    head_rows = [row for _, row in zip(range(head), ds)]
    shuffled_rows: list[dict] = []
    if shuffled > 0:
        ds2 = load_dataset(**_dataset_kwargs(source, streaming=True)).shuffle(seed=seed, buffer_size=5000)
        shuffled_rows = [row for _, row in zip(range(shuffled), ds2)]
    return head_rows, shuffled_rows


def _features_from_stream(source: InspectSource) -> list[str]:
    ds = load_dataset(**_dataset_kwargs(source, streaming=True))
    features = getattr(ds, "features", None)
    if features:
        return list(features.keys())
    for row in ds:
        return list(row.keys())
    return []


def inspect_source(source: InspectSource, *, head: int, shuffled: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"source": asdict(source)}
    try:
        result["configs_sample"] = get_dataset_config_names(source.dataset, token=_token())[:20]
    except Exception as exc:
        result["configs_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["splits"] = get_dataset_split_names(
            source.dataset,
            source.config,
            token=_token(),
        )
    except Exception as exc:
        result["splits_error"] = f"{type(exc).__name__}: {exc}"
    try:
        infos = get_dataset_infos(source.dataset, token=_token())
        info = infos.get(source.config or "default")
        if info is not None:
            result["license"] = getattr(info, "license", None)
            result["description_head"] = _short(getattr(info, "description", "") or "", 500)
    except Exception as exc:
        result["info_error"] = f"{type(exc).__name__}: {exc}"

    result["features"] = _features_from_stream(source)
    head_rows, shuffled_rows = _sample_rows(source, head=head, shuffled=shuffled, seed=seed)
    result["head_raw"] = []
    result["shuffled_raw"] = []
    artifact_counts: Counter = Counter()
    clean_changed = 0
    text_lengths = []
    empty_text = 0
    for label, rows, out_key in [
        ("head", head_rows, "head_raw"),
        ("shuffled", shuffled_rows, "shuffled_raw"),
    ]:
        for idx, row in enumerate(rows):
            text = _extract_text(row, source)
            if not text.strip():
                empty_text += 1
            text_lengths.append(len(text))
            artifact_counts.update(_artifact_counts(text))
            cleaned = clean_training_text(text)
            if cleaned != text.strip():
                clean_changed += 1
            result[out_key].append(
                {
                    "idx": idx,
                    "columns": list(row.keys()),
                    "text_len": len(text),
                    "artifact_flags": dict(_artifact_counts(text)),
                    "raw_text": _short(text),
                    "cleaned_text": _short(cleaned),
                }
            )
    total = len(head_rows) + len(shuffled_rows)
    result["sample_summary"] = {
        "sampled_rows": total,
        "empty_text": empty_text,
        "text_len_min": min(text_lengths) if text_lengths else 0,
        "text_len_max": max(text_lengths) if text_lengths else 0,
        "text_len_avg": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "clean_changed_rows": clean_changed,
        "artifact_counts": dict(artifact_counts),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(source.name for source in SOURCES))
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--shuffled", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="dataset_inspection.json")
    args = parser.parse_args()

    selected = {part.strip() for part in args.sources.split(",") if part.strip()}
    source_map = {source.name: source for source in SOURCES}
    unknown = sorted(selected - set(source_map))
    if unknown:
        raise SystemExit(f"Unknown sources: {unknown}; available={sorted(source_map)}")

    results = []
    for name in [source.name for source in SOURCES if source.name in selected]:
        source = source_map[name]
        print(f"[inspect] {name} -> {source.dataset} {source.config or ''} {source.split}", flush=True)
        try:
            results.append(inspect_source(source, head=args.head, shuffled=args.shuffled, seed=args.seed))
        except Exception as exc:
            results.append({"source": asdict(source), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[inspect] ERROR {name}: {type(exc).__name__}: {exc}", flush=True)

    path = Path(args.output)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
