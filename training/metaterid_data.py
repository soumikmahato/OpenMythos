from __future__ import annotations

import itertools
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class DataSource:
    name: str
    weight: float
    dataset: str | None = None
    config: str | None = None
    data_dir: str | None = None
    split: str = "train"
    text_field: str = "text"
    local_jsonl: str | None = None
    formatter: str = "auto"
    fallback: "DataSource | None" = None


SEPARATOR_LINE_RE = re.compile(r"^\s*[\*\|/#\\]*(?:[-_=*#~]{12,})[\*\|/#\\\s]*$")
MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|){2,}\s*:?-{3,}:?\s*\|?\s*$")
LONG_REPEAT_RE = re.compile(r"([ \t\-_=*#~])\1{15,}")
UNICODE_SPACE_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
LATIN_RE = re.compile(r"[A-Za-z]")
NON_LATIN_RE = re.compile(r"[^\W\d_A-Za-z]", re.UNICODE)
COMMENTISH_RE = re.compile(r"^\s*(?:#|//|/\*|\*|\*/|;|--|%|REM\b|\*&)")
HEADER_KEYWORD_RE = re.compile(
    r"(?i)\b(?:copyright|license|licence|author|created by|generated|auto-generated|"
    r"report\s+\w+|filter empty values|local interface|importing)\b"
)


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
    """
    Apply the tokenizer-production artifact filters to training streams.

    This keeps markdown/code separator junk and unicode-space artifacts from
    becoming overrepresented in mmap shards or online streaming batches.
    """
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


METATERID_MAIN_BASE_V0_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.62,
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
        weight=0.10,
        dataset="codeparrot/codeparrot-clean",
        split="train",
        text_field="content",
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
        name="reference_wiki_low",
        weight=0.04,
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
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


METATERID_MAIN_STABLE_WEB_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.90,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.05,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="codeparrot_clean_code",
        weight=0.05,
        dataset="codeparrot/codeparrot-clean",
        split="train",
        text_field="content",
    ),
]


METATERID_MAIN_REASONING_BOOTSTRAP_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.60,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.20,
        dataset="open-web-math/open-web-math",
        split="train",
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


METATERID_MAIN_LOCAL_CURATED_MIX = [
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.70,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="local_arithmetic",
        weight=0.10,
        local_jsonl="data/arithmetic.jsonl",
        formatter="auto",
    ),
    DataSource(
        name="local_factual_qa",
        weight=0.08,
        local_jsonl="data/factual_qa.jsonl",
        formatter="auto",
    ),
    DataSource(
        name="local_instruction",
        weight=0.07,
        local_jsonl="data/instruction.jsonl",
        formatter="auto",
    ),
    DataSource(
        name="local_tool_chat",
        weight=0.05,
        local_jsonl="data/tool_chat.jsonl",
        formatter="auto",
    ),
]


# The Stack v2 train-smol release is the preferred long-term code source, but
# its public HF training split stores Software Heritage/provenance ids rather
# than direct file content. StarCoderData and Stack-smol are gated separately
# and were not accessible with the current HF token during the 2026-06-04 audit.
# Immediate production uses selected accessible The Stack language slices and
# the shared artifact cleaner; avoid ABAP and other header-heavy enterprise
# languages unless we add stronger language-specific cleaning.


METATERID_BASE_FIRST_V1_MIX = [
    DataSource(
        name="ultrafineweb_en",
        weight=0.37,
        dataset="openbmb/Ultra-FineWeb",
        split="en",
        text_field="content",
    ),
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.15,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="the_stack_python",
        weight=0.13,
        dataset="bigcode/the-stack",
        data_dir="data/python",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="the_stack_javascript",
        weight=0.04,
        dataset="bigcode/the-stack",
        data_dir="data/javascript",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="the_stack_markdown",
        weight=0.01,
        dataset="bigcode/the-stack",
        data_dir="data/markdown",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.10,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="ultradata_math_l3_textbook",
        weight=0.05,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-Textbook-Exercise-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultrafineweb_l3_multistyle_en",
        weight=0.07,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultrafineweb_l3_qa_en",
        weight=0.04,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-QA-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultradata_sft_code_no_think",
        weight=0.010,
        dataset="openbmb/UltraData-SFT-2605",
        config="Code",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="ultradata_sft_math_no_think",
        weight=0.008,
        dataset="openbmb/UltraData-SFT-2605",
        config="Math",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="ultradata_sft_if_no_think",
        weight=0.007,
        dataset="openbmb/UltraData-SFT-2605",
        config="IF",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="tool_chat_hermes_function_calling",
        weight=0.015,
        dataset="NousResearch/hermes-function-calling-v1",
        split="train",
        formatter="auto",
    ),
]


METATERID_BASE_MIDDLE_V1_MIX = [
    DataSource(
        name="ultrafineweb_en",
        weight=0.26,
        dataset="openbmb/Ultra-FineWeb",
        split="en",
        text_field="content",
    ),
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.10,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="the_stack_python",
        weight=0.12,
        dataset="bigcode/the-stack",
        data_dir="data/python",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="the_stack_javascript",
        weight=0.05,
        dataset="bigcode/the-stack",
        data_dir="data/javascript",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="the_stack_markdown",
        weight=0.01,
        dataset="bigcode/the-stack",
        data_dir="data/markdown",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.10,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="ultradata_math_l3_qa",
        weight=0.06,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-QA-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultradata_math_l3_textbook",
        weight=0.04,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-Textbook-Exercise-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultrafineweb_l3_multistyle_en",
        weight=0.09,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultrafineweb_l3_qa_en",
        weight=0.08,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-QA-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultradata_sft_code_no_think",
        weight=0.025,
        dataset="openbmb/UltraData-SFT-2605",
        config="Code",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="ultradata_sft_math_no_think",
        weight=0.020,
        dataset="openbmb/UltraData-SFT-2605",
        config="Math",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="ultradata_sft_if_no_think",
        weight=0.015,
        dataset="openbmb/UltraData-SFT-2605",
        config="IF",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="tool_chat_hermes_function_calling",
        weight=0.030,
        dataset="NousResearch/hermes-function-calling-v1",
        split="train",
        formatter="auto",
    ),
]


METATERID_BASE_FINAL_V1_MIX = [
    DataSource(
        name="ultrafineweb_en",
        weight=0.18,
        dataset="openbmb/Ultra-FineWeb",
        split="en",
        text_field="content",
    ),
    DataSource(
        name="filtered_fineweb_edu",
        weight=0.06,
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_field="text",
    ),
    DataSource(
        name="the_stack_python",
        weight=0.13,
        dataset="bigcode/the-stack",
        data_dir="data/python",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="the_stack_javascript",
        weight=0.06,
        dataset="bigcode/the-stack",
        data_dir="data/javascript",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="the_stack_markdown",
        weight=0.01,
        dataset="bigcode/the-stack",
        data_dir="data/markdown",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="math_stem_openwebmath",
        weight=0.08,
        dataset="open-web-math/open-web-math",
        split="train",
        text_field="text",
    ),
    DataSource(
        name="ultradata_math_l3_qa",
        weight=0.08,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-QA-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultradata_math_l3_textbook",
        weight=0.04,
        dataset="openbmb/UltraData-Math",
        config="UltraData-Math-L3-Textbook-Exercise-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultrafineweb_l3_multistyle_en",
        weight=0.10,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultrafineweb_l3_qa_en",
        weight=0.12,
        dataset="openbmb/Ultra-FineWeb-L3",
        config="Ultra-FineWeb-L3-en-QA-Synthetic",
        split="train",
        text_field="content",
    ),
    DataSource(
        name="ultradata_sft_code_no_think",
        weight=0.040,
        dataset="openbmb/UltraData-SFT-2605",
        config="Code",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="ultradata_sft_math_no_think",
        weight=0.030,
        dataset="openbmb/UltraData-SFT-2605",
        config="Math",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="ultradata_sft_if_no_think",
        weight=0.025,
        dataset="openbmb/UltraData-SFT-2605",
        config="IF",
        split="no_think",
        formatter="auto",
    ),
    DataSource(
        name="tool_chat_hermes_function_calling",
        weight=0.045,
        dataset="NousResearch/hermes-function-calling-v1",
        split="train",
        formatter="auto",
    ),
]


MIX_PRESETS = {
    "pilot": METATERID_T4_PILOT_MIX,
    "kaggle_chunk": METATERID_T4_KAGGLE_CHUNK_MIX,
    "kaggle_fineweb_only": METATERID_T4_KAGGLE_FINEWEB_ONLY_MIX,
    "kaggle_no_math": METATERID_T4_KAGGLE_NO_MATH_MIX,
    "kaggle_fineweb_math": METATERID_T4_KAGGLE_FINEWEB_MATH_MIX,
    "kaggle_fineweb_code_instruct": METATERID_T4_KAGGLE_FINEWEB_CODE_INSTRUCT_MIX,
    "kaggle_factual_reference": METATERID_T4_KAGGLE_FACTUAL_REFERENCE_MIX,
    "kaggle_instruct_tool": METATERID_T4_KAGGLE_INSTRUCT_TOOL_MIX,
    "kaggle_multilingual": METATERID_T4_KAGGLE_MULTILINGUAL_MIX,
    "kaggle_consolidate": METATERID_T4_KAGGLE_CONSOLIDATE_MIX,
    "final": METATERID_MAIN_BASE_V0_MIX,
    "current_mixed": METATERID_MAIN_BASE_V0_MIX,
    "main_base_v0": METATERID_MAIN_BASE_V0_MIX,
    "main_base_first_v1": METATERID_BASE_FIRST_V1_MIX,
    "main_base_middle_v1": METATERID_BASE_MIDDLE_V1_MIX,
    "main_base_final_v1": METATERID_BASE_FINAL_V1_MIX,
    "main_smoke_50m_v1": METATERID_BASE_FIRST_V1_MIX,
    "main_stable_web": METATERID_MAIN_STABLE_WEB_MIX,
    "main_reasoning_bootstrap": METATERID_MAIN_REASONING_BOOTSTRAP_MIX,
    "main_local_curated": METATERID_MAIN_LOCAL_CURATED_MIX,
}


def get_mix_sources(name: str) -> list[DataSource]:
    try:
        return MIX_PRESETS[name]
    except KeyError as exc:
        options = ", ".join(sorted(MIX_PRESETS))
        raise ValueError(f"Unknown mix preset '{name}'. Available: {options}") from exc


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
            data_dir=source.data_dir,
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
                    text = clean_training_text(_format_sample(row, source))
                    if text:
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
    if source.data_dir:
        kwargs["data_dir"] = source.data_dir

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
        text = clean_training_text(text)
        if text:
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


@dataclass(frozen=True)
class MMapShard:
    path: Path
    dtype: str
    record_len: int
    records: int
    weight: float = 1.0


def _load_mmap_manifest(mmap_dir: str | Path) -> list[MMapShard]:
    root = Path(mmap_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing mmap manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards: list[MMapShard] = []
    for row in manifest.get("shards", []):
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        records = int(row["records"])
        if records <= 0:
            continue
        shards.append(
            MMapShard(
                path=path,
                dtype=str(row["dtype"]),
                record_len=int(row["record_len"]),
                records=records,
                weight=float(row.get("weight", 1.0)),
            )
        )
    if not shards:
        raise ValueError(f"No usable mmap shards found in {manifest_path}")
    return shards


class MMapTokenDataset(IterableDataset):
    """
    Pre-tokenized fixed-record dataset backed by mmap shards.

    Each record is exactly seq_len + 1 token ids. Iteration returns next-token
    pairs without tokenizer calls, which keeps H100 training off the Python text
    formatting/tokenization path.
    """

    def __init__(
        self,
        mmap_dir: str | Path,
        seq_len: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1337,
    ):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shards = _load_mmap_manifest(mmap_dir)
        for shard in self.shards:
            if shard.record_len != seq_len + 1:
                raise ValueError(
                    f"Shard {shard.path} record_len={shard.record_len}; expected {seq_len + 1}"
                )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        total_workers = self.world_size * num_workers
        global_worker = self.rank * num_workers + worker_id
        rng = random.Random(self.seed + 9973 * global_worker)
        weights = [shard.weight for shard in self.shards]
        arrays = [
            np.memmap(
                shard.path,
                mode="r",
                dtype=np.dtype(shard.dtype),
                shape=(shard.records, shard.record_len),
            )
            for shard in self.shards
        ]
        cursors = [global_worker % shard.records for shard in self.shards]

        while True:
            shard_idx = rng.choices(range(len(self.shards)), weights=weights, k=1)[0]
            shard = self.shards[shard_idx]
            record_idx = cursors[shard_idx]
            cursors[shard_idx] = (record_idx + total_workers) % shard.records
            row = np.asarray(arrays[shard_idx][record_idx], dtype=np.int64)
            yield (
                torch.from_numpy(row[:-1].copy()),
                torch.from_numpy(row[1:].copy()),
            )


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
