#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str
    prompt: str
    any_contains: tuple[str, ...] = ()
    all_contains: tuple[str, ...] = ()
    regex: str | None = None
    forbidden: tuple[str, ...] = ()
    max_new_tokens: int = 48
    temperature: float = 0.4
    top_k: int = 40
    max_trigram_repeat: float = 0.45


EVAL_CASES = [
    EvalCase("sun_direction", "language_factual", "The sun rises in the", any_contains=("east", "sky"), forbidden=("References", "|-")),
    EvalCase("ocean_sentence", "language", "Write one short sentence about the ocean.", regex=r"[A-Za-z][^.?!]{8,}[.?!]", max_new_tokens=32),
    EvalCase("france_capital", "factual", "The capital of France is", any_contains=("Paris",), max_new_tokens=24),
    EvalCase("usa_definition", "factual", "The United States of America is", any_contains=("country", "nation", "federal", "republic"), forbidden=("population of 1,869",), max_new_tokens=48),
    EvalCase("india_definition", "factual", "India is", any_contains=("country", "nation", "South Asia", "Asia"), forbidden=("language itself",), max_new_tokens=48),
    EvalCase("photosynthesis", "science", "Photosynthesis is", any_contains=("plants", "light", "energy", "carbon dioxide"), forbidden=("protein", "insulin"), max_new_tokens=48),
    EvalCase("insulin", "science", "Insulin is", any_contains=("hormone", "blood sugar", "glucose"), forbidden=("blood circulation",), max_new_tokens=48),
    EvalCase("cancer", "science", "Cancer is", any_contains=("disease", "cells", "abnormal", "tumor", "tumour"), forbidden=("chemical changes the chemical"), max_new_tokens=48),
    EvalCase("two_plus_two", "arithmetic", "2 + 2 =", any_contains=("4",), forbidden=("self.assert",), max_new_tokens=12, temperature=0.2, top_k=10),
    EvalCase("ten_plus_ten", "arithmetic", "10 + 10 =", any_contains=("20",), forbidden=("Lemma", "user"), max_new_tokens=12, temperature=0.2, top_k=10),
    EvalCase("ten_times_ten", "arithmetic", "10 * 10 =", any_contains=("100",), forbidden=("self.assert",), max_new_tokens=12, temperature=0.2, top_k=10),
    EvalCase("python_print", "code", "Python code to print hello world:", any_contains=("print", "Hello"), max_new_tokens=48, temperature=0.3, top_k=30),
    EvalCase("html_basic", "code", "A minimal HTML page starts with", any_contains=("<html", "<!DOCTYPE", "html"), forbidden=("self):",), max_new_tokens=48, temperature=0.3, top_k=30),
    EvalCase("nodejs", "code_factual", "Node.js is", any_contains=("JavaScript", "runtime", "server"), forbidden=("tuple"), max_new_tokens=48),
    EvalCase("chat_short", "instruction", "<|user|>Write one short sentence about rain.<|assistant|>", regex=r"[A-Za-z][^.?!]{8,}[.?!]", max_new_tokens=32),
    EvalCase("tool_call", "tool", "<|tool_call|>", any_contains=("{", "name", "arguments"), max_new_tokens=32, temperature=0.4, top_k=40),
]


def _trigram_repeat_rate(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < 6:
        return 0.0
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    if not trigrams:
        return 0.0
    return 1.0 - len(set(trigrams)) / len(trigrams)


def _load_cfg(checkpoint: dict, tokenizer: MetaTeridTokenizer, seq_len: int | None):
    cfg = checkpoint.get("cfg")
    if cfg is None:
        cfg = metaterid_t4_pilot()
    cfg.vocab_size = tokenizer.vocab_size
    if seq_len is not None:
        cfg.max_seq_len = seq_len
    return cfg


def _load_model(args: argparse.Namespace):
    tokenizer = MetaTeridTokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = _load_cfg(checkpoint, tokenizer, args.seq_len)
    model = MetaTeridForCausalLM(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return tokenizer, model


def _score(case: EvalCase, generated: str) -> tuple[bool, list[str], float]:
    continuation = generated[len(case.prompt) :].strip()
    lowered = continuation.lower()
    reasons: list[str] = []
    passed = True

    if case.regex:
        if re.search(case.regex, continuation):
            reasons.append("regex")
        else:
            reasons.append("missing_regex")
            passed = False

    if case.any_contains:
        if any(option.lower() in lowered for option in case.any_contains):
            reasons.append("any_contains")
        else:
            reasons.append("missing_any_contains")
            passed = False

    for option in case.all_contains:
        if option.lower() not in lowered:
            reasons.append(f"missing:{option}")
            passed = False

    for option in case.forbidden:
        if option.lower() in lowered:
            reasons.append(f"forbidden:{option}")
            passed = False

    repeat_rate = _trigram_repeat_rate(continuation)
    if repeat_rate > case.max_trigram_repeat:
        reasons.append(f"repeat:{repeat_rate:.2f}")
        passed = False

    if not continuation:
        reasons.append("empty")
        passed = False

    if not reasons:
        reasons.append("nonempty")
    return passed, reasons, repeat_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diversified MetaTerid generation eval suite.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = _load_model(args)
    cases = EVAL_CASES[: args.limit] if args.limit > 0 else EVAL_CASES
    results = []

    for case in cases:
        input_ids = torch.tensor(
            [tokenizer.encode(case.prompt)], dtype=torch.long, device=args.device
        )
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=case.max_new_tokens,
                n_loops=args.n_loops,
                temperature=case.temperature,
                top_k=case.top_k,
            )
        generated = tokenizer.decode(output[0].tolist())
        passed, reasons, repeat_rate = _score(case, generated)
        results.append(
            {
                "case": asdict(case),
                "passed": passed,
                "reasons": reasons,
                "trigram_repeat_rate": repeat_rate,
                "generated": generated,
            }
        )

    by_category: dict[str, list[dict]] = {}
    for item in results:
        by_category.setdefault(item["case"]["category"], []).append(item)

    passed_count = sum(item["passed"] for item in results)
    summary = {
        "passed": passed_count,
        "total": len(results),
        "pass_rate": passed_count / max(1, len(results)),
        "categories": {
            category: {
                "passed": sum(item["passed"] for item in items),
                "total": len(items),
                "pass_rate": sum(item["passed"] for item in items) / max(1, len(items)),
            }
            for category, items in sorted(by_category.items())
        },
    }

    print(f"MetaTerid eval suite: {passed_count}/{len(results)} passed ({summary['pass_rate']:.1%})")
    print("=" * 88)
    for category, cat_summary in summary["categories"].items():
        print(
            f"{category:>18}: {cat_summary['passed']}/{cat_summary['total']} "
            f"({cat_summary['pass_rate']:.1%})"
        )
    print("=" * 88)
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        case = item["case"]
        print(f"[{status}] {case['category']}::{case['name']} reasons={','.join(item['reasons'])}")
        print(item["generated"])
        print("-" * 88)

    if args.output_json:
        payload = {"summary": summary, "results": results}
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
