#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    contains: tuple[str, ...] = ()
    regex: str | None = None
    max_new_tokens: int = 24
    temperature: float = 0.4
    top_k: int = 30


EVAL_CASES = [
    EvalCase(
        name="sentence_sun",
        prompt="The sun rises in the",
        contains=("sky", "east"),
        max_new_tokens=12,
    ),
    EvalCase(
        name="sentence_france",
        prompt="The capital of France is",
        contains=("Paris", "paris"),
        max_new_tokens=12,
    ),
    EvalCase(
        name="definition_ai",
        prompt="Artificial intelligence is",
        contains=("computer", "machine", "technology", "system"),
        max_new_tokens=24,
    ),
    EvalCase(
        name="python_definition",
        prompt="Python is a programming language that",
        contains=("code", "program", "software", "used"),
        max_new_tokens=24,
    ),
    EvalCase(
        name="simple_math_2_plus_2",
        prompt="2 + 2 =",
        contains=("4",),
        max_new_tokens=8,
        temperature=0.2,
        top_k=10,
    ),
    EvalCase(
        name="simple_math_3_plus_5",
        prompt="3 + 5 =",
        contains=("8",),
        max_new_tokens=8,
        temperature=0.2,
        top_k=10,
    ),
    EvalCase(
        name="chat_sentence",
        prompt="<|user|>Write one short sentence about the ocean.<|assistant|>",
        regex=r"[A-Za-z][^.?!]{8,}[.?!]",
        max_new_tokens=24,
    ),
    EvalCase(
        name="tool_format_probe",
        prompt="<|tool_call|>",
        contains=("{", "name", "arguments"),
        max_new_tokens=24,
        temperature=0.5,
        top_k=40,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small MetaTerid checkpoint eval.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _load_model(args: argparse.Namespace):
    tokenizer = MetaTeridTokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg") or metaterid_t4_pilot()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = args.seq_len

    model = MetaTeridForCausalLM(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return tokenizer, model


def _score(case: EvalCase, generated: str) -> tuple[bool, str]:
    continuation = generated[len(case.prompt) :].strip()
    if case.regex and re.search(case.regex, continuation):
        return True, "regex"
    if case.contains:
        lowered = continuation.lower()
        for option in case.contains:
            if option.lower() in lowered:
                return True, f"contains:{option}"
        return False, "missing_contains"
    return bool(continuation), "nonempty" if continuation else "empty"


def main() -> None:
    args = parse_args()
    tokenizer, model = _load_model(args)
    results = []

    for case in EVAL_CASES:
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
        passed, reason = _score(case, generated)
        results.append(
            {
                "name": case.name,
                "passed": passed,
                "reason": reason,
                "prompt": case.prompt,
                "generated": generated,
            }
        )

    passed_count = sum(item["passed"] for item in results)
    print(f"MetaTerid eval: {passed_count}/{len(results)} passed")
    print("=" * 80)
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['name']} ({item['reason']})")
        print(item["generated"])
        print("-" * 80)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
