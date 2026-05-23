#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_t4_pilot
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer


DEFAULT_PROMPTS = [
    "Artificial intelligence is",
    "The sun rises in the",
    "The capital of France is",
    "Photosynthesis is",
    "2 + 2 =",
    "Python code to print hello world:",
]


def _read_prompts(path: str) -> list[str]:
    if not path:
        return []
    prompts = []
    file_path = Path(path)
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if file_path.suffix == ".jsonl":
            row = json.loads(line)
            prompts.append(str(row.get("prompt", "")))
        else:
            prompts.append(line)
    return [prompt for prompt in prompts if prompt]


def _wrap_chat(prompt: str) -> str:
    return f"<|user|>{prompt.strip()}<|assistant|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generation for MetaTerid checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = args.prompt + _read_prompts(args.prompts_file)
    if not prompts:
        prompts = DEFAULT_PROMPTS
    if args.chat:
        prompts = [_wrap_chat(prompt) for prompt in prompts]

    tokenizer = MetaTeridTokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg") or metaterid_t4_pilot()
    cfg.vocab_size = tokenizer.vocab_size
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len

    model = MetaTeridForCausalLM(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                n_loops=args.n_loops,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        generated = tokenizer.decode(output[0].tolist())
        rows.append({"prompt": prompt, "generated": generated})
        print("=" * 88)
        print(f"PROMPT: {prompt}")
        print("-" * 88)
        print(generated)

    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
