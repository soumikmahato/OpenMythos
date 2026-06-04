#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from metaterid_checkpoint import load_metaterid_model
from open_mythos.metaterid_tokenizer import MetaTeridTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a MetaTerid checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--moe-backend",
        default="checkpoint",
        choices=("checkpoint", "auto", "grouped_mm", "padded", "sorted"),
    )
    parser.add_argument(
        "--model-param-dtype",
        default="checkpoint",
        choices=("checkpoint", "fp32", "bf16", "fp16"),
    )
    parser.add_argument(
        "--moe-param-dtype",
        default="auto",
        choices=("auto", "fp32", "bf16", "fp16"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = MetaTeridTokenizer(args.tokenizer)

    model, _ = load_metaterid_model(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        device=args.device,
        seq_len=args.seq_len,
        model_param_dtype=args.model_param_dtype,
        moe_param_dtype=args.moe_param_dtype,
        moe_backend=args.moe_backend,
    )

    prompt = args.prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            n_loops=args.n_loops,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
