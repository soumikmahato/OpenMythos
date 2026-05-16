#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from open_mythos.metaterid import MetaTeridForCausalLM, metaterid_t4_pilot
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = MetaTeridTokenizer(args.tokenizer)

    cfg = metaterid_t4_pilot()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = args.seq_len

    model = MetaTeridForCausalLM(cfg).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

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
