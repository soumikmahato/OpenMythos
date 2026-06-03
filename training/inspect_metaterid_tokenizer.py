#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

try:
    from transformers import PreTrainedTokenizerFast
except ImportError as exc:
    raise ImportError(
        "Tokenizer inspection requires `transformers`. "
        "Install tokenizer dependencies with `pip install tokenizers transformers datasets`."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_MODULE_PATH = ROOT / "open_mythos" / "metaterid_tokenizer.py"
spec = importlib.util.spec_from_file_location("_metaterid_tokenizer", TOKENIZER_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {TOKENIZER_MODULE_PATH}")
_tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_tokenizer_module)

METATERID_SPECIAL_TOKENS = _tokenizer_module.METATERID_SPECIAL_TOKENS


PROBE_TEXTS = {
    "special": [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|think|>",
        "<|end_think|>",
        "<|answer|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|eot|>",
    ],
    "english": [
        "the",
        "The",
        "and",
        "information",
        "communication",
        "artificial intelligence",
        "The capital of France is Paris.",
        "Photosynthesis is the process by which plants convert light into chemical energy.",
    ],
    "math": [
        "2 + 2 = 4",
        "2 * 2 = 4",
        "10 * 10 = 100",
        "10 * 0 = 0",
        "x^2 + y^2 = z^2",
        "∑_{i=1}^{n} i = n(n+1)/2",
        "3.14159",
    ],
    "latex_math": [
        r"Let $a,b \in \mathbb{R}$. Then $(a+b)^2 = a^2 + 2ab + b^2$.",
        r"\begin{align} y &= mx + b \\ \Delta &= b^2 - 4ac \end{align}",
        r"The loss is \(\mathcal{L} = -\sum_i y_i \log p_i\).",
        r"```latex\n\frac{\partial \mathcal{L}}{\partial \theta} = \nabla_\theta J(\theta)\n```",
    ],
    "code": [
        "print('hello world')",
        "def add(a, b):\n    return a + b",
        "const x = await fetch(url);",
        "<html><body>Hello</body></html>",
        "self.assertEqual(result, expected)",
        "CREATE TABLE runs (id BIGINT PRIMARY KEY, status TEXT NOT NULL);",
    ],
    "tool": [
        "<|tool_call|>{\"name\":\"web_search\",\"arguments\":{\"query\":\"weather\"}}<|eot|>",
        "<|tool_result|>{\"results\":[{\"title\":\"Example\",\"url\":\"https://example.com\"}]}<|eot|>",
    ],
    "chat": [
        "<|system|>You are MetaTerid.<|user|>Explain briefly.<|assistant|><|answer|>Done.<|eot|>",
        "<|user|>Think briefly, then answer.<|assistant|><|think|>Short reasoning.<|end_think|><|answer|>Final.<|eot|>",
    ],
    "fim": [
        "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\nprint(add(2, 3))<|fim_middle|>return a + b",
    ],
}

POLLUTED_TOKEN_RE = re.compile(
    r"([ \t])\1{7,}|([\-_=*#~])\2{7,}|^[\-_=*#~\s]{10,}$"
)


def _load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(path)


def _format_tokens(tokens: list[str], max_tokens: int) -> str:
    shown = tokens[:max_tokens]
    suffix = "" if len(tokens) <= max_tokens else f" ... (+{len(tokens) - max_tokens})"
    return " ".join(repr(token) for token in shown) + suffix


def _decoded_vocab_token(tok: PreTrainedTokenizerFast, token_id: int, raw_token: str) -> str:
    try:
        decoded = tok.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        decoded = tok.decode([token_id])
    except Exception:
        decoded = ""
    return decoded if decoded else raw_token


def _pollution_report(
    tok: PreTrainedTokenizerFast,
    id_to_token: dict[int, str],
    *,
    max_examples: int,
) -> dict:
    polluted: list[dict] = []
    punctuation_heavy: list[dict] = []
    whitespace_tokens: list[dict] = []

    for token_id, raw_token in sorted(id_to_token.items()):
        decoded = _decoded_vocab_token(tok, token_id, raw_token)
        visible = decoded.strip()
        if POLLUTED_TOKEN_RE.search(decoded):
            polluted.append({"id": token_id, "token": raw_token, "decoded": decoded})
        if decoded and len(decoded) >= 10 and not visible:
            whitespace_tokens.append({"id": token_id, "token": raw_token, "decoded": decoded})
        if (
            len(visible) >= 12
            and sum(ch in "-_=*#~" for ch in visible) / max(1, len(visible)) > 0.7
        ):
            punctuation_heavy.append({"id": token_id, "token": raw_token, "decoded": decoded})

    return {
        "polluted_count": len(polluted),
        "punctuation_heavy_count": len(punctuation_heavy),
        "whitespace_token_count": len(whitespace_tokens),
        "examples": polluted[:max_examples],
        "punctuation_heavy_examples": punctuation_heavy[:max_examples],
        "whitespace_examples": whitespace_tokens[:max_examples],
    }


def inspect_tokenizer(path: Path, *, max_tokens: int = 40) -> dict:
    tok = _load_tokenizer(path)
    vocab = tok.get_vocab()
    id_to_token = {idx: token for token, idx in vocab.items()}

    print(f"Tokenizer path: {path}")
    print(f"Vocab size: {len(tok):,}")
    print()

    print("Special tokens")
    print("-" * 80)
    missing = []
    special_report = {}
    for token in METATERID_SPECIAL_TOKENS:
        token_id = tok.convert_tokens_to_ids(token)
        encoded = tok.encode(token, add_special_tokens=False)
        ok = token_id is not None and token_id >= 0 and encoded == [token_id]
        if not ok:
            missing.append(token)
        special_report[token] = {"id": token_id, "encoded": encoded, "single_token": ok}
        print(f"{token:18s} id={str(token_id):>6s} encoded={encoded} single={ok}")
    print()

    print("Probe tokenization")
    print("-" * 80)
    fertility_rows = []
    for group, texts in PROBE_TEXTS.items():
        print(f"[{group}]")
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False)
            tokens = tok.convert_ids_to_tokens(ids)
            chars = max(1, len(text))
            fertility = len(ids) / chars
            fertility_rows.append((group, text, len(ids), len(text), fertility))
            print(f"text: {text!r}")
            print(f"ids:  {ids[:max_tokens]}{' ...' if len(ids) > max_tokens else ''}")
            print(f"tok:  {_format_tokens(tokens, max_tokens)}")
            print(f"fertility tokens/char={fertility:.3f} tokens={len(ids)} chars={len(text)}")
            print()

    print("Vocabulary probes")
    print("-" * 80)
    wanted = [
        " Paris",
        " France",
        " India",
        " Cancer",
        " Photosynthesis",
        " Insulin",
        " artificial",
        " intelligence",
        " communication",
        " python",
        " Python",
        " html",
        " HTML",
        " 0",
        " 1",
        " 2",
        " 4",
        " 10",
        " 100",
        "+",
        "*",
        "=",
        " =",
        " *",
        " +",
        "self",
        "assertEqual",
        "web_search",
    ]
    vocab_hits = {}
    for token in wanted:
        token_id = vocab.get(token)
        vocab_hits[token] = token_id
        print(f"{token!r:20s} -> {token_id}")
    print()

    print("Lowest token IDs")
    print("-" * 80)
    for idx in range(min(80, len(id_to_token))):
        print(f"{idx:6d} {id_to_token[idx]!r}")
    print()

    pollution = _pollution_report(tok, id_to_token, max_examples=max_tokens)
    print("Pollution probes")
    print("-" * 80)
    print(f"polluted_count={pollution['polluted_count']}")
    print(f"punctuation_heavy_count={pollution['punctuation_heavy_count']}")
    print(f"whitespace_token_count={pollution['whitespace_token_count']}")
    for item in pollution["examples"]:
        print(f"id={item['id']:6d} token={item['token']!r} decoded={item['decoded']!r}")
    print()

    group_summary = {}
    for group, _text, n_tokens, n_chars, fertility in fertility_rows:
        group_summary.setdefault(group, []).append(fertility)
    print("Fertility summary")
    print("-" * 80)
    for group, values in group_summary.items():
        avg = sum(values) / len(values)
        print(f"{group:10s} avg_tokens_per_char={avg:.3f}")

    return {
        "path": str(path),
        "vocab_size": len(tok),
        "missing_special_tokens": missing,
        "special_tokens": special_report,
        "vocab_hits": vocab_hits,
        "pollution": pollution,
        "fertility": [
            {
                "group": group,
                "text": text,
                "tokens": n_tokens,
                "chars": n_chars,
                "tokens_per_char": fertility,
            }
            for group, text, n_tokens, n_chars, fertility in fertility_rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the MetaTerid tokenizer.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument(
        "--fail-on-pollution",
        action="store_true",
        help="Exit non-zero if long whitespace/separator tokens entered the vocabulary.",
    )
    args = parser.parse_args()

    report = inspect_tokenizer(Path(args.tokenizer), max_tokens=args.max_tokens)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.fail_on_pollution:
        pollution = report["pollution"]
        if (
            pollution["polluted_count"]
            or pollution["punctuation_heavy_count"]
            or pollution["whitespace_token_count"]
        ):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
