# MetaTerid Tokenizer Training Guide

This guide trains `metaterid-tokenizer-v1`, the byte-level BPE tokenizer used by
MetaTerid main training. Train the tokenizer before any serious model run,
because the model embedding/head shape depends on the tokenizer vocabulary.

## Goal

- Vocabulary: `65,536`
- Algorithm: byte-level BPE
- Format: HuggingFace `PreTrainedTokenizerFast`
- Required coverage: English prose, code, LaTeX/math, chat turns, tool calls,
  tool results, hidden-thinking tags, answer tags, and fill-in-the-middle code.
- Required special tokens remain single tokens:
  `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`,
  `<|assistant|>`, `<|tool|>`, `<|tool_call|>`, `<|tool_result|>`,
  `<|think|>`, `<|end_think|>`, `<|answer|>`, `<|fim_prefix|>`,
  `<|fim_middle|>`, `<|fim_suffix|>`, `<|eot|>`.
- Future reserved tokens: `<|reserved_special_00|>` through
  `<|reserved_special_49|>` must also remain single tokens.
- V1 scope: English-first. Keep math symbols and LaTeX/code punctuation, but
  defer broad non-Latin tokenizer coverage to a future extension.

## 1. Install Tokenizer Dependencies

```bash
pip install -q tokenizers transformers datasets
```

## 2. Build The Corpus

The default corpus builder streams a weighted mix of:

| Source | Purpose |
|---|---|
| `fineweb_edu` | Diverse clean English educational prose |
| `ultrafineweb_l3_qa_en` | English L3 synthetic QA/refined web coverage |
| `ultrafineweb_l3_multistyle_en` | English multi-style rewritten knowledge |
| `openwebmath` | Math/STEM text and LaTeX syntax |
| `ultradata_math_l3_qa` | Math QA syntax and explanations |
| `ultradata_math_l3_textbook` | Textbook/exercise math formatting |
| `ultradata_sft_math` | SFT math turns |
| `ultradata_sft_code` | SFT code turns |
| `ultradata_sft_if` | Instruction-following turns |
| `codeparrot` | Python/code syntax |
| `the_stack_python`, `the_stack_markdown`, `the_stack_tex`, `the_stack_javascript`, `the_stack_sql` | Gated code/markdown/TeX slices with artifact filtering |
| `openhermes` | Instruction/chat formatting |
| `hermes_tools` | Function/tool-call formatting |
| `synthetic_tools` | Guaranteed MetaTerid tool tokens and JSON calls |
| `synthetic_code` | Dense code snippets across Python, JS, SQL, HTML |
| `synthetic_math` | Arithmetic, algebra, and LaTeX patterns |
| `synthetic_chat` | MetaTerid chat/thinking/answer tags |
| `synthetic_fim` | Fill-in-the-middle tokens and code spans |

Recommended full tokenizer corpus:

```bash
python training/prepare_metaterid_tokenizer_corpus.py \
  --output-dir data/tokenizer_corpus \
  --total-docs 4000000 \
  --target-chars 12000000000 \
  --shards 256 \
  --max-chars 32768 \
  --report-every 25000 \
  --hard-exit
```

`--target-chars 12000000000` is the planning proxy for at least about 3B
pre-tokenizer tokens. Increase to `15000000000` if the final tokenizer
inspection shows high fertility on code/math.

Fast smoke corpus:

```bash
python training/prepare_metaterid_tokenizer_corpus.py \
  --output-dir data/tokenizer_corpus_smoke \
  --sources synthetic_tools,synthetic_code,synthetic_math,synthetic_chat,synthetic_fim \
  --total-docs 2000 \
  --shards 2 \
  --report-every 500 \
  --hard-exit
```

The corpus builder filters tokenizer-polluting artifacts before training:
long whitespace runs, banner separator lines, long dash/equal/star/hash runs,
separator-dominated samples, low-alphanumeric samples, repeated exact texts,
and non-English/non-Latin-heavy samples.

## 3. Train The Tokenizer

```bash
python training/train_metaterid_tokenizer.py "data/tokenizer_corpus/*.txt" \
  --output-dir tokenizers/metaterid-tokenizer-v1 \
  --vocab-size 65536 \
  --min-frequency 2
```

For the smoke corpus only, use a small vocab so it finishes quickly:

```bash
python training/train_metaterid_tokenizer.py "data/tokenizer_corpus_smoke/*.txt" \
  --output-dir tokenizers/metaterid-tokenizer-smoke \
  --vocab-size 8192 \
  --min-frequency 1
```

## 4. Inspect The Tokenizer

```bash
python training/inspect_metaterid_tokenizer.py \
  --tokenizer tokenizers/metaterid-tokenizer-v1 \
  --output tokenizer_inspection.json \
  --fail-on-pollution
```

Check:

- Every required special token reports `single=True`.
- Code, LaTeX, tool JSON, chat, and FIM probes are not excessively fragmented.
- Common factual and arithmetic strings are not pathological.
- Pollution probes report zero long whitespace/separator tokens.
- The tokenizer can round-trip normal text and code.

## Modal Production Run

The production Modal runner is intentionally ignored by git under
`modal_scripts/`. It trains the corpus, trains the tokenizer, runs inspection,
and pushes the final tokenizer repo:

```bash
HF_TOKEN=... MODAL_PROFILE=thecodevibesx \
  .venv/bin/modal run modal_scripts/metaterid_tokenizer_production.py \
  --target-chars 12000000000 \
  --total-docs 4000000 \
  --shards 256 \
  --repo-id metaterid-tokenizer-v1
```

Use `--smoke --no-push --target-chars 200000 --total-docs 200 --shards 4` for
a quick Modal validation. Do not commit `.env` or Modal launch scratch files.

## 5. Use It In Training

```bash
python training/metaterid_main_train.py \
  --tokenizer tokenizers/metaterid-tokenizer-v1 \
  --ckpt-dir checkpoints/metaterid_main \
  --target-tokens 100000000 \
  --mix final
```

Do not change the tokenizer after a model run starts unless you also restart
the model from scratch. A different tokenizer changes token IDs and embedding
shape, so old checkpoints are not compatible.
