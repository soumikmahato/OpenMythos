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

## 1. Install Tokenizer Dependencies

```bash
pip install -q tokenizers transformers datasets
```

## 2. Build The Corpus

The default corpus builder streams a weighted mix of:

| Source | Purpose |
|---|---|
| `fineweb_edu` | Diverse clean English educational prose |
| `openwebmath` | Math/STEM text and LaTeX syntax |
| `codeparrot` | Python/code syntax |
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
  --total-docs 500000 \
  --shards 32 \
  --max-chars 32768 \
  --report-every 10000 \
  --hard-exit
```

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

Optional multilingual coverage can be added with:

```bash
python training/prepare_metaterid_tokenizer_corpus.py \
  --sources fineweb_edu,openwebmath,codeparrot,openhermes,hermes_tools,fineweb2_de,synthetic_tools,synthetic_code,synthetic_math,synthetic_chat,synthetic_fim
```

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
  --output tokenizer_inspection.json
```

Check:

- Every required special token reports `single=True`.
- Code, LaTeX, tool JSON, chat, and FIM probes are not excessively fragmented.
- Common factual and arithmetic strings are not pathological.
- The tokenizer can round-trip normal text and code.

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
