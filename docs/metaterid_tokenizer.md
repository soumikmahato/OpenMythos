# MetaTerid Tokenizer

MetaTerid uses a custom byte-level BPE tokenizer rather than the current
OpenMythos default `openai/gpt-oss-20b` tokenizer.

## Target

- Name: `metaterid-tokenizer-v1`
- Target vocabulary size: `65,536`, including reserved special tokens
- Algorithm: byte-level BPE
- Normalization: minimal and mostly lossless; preserve code, markup, math,
  LaTeX, markdown, and tool syntax while filtering corpus artifacts before BPE
  training
- Compatibility: HuggingFace `PreTrainedTokenizerFast`

This size is the default for MetaTerid 1B because it keeps embedding/head
parameters reasonable while still handling English, code, math, reasoning
formats, markdown, and agentic tool traces with acceptable compression. The v1
production tokenizer is English-first; non-Latin expansion is reserved for a
future tokenizer update.

## Reserved Tokens

The tokenizer reserves chat, thinking, tool, and fill-in-the-middle tokens:

```text
<|pad|>
<|unk|>
<|bos|>
<|eos|>
<|system|>
<|user|>
<|assistant|>
<|tool|>
<|tool_call|>
<|tool_result|>
<|think|>
<|end_think|>
<|answer|>
<|fim_prefix|>
<|fim_middle|>
<|fim_suffix|>
<|eot|>
```

It also reserves `<|reserved_special_00|>` through
`<|reserved_special_49|>` for future protocol, modality, or training-control
tokens.

## Corpus

Before training the BPE model, build a diverse English-first corpus that
explicitly includes prose, code, math/LaTeX, markdown, chat-format text, tool
JSON, and FIM examples:

```bash
python training/prepare_metaterid_tokenizer_corpus.py \
  --output-dir data/tokenizer_corpus \
  --total-docs 4000000 \
  --target-chars 12000000000 \
  --shards 256 \
  --max-chars 32768 \
  --hard-exit
```

The default corpus mix uses FineWeb-Edu, Ultra-FineWeb-L3 English QA and
Multi-Style data, OpenWebMath, UltraData-Math, UltraData-SFT Math/Code/IF,
CodeParrot, gated The Stack language slices, OpenHermes/Hermes tool examples,
and synthetic MetaTerid format examples. The builder filters long whitespace,
separator banners, repeated dash/equal/star/hash lines, low-alphanumeric
samples, and non-English/non-Latin-heavy samples before writing shards.

## Training

Train from local corpus text files, directories of `.txt` files, or glob
patterns:

```bash
python training/train_metaterid_tokenizer.py "data/tokenizer_corpus/*.txt" \
  --output-dir tokenizers/metaterid-tokenizer-v1 \
  --vocab-size 65536 \
  --min-frequency 2
```

The output directory is directly loadable with:

```python
from open_mythos import MetaTeridTokenizer

tok = MetaTeridTokenizer("tokenizers/metaterid-tokenizer-v1")
ids = tok.encode("<|user|>What is 2+2?<|assistant|><|think|>")
text = tok.decode(ids)
```

## Notes

- Keep all reserved tokens as single tokens in every trained tokenizer.
- Do not use lossy lowercasing or Unicode stripping.
- Before full pretraining, evaluate fertility on the actual training mixture:
  English prose, code, math, tool traces, reasoning traces, and selected
  multilingual samples.

Run the built-in inspection script after training:

```bash
python training/inspect_metaterid_tokenizer.py \
  --tokenizer tokenizers/metaterid-tokenizer-v1 \
  --output tokenizer_inspection.json \
  --fail-on-pollution
```

See [`../TOKENIZER_TRAINING_GUIDE.md`](../TOKENIZER_TRAINING_GUIDE.md) for the
step-by-step workflow.
