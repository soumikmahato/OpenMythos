# MetaTerid Tokenizer

MetaTerid uses a custom byte-level BPE tokenizer rather than the current
OpenMythos default `openai/gpt-oss-20b` tokenizer.

## Target

- Name: `metaterid-tokenizer-v1`
- Target vocabulary size: `65,536`, including reserved special tokens
- Algorithm: byte-level BPE
- Normalization: minimal and lossless; preserve code, whitespace, markup,
  math, and non-English text
- Compatibility: HuggingFace `PreTrainedTokenizerFast`

This size is the default for MetaTerid 1B because it keeps embedding/head
parameters reasonable while still handling English, code, math, reasoning
formats, and selected multilingual data with acceptable compression.

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

## Training

Train from local corpus text files, directories of `.txt` files, or glob
patterns:

```bash
python training/train_metaterid_tokenizer.py data/**/*.txt \
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
