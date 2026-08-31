# NedoTokenizer

Byte-exact, morphology-aware tokenizer for Turkish and mixed Turkish/code text.

## Python

```bash
pip install .
```

```python
from pathlib import Path
from nedotokenizer import SurfaceTokenizer

vocab = Path("assets/surface-vocab.bin").read_bytes()
tokenizer = SurfaceTokenizer(vocab)

text = "Evlerimizden çıktık.".encode("utf-8")
ids = tokenizer.encode_ids(text)
assert tokenizer.decode_ids(ids) == text
print(ids)
```

## Whitespace and Unicode

The surface encoder is byte-exact and has no `UNK` token. IDs `3..258` are a complete one-byte fallback, so every UTF-8 sequence (including Cyrillic, CJK, emoji, and unseen scripts) remains representable and decodes exactly. The trainer also observes frequent non-ASCII Unicode scalars as learned candidates to reduce byte fallback on scripts that occur in training data.

Ordinary single ASCII inter-word spaces are prefix-bridgeable at the surface-vocabulary layer: a learned piece may be `" dünya"` or `" Rusya"` instead of spending a separate token on `" "`. The bridge may enter only the following word/number's first segment; morphology cuts remain hard boundaries. Multi-space runs, tabs, line breaks, controls, and opaque bytes remain structurally exact and independently representable. A vocabulary must be trained with the prefix-bridge-aware trainer before such prefixed entries can appear; the bundled v0.2.0 asset predates this training policy.

## Source

The tokenizer core is implemented in Rust. The Python package calls the same native core through PyO3.

The released 32K surface vocabulary is `assets/surface-vocab.bin`. The large precompiled surface-analysis lookup table is only a throughput accelerator and is not required for correct tokenization, so it is not included here.

## License

Apache-2.0. See `THIRD_PARTY_LICENSES/ZEMBEREK_NOTICE.md` for the morphology resource notice.
