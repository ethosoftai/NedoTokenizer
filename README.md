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

The surface encoder is byte-exact and has no `UNK` token. IDs `3..258` are a complete one-byte fallback, so every UTF-8 sequence (including Cyrillic, CJK, emoji, and unseen scripts) remains representable and decodes exactly. Byte-BPE training promotes frequent UTF-8 byte sequences into learned merge tokens automatically, so common non-ASCII characters and longer multilingual/emoji sequences can become compact without sacrificing coverage.

Ordinary single ASCII inter-word spaces are prefix-bridgeable at the surface-vocabulary layer: a learned piece may be `" dünya"` or `" Rusya"` instead of spending a separate token on `" "`. Multi-space runs, tabs, line breaks, controls, and opaque bytes remain structurally exact and independently representable.

## Byte-BPE v3

The tokenizer can load three explicitly versioned surface-vocabulary contracts: legacy greedy-longest (`NDSRF001`), morphology-constrained byte BPE (`NDSRF002`), and lexical-boundary byte BPE (`NDSRF003`). The BPE formats use learned-entry order as merge rank while retaining the fixed 256 exact-byte IDs, so there is still no `UNK`. In the selected lexical policy, morphology remains available as analysis metadata but does not force final LM token boundaries. See `docs/byte-bpe-v3.md` for the ablation and acceptance gates.

## Source

The tokenizer core is implemented in Rust. The Python package calls the same native core through PyO3.

The released 32K surface vocabulary is `assets/surface-vocab.bin`. The large precompiled surface-analysis lookup table is only a throughput accelerator and is not required for correct tokenization, so it is not included here.

## License

Apache-2.0. See `THIRD_PARTY_LICENSES/ZEMBEREK_NOTICE.md` for the morphology resource notice.
