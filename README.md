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

The tokenizer can load four explicitly versioned surface-vocabulary contracts: legacy greedy-longest (`NDSRF001`), hard-all morphology Byte-BPE (`NDSRF002`), lexical-boundary Byte-BPE (`NDSRF003`), and root-hard/suffix-soft Byte-BPE (`NDSRF004`). The BPE formats use learned-entry order as merge rank while retaining the fixed 256 exact-byte IDs, so there is still no `UNK`. `NDSRF004` is the production policy. See `docs/byte-bpe-v3.md` for the ablation and acceptance gates.

## Source

The tokenizer core is implemented in Rust. The Python package calls the same native core through PyO3.

The released 32K surface vocabulary is `assets/surface-vocab.bin`. The current production vocabulary was trained from a source-aware 20 GB MercanSet V11 sample (`min_frequency=100`, maximum learned piece length 48 bytes) and uses root-hard/suffix-soft Byte-BPE (`NDSRF004`). For a morphologically analyzed Turkish word, only the first morphology cut is a hard LM-token boundary: `gel | di | m` is exposed to BPE as `gel | dim`, while `ev | ler | imiz | den` is exposed as `ev | lerimizden`. BPE may still split either side further if the vocabulary does not contain a larger learned piece. One ordinary ASCII inter-word space may prefix the root, e.g. `[ gel] [dim]`. No word-specific exceptions are used.

## License

Apache-2.0. See `THIRD_PARTY_LICENSES/ZEMBEREK_NOTICE.md` for the morphology resource notice.
