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

## Source

The tokenizer core is implemented in Rust. The Python package calls the same native core through PyO3.

The released 32K surface vocabulary is `assets/surface-vocab.bin`. The large precompiled surface-analysis lookup table is only a throughput accelerator and is not required for correct tokenization, so it is not included here.

## License

Apache-2.0. See `THIRD_PARTY_LICENSES/ZEMBEREK_NOTICE.md` for the morphology resource notice.
