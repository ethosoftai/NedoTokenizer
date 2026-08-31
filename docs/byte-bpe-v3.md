# NedoTokenizer 32K Byte-BPE v3 Design

## Goal

The LM-facing surface vocabulary must be byte-exact, `UNK`-free, efficient on Turkish natural language, and robust to multilingual UTF-8, emoji, code, whitespace, and arbitrary bytes. Morphological analysis remains part of NedoTokenizer, but its cuts are evaluated as a tokenizer design choice rather than assumed to be optimal LM-token boundaries.

The fixed 32K ID layout stays:

- `0..2`: PAD/BOS/EOS
- `3..258`: all 256 exact byte atoms/fallback IDs
- `259..31999`: 31,741 learned byte-BPE merge ranks

Any input is therefore representable even if none of its multi-byte sequences were seen during vocabulary training.

## Industry references

The design follows the same broad byte-level BPE family used by OpenAI Tiktoken and Mistral Tekken. Mistral documents Tekken as Tiktoken-based and reports improved multilingual/code compression relative to its earlier SentencePiece tokenizer. Meta reports that Llama 3 moved to a 128K tokenizer specifically to improve language encoding efficiency. These are design references, not claims of implementation identity.

- OpenAI Tiktoken: https://github.com/openai/tiktoken
- Mistral Tekken / NeMo: https://mistral.ai/news/mistral-nemo/
- Meta Llama 3: https://ai.meta.com/blog/meta-llama-3/

## Segmentation variants evaluated

1. **Legacy GreedyLongest** — released v0.2 surface vocabulary.
2. **Morphology Byte-BPE (`NDSRF002`)** — BPE rank merges, but Turkish morphology cuts remain hard LM-token boundaries. One ordinary ASCII inter-word space can prefix the following first segment.
3. **Lexical Byte-BPE (`NDSRF003`)** — BPE rank merges at scanner/lexical boundaries. Morphological analyses and cuts remain available as metadata but no longer force LM-token boundaries. One ordinary ASCII space can prefix the following word/number.
4. **Root-hard / suffix-soft Byte-BPE (`NDSRF004`)** — the first Turkish morphology cut stays hard (`root | suffix-chain`), while later suffix cuts are soft and may be merged by corpus-frequency BPE. One ordinary ASCII space may prefix the root. This is the production policy.
5. **Raw GPT-style 32K baseline** — Hugging Face ByteLevel BPE with its GPT-2-style regex pre-tokenization; used only as a compression reference.

The stable binary header identifies the segmentation algorithm, so an old greedy asset cannot silently be interpreted as a BPE asset.

## Deterministic balanced ablation

A 125,775,453-text-byte pilot was deterministically sampled from the existing 2 GB MercanSet V11 source-aware vocabulary sample, targeting about 5 MB from each of all 30 sources where available. The split seed was `0x4e45444f42504531`; records were assigned by `doc_key`, with approximately 5% held out.

Held-out set:

- 20,997 records
- 7,080,072 UTF-8 text bytes
- all 30 MercanSet source IDs represented

Results:

| Tokenizer | Tokens | Bytes/token | Token reduction vs legacy |
|---|---:|---:|---:|
| Legacy Nedo 32K | 3,092,678 | 2.2893 | — |
| Morphology Byte-BPE 32K | 2,195,435 | 3.2249 | 29.0% |
| **Lexical Byte-BPE 32K** | **1,823,968** | **3.8817** | **41.0%** |
| Raw GPT-style BPE 32K reference | 1,745,894 | 4.0553 | 43.5% |
| OpenAI `cl100k_base` reference | 2,628,523 | 2.6936 | 15.0% |
| OpenAI `o200k_base` reference | 2,128,889 | 3.3257 | 31.2% |

The OpenAI rows are not size-matched (100K/200K vs 32K) and this balanced MercanSet subset is Turkish/domain-heavy; they are reference measurements, not universal tokenizer rankings.

Lexical Byte-BPE is the strongest compression ablation, but it can absorb a Turkish root together with its suffixes. Hard-all morphology prevents that but also forces every suffix boundary to spend a separate BPE segment. The selected compromise is **Root-hard / suffix-soft Byte-BPE (`NDSRF004`)**: `gel | di | m` becomes the two BPE regions `gel | dim`, and `ev | ler | imiz | den` becomes `ev | lerimizden`. These are allowed regions, not hard-coded output tokens: BPE may split them further according to the learned 32K vocabulary. There are no word-specific exceptions.

## Training implementation

`crates/nedo-vocab` produces deterministic morphology/lexical segmented training streams and deterministic `doc_key` held-out data. `tools/train_surface_bpe.py` trains ByteLevel BPE with a full 256-byte initial alphabet, converts merge ranks back to exact raw byte strings, validates rank dependencies, truncates only a merge-rank tail to the exact 31,741 learned-entry budget, writes the checksum-protected Nedo surface asset, and emits a Tiktoken-compatible audit rank file plus a JSON manifest.

The production trainer intentionally does not reserve one dedicated ID for every Unicode scalar. That cannot fit in 32K. Instead:

- every UTF-8 byte is always representable by the 256 byte atoms;
- frequent UTF-8 byte sequences naturally become learned BPE tokens;
- frequent Cyrillic, CJK, Arabic, emoji, ZWJ sequences, etc. can therefore collapse to one or a few tokens;
- unseen Unicode and even invalid UTF-8 remain exact rather than becoming `UNK`.

## Acceptance gates for a production asset

A final asset is accepted only if all of the following pass:

- exact total vocabulary size = 32,000;
- learned merge entries = 31,741;
- stable BPE binary reload and hash parity;
- BPE merge-order validation;
- byte-exact roundtrip for Turkish, multilingual UTF-8, emoji/ZWJ, code, whitespace, and arbitrary invalid bytes;
- rich/reference and flat production encoder parity;
- deterministic train/eval split and input/output SHA-256 manifests;
- held-out compression benchmark against the released 32K asset;
- no silent fallback to a different segmentation algorithm.

## Final 2 GB root-hard / suffix-soft release run

The production asset was trained on the full deterministic 2,001,326,142-byte MercanSet V11 source-aware sample (2,744,278 records, 30 sources), with a deterministic `doc_key` 95/5 split. Held-out evaluation contains 137,059 records / 98,844,686 bytes.

| 32K policy | Held-out tokens | Bytes/token |
|---|---:|---:|
| Hard-all morphology (`NDSRF002`) | 30,001,281 | 3.294682 |
| **Root-hard / suffix-soft (`NDSRF004`)** | **26,922,024** | **3.671518** |
| Lexical BPE ablation (`NDSRF003`) | 24,782,411 | 3.988502 |

The selected policy uses about 10.3% fewer tokens than hard-all morphology while retaining the root/suffix-chain boundary as a hard invariant. The lexical ablation compresses further, but removes that invariant.

Full held-out acceptance additionally requires rich/reference IDs to equal production/flat IDs. On all 137,059 held-out documents: mismatches = 0, byte-roundtrip failures = 0, required root-boundary violations = 0.

Production asset SHA-256: `c4475789f014a425729562d46ea0d016206fe2da8b6f9e9f9f805e427ee509d7`.
