#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tokenizers as tokenizers_pkg
import tiktoken

SEGMENT_MAGIC = b"NSEG0001"
SURFACE_MORPH_MAGIC = b"NDSRF002"
SURFACE_LEXICAL_MAGIC = b"NDSRF003"
SURFACE_FIXED_IDS = 259
BYTE_COUNT = 256
DEFAULT_TOTAL_VOCAB = 32_000
DEFAULT_MAX_TOKEN_BYTES = 96


def parse_args():
    p = argparse.ArgumentParser(description="Train NedoTokenizer morphology-constrained byte BPE")
    p.add_argument("--input", required=True, help="NSEG0001 segmented training corpus")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-vocab", type=int, default=DEFAULT_TOTAL_VOCAB)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-token-bytes", type=int, default=DEFAULT_MAX_TOKEN_BYTES)
    p.add_argument("--expected-input-sha256")
    p.add_argument("--boundary-policy", choices=["morphology", "lexical"], default="morphology")
    return p.parse_args()


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def iter_segmented(path, stats=None):
    with open(path, "rb", buffering=8 << 20) as f:
        magic = f.read(8)
        if magic != SEGMENT_MAGIC:
            raise RuntimeError(f"bad segmented corpus magic: {magic!r}")
        while True:
            header = f.read(4)
            if not header:
                return
            if len(header) != 4:
                raise RuntimeError("truncated segmented record length")
            (n,) = struct.unpack("<I", header)
            data = f.read(n)
            if len(data) != n:
                raise RuntimeError("truncated segmented record")
            text = data.decode("utf-8")
            if stats is not None:
                stats["records"] += 1
                stats["segmented_bytes"] += n
                stats["sentinels"] += data.count(b"\x1f")
            yield text


def bytes_to_unicode():
    # GPT-2 / ByteLevel reversible byte alphabet.
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def bytelevel_symbol_decoder():
    return {symbol: byte for byte, symbol in bytes_to_unicode().items()}


def symbol_to_bytes(symbol, inverse):
    try:
        return bytes(inverse[ch] for ch in symbol)
    except KeyError as exc:
        raise RuntimeError(f"unexpected ByteLevel symbol {exc.args[0]!r} in {symbol!r}") from exc


def load_merge_bytes(merges_path):
    inverse = bytelevel_symbol_decoder()
    entries = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                raise RuntimeError(f"invalid merge line: {line!r}")
            left = symbol_to_bytes(parts[0], inverse)
            right = symbol_to_bytes(parts[1], inverse)
            entries.append(left + right)
    return entries


def validate_merge_order(entries, max_token_bytes):
    seen = set()
    for rank, entry in enumerate(entries):
        if len(entry) < 2 or len(entry) > max_token_bytes:
            raise RuntimeError(f"merge rank {rank} has invalid byte length {len(entry)}")
        if entry in seen:
            raise RuntimeError(f"duplicate merge entry at rank {rank}")
        valid = False
        for split in range(1, len(entry)):
            left, right = entry[:split], entry[split:]
            if (len(left) == 1 or left in seen) and (len(right) == 1 or right in seen):
                valid = True
                break
        if not valid:
            raise RuntimeError(f"merge rank {rank} cannot be formed from earlier ranks: {entry!r}")
        seen.add(entry)


def write_surface_vocab(path, entries, magic):
    payload = bytearray()
    for entry in entries:
        payload += struct.pack("<I", len(entry))
        payload += entry
    digest = hashlib.sha256(payload).digest()
    data = magic + struct.pack("<I", len(entries)) + digest + payload
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest(), len(data), digest.hex()


def write_tiktoken(path, entries):
    with open(path, "wb") as f:
        rank = 0
        for byte in range(256):
            token = bytes([byte])
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")
            rank += 1
        for entry in entries:
            f.write(base64.b64encode(entry) + b" " + str(rank).encode() + b"\n")
            rank += 1


def piece_stats(entries):
    result = {
        "learned_entries": len(entries),
        "valid_utf8_entries": 0,
        "invalid_utf8_entries": 0,
        "space_prefixed_entries": 0,
        "whitespace_only_entries": 0,
        "single_unicode_scalar_entries": 0,
        "emoji_scalar_entries": 0,
        "max_entry_bytes": 0,
        "mean_entry_bytes": 0.0,
    }
    total = 0
    for entry in entries:
        total += len(entry)
        result["max_entry_bytes"] = max(result["max_entry_bytes"], len(entry))
        if entry.startswith(b" "):
            result["space_prefixed_entries"] += 1
        if entry and all(b in b" \t\r\n\v\f" for b in entry):
            result["whitespace_only_entries"] += 1
        try:
            text = entry.decode("utf-8")
        except UnicodeDecodeError:
            result["invalid_utf8_entries"] += 1
            continue
        result["valid_utf8_entries"] += 1
        if len(text) == 1:
            result["single_unicode_scalar_entries"] += 1
            cp = ord(text)
            if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
                result["emoji_scalar_entries"] += 1
    result["mean_entry_bytes"] = total / max(1, len(entries))
    return result


def parity_smoke(entries):
    ranks = {bytes([b]): b for b in range(256)}
    for i, entry in enumerate(entries, 256):
        ranks[entry] = i
    enc = tiktoken.Encoding(
        name="nedo_surface_bpe_audit",
        pat_str=r"(?s).+",
        mergeable_ranks=ranks,
        special_tokens={},
    )
    probes = [
        "Bugün hava çok güzel.",
        "Evlerimizden çıktık.",
        "Российская Федерация",
        "中华人民共和国",
        "😀 👨‍👩‍👧‍👦",
        "        print(x)",
        "\n\n    indentation",
    ]
    out = []
    for text in probes:
        ids = enc.encode_ordinary(text)
        decoded = enc.decode_bytes(ids)
        if decoded != text.encode("utf-8"):
            raise RuntimeError(f"tiktoken parity smoke failed for {text!r}")
        out.append({"text": text, "tokens": len(ids)})
    return out


def main():
    args = parse_args()
    started = time.time()
    input_path = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_sha = sha256_file(input_path)
    if args.expected_input_sha256 and input_sha != args.expected_input_sha256:
        raise RuntimeError("segmented corpus SHA-256 mismatch")

    learned_target = args.total_vocab - SURFACE_FIXED_IDS
    if learned_target <= 0:
        raise RuntimeError("total vocab is too small for fixed IDs")
    # Hugging Face BPE can retain a small number of observed alphabet symbols
    # in addition to ByteLevel's fixed 256-symbol alphabet. Train a deterministic
    # rank tail beyond the final budget, then keep the exact prefix of merge ranks.
    # Prefix truncation is BPE-safe because every retained merge depends only on
    # raw bytes or earlier retained ranks.
    trainer_rank_margin = 1024
    trainer_vocab = BYTE_COUNT + learned_target + trainer_rank_margin

    model = models.BPE(unk_token=None)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split("\x1f", behavior="removed"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=trainer_vocab,
        min_frequency=args.min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=args.max_token_bytes,
    )

    corpus_stats = {"records": 0, "segmented_bytes": 0, "sentinels": 0}
    tokenizer.train_from_iterator(iter_segmented(input_path, corpus_stats), trainer=trainer)

    model_files = tokenizer.model.save(str(out), "hf-byte-bpe")
    tokenizer.save(str(out / "hf-tokenizer.json"))
    merges_path = next(Path(p) for p in model_files if p.endswith("merges.txt"))
    trained_entries = load_merge_bytes(merges_path)
    if len(trained_entries) < learned_target:
        raise RuntimeError(
            f"expected at least {learned_target} learned merges, got {len(trained_entries)}; "
            f"trainer vocabulary={tokenizer.get_vocab_size()}"
        )
    entries = trained_entries[:learned_target]
    validate_merge_order(entries, args.max_token_bytes)

    vocab_path = out / "surface-vocab-32k-nedo-bpe.bin"
    magic = SURFACE_MORPH_MAGIC if args.boundary_policy == "morphology" else SURFACE_LEXICAL_MAGIC
    vocab_sha, vocab_bytes, payload_sha = write_surface_vocab(vocab_path, entries, magic)
    tiktoken_path = out / "surface-vocab-32k-nedo-bpe.tiktoken"
    write_tiktoken(tiktoken_path, entries)
    merge_sha = sha256_file(merges_path)
    tokenizer_sha = sha256_file(out / "hf-tokenizer.json")
    tiktoken_sha = sha256_file(tiktoken_path)

    manifest = {
        "schema": "nedo_surface_bpe_training_v1",
        "status": "PASS",
        "algorithm": f"{args.boundary_policy}-constrained byte-level BPE",
        "boundary_policy": args.boundary_policy,
        "training_backend": "huggingface-tokenizers BPE + ByteLevel(use_regex=false)",
        "boundary_sentinel_hex": "1f",
        "input": str(input_path),
        "input_sha256": input_sha,
        "corpus": corpus_stats,
        "total_embedding_vocab": args.total_vocab,
        "fixed_control_ids": 3,
        "fixed_byte_fallback_ids": 256,
        "learned_merge_entries": len(entries),
        "trainer_requested_vocab_size": trainer_vocab,
        "trainer_vocab_size": tokenizer.get_vocab_size(),
        "trainer_merge_entries": len(trained_entries),
        "retained_merge_entries": len(entries),
        "min_frequency": args.min_frequency,
        "max_token_bytes": args.max_token_bytes,
        "surface_vocab": str(vocab_path),
        "surface_vocab_sha256": vocab_sha,
        "surface_vocab_bytes": vocab_bytes,
        "surface_payload_sha256": payload_sha,
        "tiktoken_audit_file": str(tiktoken_path),
        "tiktoken_audit_sha256": tiktoken_sha,
        "hf_merges_sha256": merge_sha,
        "hf_tokenizer_sha256": tokenizer_sha,
        "tokenizers_version": tokenizers_pkg.__version__,
        "tiktoken_version": tiktoken.__version__,
        "piece_stats": piece_stats(entries),
        "roundtrip_smoke": parity_smoke(entries),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = out / "surface-vocab-32k-nedo-bpe.manifest.json"
    tmp = str(manifest_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
