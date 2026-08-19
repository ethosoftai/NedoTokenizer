__version__: str

from typing import TypedDict

class AssetInfo(TypedDict):
    schema_version: int
    morphology_sha256: str
    model_sha256: str
    runtime: str
    python_hot_path: bool
    compiled_surface_table_supported: bool
    nedoformer_supported: bool
    nedoformer_contract_version: int
    nedoformer_lattice_schema_version: int
    nedoformer_input_encoding_version: int
    nedoformer_sidecar_supported: bool
    nedoformer_sidecar_schema_version: int

class Tokenizer:
    def __init__(
        self,
        mode: str = "auto",
        max_sentence_tokens: int = 512,
        max_fallback_chars: int = 48,
        contextual_disambiguation: bool = True,
        detect_unmarked_code: bool = True,
    ) -> None: ...
    def tokenize_batch(self, documents: list[bytes], threads: int = 1) -> list[bytes]: ...
    @staticmethod
    def decode_batch(documents: list[bytes]) -> list[bytes]: ...
    def roundtrip_batch(self, documents: list[bytes], threads: int = 1) -> bool: ...

class NedoFormerInputEncoding(TypedDict):
    ids: list[int]
    segment_offsets: list[int]
    pooled_segments: list[int]
    pool_spans: list[tuple[int, int]]
    pool_modes: list[str]
    pool_group_ids: list[int | None]

class NedoFormerTokenizer:
    def __init__(
        self,
        mode: str = "auto",
        max_sentence_tokens: int = 512,
        max_fallback_chars: int = 48,
        contextual_disambiguation: bool = True,
        detect_unmarked_code: bool = True,
        character_vocabulary: bytes | None = None,
        generation_vocabulary: bytes | None = None,
        compiled_analysis_table: bytes | None = None,
    ) -> None: ...
    def lattice(self, document: bytes) -> bytes: ...
    @staticmethod
    def lattice_metadata_json(lattice: bytes) -> str: ...
    def lattice_batch(self, documents: list[bytes], threads: int = 1) -> list[bytes]: ...
    def lattice_sidecar(self, document: bytes) -> bytes: ...
    def lattice_sidecar_batch(self, documents: list[bytes], threads: int = 1) -> list[bytes]: ...
    def input_encoding(
        self,
        document: bytes,
        policy: str = "best",
        seed: int = 0,
        temperature: float = 1.0,
    ) -> NedoFormerInputEncoding: ...
    def input_encoding_from_sidecar(
        self,
        document: bytes,
        sidecar: bytes,
        policy: str = "best",
        seed: int = 0,
        temperature: float = 1.0,
    ) -> NedoFormerInputEncoding: ...
    @staticmethod
    def sample_lattice(
        lattice: bytes,
        policy: str = "best",
        seed: int = 0,
        temperature: float = 1.0,
    ) -> bytes: ...
    def train_assets(
        self,
        documents: list[bytes],
        max_chars: int = 500,
        max_roots: int = 16000,
        max_code_pieces: int = 4096,
    ) -> tuple[bytes, bytes, str]: ...
    def generation_ids(self, document: bytes) -> list[int]: ...
    def generation_ids_from_lattice(
        self,
        lattice: bytes,
        policy: str = "best",
        seed: int = 0,
        temperature: float = 1.0,
    ) -> list[int]: ...
    def generation_decode(self, ids: list[int]) -> bytes: ...
    def contract_fingerprint(self) -> str: ...

class SurfaceTokenizer:
    def __init__(
        self,
        vocabulary: bytes,
        mode: str = "auto",
        max_sentence_tokens: int = 512,
        max_fallback_chars: int = 48,
        contextual_disambiguation: bool = True,
        detect_unmarked_code: bool = True,
        analysis_table: bytes | None = None,
    ) -> None: ...
    def inspect_json(self, document: bytes) -> str: ...
    def encode_ids(self, document: bytes) -> list[int]: ...
    def encode_ids_batch(self, documents: list[bytes], threads: int = 1) -> list[list[int]]: ...
    def clear_runtime_caches(self) -> None: ...
    def runtime_cache_stats(self, threads: int) -> dict[str, int]: ...
    def decode_ids(self, ids: list[int]) -> bytes: ...
    def vocabulary_size(self) -> int: ...

def asset_info() -> AssetInfo: ...
